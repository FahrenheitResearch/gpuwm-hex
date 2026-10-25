"""Fast Thompson phases, with cell-local processes parallel over (level, cell).

The source statements come from the column transcription. Column reductions
keep their vertical order. Sedimentation (fall speeds and sub-steps) runs one
warp per column (see _sediment_warp). Strict composition uses the unchanged
column kernel in thompson_v842.
"""
import hashlib
import re


# Boundaries are native module_mp_thompson.F lines, each at column scope.
BOUNDARIES = (1513, 1951, 2022, 2572, 2674, 2931, 3229, 3550)
# Native line where the fall speeds are known and the sub-stepped fluxes begin
# (module_mp_thompson.F:3409-3549, rain, cloud, ice, snow, graupel).
SEDIMENT_SPLIT = 3409
METHOD_CELL_PHASES = frozenset((0, 2, 4, 5, 6, 7, 8, 9, 11, 13))
#: Phase 12 is sedimentation (module_mp_thompson.F:3229-3549), one 32-lane warp per column.
WARP_PHASE = 12
#: Phases 3 and 10 are the graupel intercept's running minimum down the column
#: (module_mp_thompson.F:1912-1940 and 2992-3020), also one warp per column.
WARP_PHASES = frozenset((3, 10, WARP_PHASE))
#: Phase 14 is the driver commit; it calls no process method.
TAIL_PHASE = 14
# Consecutive level-local phases share one slab invocation. Prefix scans and
# the early-exit reduction remain ordered column operations.
PHASE_GROUPS = ((0,), (1,), (2,), (3,), (4, 5, 6, 7, 8, 9),
                (10,), (11,), (12,), (13,), (14,))
CELL_PHASES = frozenset(i for i, group in enumerate(PHASE_GROUPS)
                       if all(phase in METHOD_CELL_PHASES for phase in group))
WARP_GROUPS = frozenset(i for i, group in enumerate(PHASE_GROUPS) if set(group) <= WARP_PHASES)
KERNELS = ("mpas_thompson_step",) + tuple(
    "mpas_thompson_phase_" + str(i) for i in range(1, len(PHASE_GROUPS)))
CARRY_BYTES_PER_COLUMN = 20  # no_micro plus four precipitation accumulators
FUSED_METHODS = (4, 5, 6, 7, 8, 9)


def _localize_process_rates(declarations, pieces):
    """Keep the conservation rates inside the fused cell method.

    A field qualifies only if its sole initialization use is the exact zero
    store, its later uses are exactly ice, conservation and state update,
    or warm-rain and state update, and every index is the current cell.
    The warm-rain set adds six doubles without the 22 extra ice rates
    that spilled in the wider prototype.
    Preserve its declared element type and all expressions.
    Workspace slots stay reserved so other kernels keep their existing layout.
    """
    index = '0+tc_sub(k,(kts))*(1)'
    fused = ''.join(pieces[i] for i in FUSED_METHODS)
    fields = []
    for match in re.finditer(r'TcCol<(float|double|bool)> (\w+)=ws\.\w+\(\d+\);', declarations):
        kind, name = match.groups()
        token = r'\b' + re.escape(name) + r'\b'
        uses = {i for i, piece in enumerate(pieces) if re.search(token, piece)}
        if uses not in ({0, 5, 8}, {0, 6, 7, 8}):
            continue
        zero = f'{name}[{index}] = 0.f;\n'
        if pieces[0].count(zero) != 1 or len(re.findall(token, pieces[0])) != 1:
            continue
        references = re.findall(token + r'\[([^\]]+)\]', fused)
        if not references or set(references) != {index}:
            raise ValueError('Thompson cell process rate has a nonlocal index: ' + name)
        if len(references) != len(re.findall(token, fused)):
            raise ValueError('Thompson cell process rate escapes as a view: ' + name)
        pieces[0] = pieces[0].replace(zero, '', 1)
        fused = fused.replace(f'{name}[{index}]', f'tc_cell_{name}')
        declarations = declarations.replace(match.group(), '', 1)
        fields.append((kind, name))
    if len(fields) != 23:
        raise ValueError(f'Thompson cell process rate live set changed: {len(fields)}')
    initializers = ''.join(f'{kind} tc_cell_{name}=0.f;\n' for kind, name in fields)
    pieces[4] = initializers + fused
    return declarations, pieces, fields


def _defer_zero_stores(declarations, pieces):
    """Zero a workspace array in the first cell phase that uses it, not in initialization.

    Initialization zeroes 60-odd rate and tendency planes per cell that later phases fill;
    across that kernel boundary every plane makes a round trip to DRAM (the tile is far larger
    than L2). An array qualifies when its only appearance in initialization is the zero store
    of the current cell and its first later user is a cell phase (same thread, same cell);
    the store moves to the start of that phase unchanged. No phase between them reads it.
    """
    index = '0+tc_sub(k,(kts))*(1)'
    moved = {}
    for match in re.finditer(r'TcCol<(?:float|double)> (\w+)=ws\.\w+\(\d+\);', declarations):
        name = match.group(1)
        token = re.compile(r'\b' + re.escape(name) + r'\b')
        stores = re.findall(r'^' + re.escape(name + '[' + index + ']') + r' = 0(?:\.f|\.0f|\.0|\.)?;\n',
                            pieces[0], re.M)
        if len(stores) != 1 or len(token.findall(pieces[0])) != 1:
            continue
        users = [i for i in range(1, len(pieces)) if i not in FUSED_METHODS[1:] and token.search(pieces[i])]
        if not users or users[0] not in METHOD_CELL_PHASES:
            continue
        pieces[0] = pieces[0].replace(stores[0], '', 1)
        moved.setdefault(users[0], []).append(stores[0])
    for phase, stores in moved.items():
        pieces[phase] = 'k=phase_k;\n' + ''.join(stores) + pieces[phase]
    return pieces, sum(len(s) for s in moved.values())


def _marker(line):
    return "// MPAS module_mp_thompson.F:" + str(line) + "\n"


def _localize_initial_temporaries(declarations, pieces):
    """Keep initialization-only values in their original element types.

    Every reference must belong to this cell and this method. Preserve all
    assignments, branches and expressions; only their storage changes.
    Reserved workspace slots keep every other phase's layout unchanged.
    """
    index = '0+tc_sub(k,(kts))*(1)'
    expected = {'satw': 'float', 'sati': 'float', 'l_nifa': 'bool'}
    fields = []
    for match in re.finditer(r'TcCol<(float|double|bool)> (\w+)=ws\.\w+\(\d+\);', declarations):
        kind, name = match.groups()
        if name not in expected:
            continue
        token = r'\b' + re.escape(name) + r'\b'
        uses = {i for i, piece in enumerate(pieces) if re.search(token, piece)}
        refs = re.findall(token + r'\[([^\]]+)\]', pieces[0])
        if (kind != expected[name] or uses != {0} or not refs
                or set(refs) != {index}
                or len(refs) != len(re.findall(token, pieces[0]))):
            raise ValueError('Thompson initialization temporary escapes its cell: ' + name)
        pieces[0] = re.sub(token + r'\[' + re.escape(index) + r'\]',
                           f'tc_init_{name}', pieces[0])
        declarations = declarations.replace(match.group(), '', 1)
        fields.append((kind, name))
    if dict((name, kind) for kind, name in fields) != expected:
        raise ValueError('Thompson initialization temporary live set changed')
    pieces[0] = ''.join(f'{kind} tc_init_{name};\n' for kind, name in fields) + pieces[0]
    return declarations, pieces


#: sha256 of the native sub-step text (module_mp_thompson.F:3409-3549) that SEDIMENT_WARP
#: restates. A changed transcription is refused rather than silently diverging.
SEDIMENT_SOURCE_SHA256 = "5bb0e1e05c75f11050c55c08b16c981caa5f6517c641dcca0d34d2ea0fde3c2d"

_SEDIMENT_WARP_SUPPORT = r'''
// Sub-stepped sedimentation, one warp per column (fast numerics; MPAS module_mp_thompson.F:3409-3549).
// Lane l owns levels kts+l+32*i. Within one sub-step every flux sed(k)=vt(k)*q(k) is formed
// before any level is updated, so the levels of a sub-step are independent: the warp updates
// them together and takes the flux from the level above by shuffle. Each level keeps the native
// statements and operand order, and only the levels the native loops touch are updated (the top
// level, then ksed1 down to kts). The species of a column no longer wait on the slowest column
// of a 32-column warp, and the sub-step state stays in registers.
#define TC_SED_PER_LANE ((THOMPSON_COLUMN_KMAX+31)/32)
__device__ __forceinline__ float tc_sed_above(const float (&s)[TC_SED_PER_LANE],int i,int lane) {
    const float a=__shfl_down_sync(0xffffffffu,s[i],1);
    const float b=__shfl_sync(0xffffffffu,i+1<TC_SED_PER_LANE?s[i+1<TC_SED_PER_LANE?i+1:i]:0.0f,0);
    return lane==31?b:a;
}
template<bool HasNumber>
__device__ __forceinline__ void tc_sed_warp(TcCol<float> vt,TcCol<float> vtn,TcCol<float> q,TcCol<float> qn,
    TcCol<float> qten,TcCol<float> nten,const float (&odzq)[TC_SED_PER_LANE],const float (&orho)[TC_SED_PER_LANE],
    int lane,int kts,int kte,int ksed1,float onstep,float dt,float qmin,float nmin,float& ppt) {
    const int nl=kte-kts+1;
    const int nstep=((int)round(tc_div(1.f,onstep)));
    float r[TC_SED_PER_LANE],n[TC_SED_PER_LANE],v[TC_SED_PER_LANE],vn[TC_SED_PER_LANE],qt[TC_SED_PER_LANE],nt[TC_SED_PER_LANE];
    #pragma unroll
    for(int i=0;i<TC_SED_PER_LANE;i++) {
        const int j=lane+32*i;
        r[i]=0.0f;n[i]=0.0f;v[i]=0.0f;vn[i]=0.0f;qt[i]=0.0f;nt[i]=0.0f;
        if(j<nl) {r[i]=q[j];v[i]=vt[j];qt[i]=qten[j];if(HasNumber){n[i]=qn[j];vn[i]=vtn[j];nt[i]=nten[j];}}
    }
    for(int step=1;step<=nstep;step+=1) {
        float sr[TC_SED_PER_LANE],sn[TC_SED_PER_LANE];
        #pragma unroll
        for(int i=0;i<TC_SED_PER_LANE;i++) {sr[i]=tc_mul(v[i],r[i]);sn[i]=HasNumber?tc_mul(vn[i],n[i]):0.0f;}
        #pragma unroll
        for(int i=0;i<TC_SED_PER_LANE;i++) {
            const float ur=tc_sed_above(sr,i,lane);
            const float un=HasNumber?tc_sed_above(sn,i,lane):0.0f;
            const int j=lane+32*i;const int k=kts+j;
            if(k==kte) {
                qt[i]=tc_sub(qt[i],tc_mul(tc_mul(tc_mul(sr[i],odzq[i]),onstep),orho[i]));
                if(HasNumber) nt[i]=tc_sub(nt[i],tc_mul(tc_mul(tc_mul(sn[i],odzq[i]),onstep),orho[i]));
                r[i]=tc_max(qmin,tc_sub(r[i],tc_mul(tc_mul(tc_mul(sr[i],odzq[i]),dt),onstep)));
                if(HasNumber) n[i]=tc_max(nmin,tc_sub(n[i],tc_mul(tc_mul(tc_mul(sn[i],odzq[i]),dt),onstep)));
            } else if(j<nl&&k<=ksed1) {
                qt[i]=tc_add(qt[i],tc_mul(tc_mul(tc_mul((tc_sub(ur,sr[i])),odzq[i]),onstep),orho[i]));
                if(HasNumber) nt[i]=tc_add(nt[i],tc_mul(tc_mul(tc_mul((tc_sub(un,sn[i])),odzq[i]),onstep),orho[i]));
                r[i]=tc_max(qmin,tc_add(r[i],tc_mul(tc_mul(tc_mul((tc_sub(ur,sr[i])),odzq[i]),dt),onstep)));
                if(HasNumber) n[i]=tc_max(nmin,tc_add(n[i],tc_mul(tc_mul(tc_mul((tc_sub(un,sn[i])),odzq[i]),dt),onstep)));
            }
        }
        const float r0=__shfl_sync(0xffffffffu,r[0],0),s0=__shfl_sync(0xffffffffu,sr[0],0);
        if((r0>tc_mul(qmin,1000.f))) { ppt=tc_add(ppt,tc_mul(tc_mul(s0,dt),onstep)); }
    }
    #pragma unroll
    for(int i=0;i<TC_SED_PER_LANE;i++) {
        const int j=lane+32*i;
        if(j<nl) {q[j]=r[i];qten[j]=qt[i];if(HasNumber){qn[j]=n[i];nten[j]=nt[i];}}
    }
}
// Fall-speed pass helpers (MPAS module_mp_thompson.F:3229-3401). The native loops walk down the
// column; a level whose species is absent takes the fall speed of the level above (X(k)=X(k+1)),
// and level kte+1 holds the zero the native loop starts from. src is, for each level, the nearest
// level at or above it that computed its own value; the copy is exact. Each lane owns levels
// lane+32*i, as in the sub-steps.
__device__ __forceinline__ void tc_sed_chain_src(const bool (&cp)[TC_SED_PER_LANE],int (&src)[TC_SED_PER_LANE],int lane,int nl) {
    int carry=nl;
    #pragma unroll
    for(int i=TC_SED_PER_LANE-1;i>=0;i--) {
        const int j=lane+32*i;
        int s=(j<nl&&cp[i])?0x7fffffff:j;
        #pragma unroll
        for(int off=1;off<32;off<<=1) {const int o=__shfl_down_sync(0xffffffffu,s,off);if(lane+off<32)s=min(s,o);}
        s=min(s,carry);
        src[i]=s;
        carry=__shfl_sync(0xffffffffu,s,0);
    }
}
__device__ __forceinline__ void tc_sed_chain_copy(TcCol<float> x,const bool (&cp)[TC_SED_PER_LANE],const int (&src)[TC_SED_PER_LANE],int lane,int nl) {
    // sources are levels that computed their own value, so no level read here is written here
    #pragma unroll
    for(int i=0;i<TC_SED_PER_LANE;i++) {const int j=lane+32*i;if(j<nl&&cp[i])x[j]=x[src[i]];}
}
// The native running maxima (ksed1, nstep) are integer maxima, so a warp maximum is exact.
__device__ __forceinline__ int tc_warp_max(int v) {
    #pragma unroll
    for(int off=16;off>0;off>>=1) v=max(v,__shfl_xor_sync(0xffffffffu,v,off));
    return v;
}
// The graupel intercept's running minimum (module_mp_thompson.F:1912-1940), top down: level j
// takes the minimum of start and every level at or above it. Out-of-column levels hold start.
__device__ __forceinline__ void tc_warp_running_min(double (&v)[TC_SED_PER_LANE],double start,int lane,int nl) {
    double carry=start;
    #pragma unroll
    for(int i=TC_SED_PER_LANE-1;i>=0;i--) {
        const int j=lane+32*i;
        double s=j<nl?v[i]:start;
        #pragma unroll
        for(int off=1;off<32;off<<=1) {const double o=__shfl_down_sync(0xffffffffu,s,off);if(lane+off<32)s=tc_min(o,s);}
        s=tc_min(s,carry);
        v[i]=s;
        carry=__shfl_sync(0xffffffffu,s,0);
    }
}
__device__ __forceinline__ void tc_sed_cloud_warp(TcCol<float> vtc,TcCol<float> vtnc,TcCol<float> rc,TcCol<float> nc,
    TcCol<float> qcten,TcCol<float> ncten,const float (&odzq)[TC_SED_PER_LANE],const float (&orho)[TC_SED_PER_LANE],
    int lane,int kts,int kte,int ksed1,float dt,float qmin) {
    const int nl=kte-kts+1;
    float sc[TC_SED_PER_LANE],sn[TC_SED_PER_LANE];
    #pragma unroll
    for(int i=0;i<TC_SED_PER_LANE;i++) {
        const int j=lane+32*i;
        sc[i]=0.0f;sn[i]=0.0f;
        if(j<nl) {sc[i]=tc_mul(vtc[j],rc[j]);sn[i]=tc_mul(vtnc[j],nc[j]);}
    }
    #pragma unroll
    for(int i=0;i<TC_SED_PER_LANE;i++) {
        const float uc=tc_sed_above(sc,i,lane),un=tc_sed_above(sn,i,lane);
        const int j=lane+32*i;const int k=kts+j;
        if(j<nl&&k<=ksed1) {
            qcten[j]=tc_add(qcten[j],tc_mul(tc_mul((tc_sub(uc,sc[i])),odzq[i]),orho[i]));
            ncten[j]=tc_add(ncten[j],tc_mul(tc_mul((tc_sub(un,sn[i])),odzq[i]),orho[i]));
            rc[j]=tc_max(qmin,tc_add(rc[j],tc_mul(tc_mul((tc_sub(uc,sc[i])),odzq[i]),dt)));
            nc[j]=tc_max(10.f,tc_add(nc[j],tc_mul(tc_mul((tc_sub(un,sn[i])),odzq[i]),dt)));
        }
    }
}
'''

#: The warp form of the sub-steps (module_mp_thompson.F:3409-3549), species in native order.
SEDIMENT_WARP = r'''{
const int tcs_nl=kte-kts+1;
const unsigned tcs_all=0xffffffffu;
float tcs_odzq[TC_SED_PER_LANE],tcs_orho[TC_SED_PER_LANE];
bool tcs_qr=false,tcs_qc=false,tcs_qi=false,tcs_qs=false,tcs_qg=false;
#pragma unroll
for(int tcs_i=0;tcs_i<TC_SED_PER_LANE;tcs_i++) {
    const int j=tcs_lane+32*tcs_i;
    tcs_odzq[tcs_i]=0.0f;tcs_orho[tcs_i]=0.0f;
    if(j<tcs_nl) {
        tcs_odzq[tcs_i]=tc_div(1.f,dzq[j]);tcs_orho[tcs_i]=tc_div(1.f,rho[j]);
        tcs_qr=tcs_qr||l_qr[j];tcs_qc=tcs_qc||l_qc[j];tcs_qi=tcs_qi||l_qi[j];tcs_qs=tcs_qs||l_qs[j];tcs_qg=tcs_qg||l_qg[j];
    }
}
if(__any_sync(tcs_all,tcs_qr)) tc_sed_warp<true>(vtrk,vtnrk,rr,nr,qrten,nrten,tcs_odzq,tcs_orho,tcs_lane,kts,kte,ksed1[0],onstep[0],dt,r1,r2,pptrain);
if(__any_sync(tcs_all,tcs_qc)) tc_sed_cloud_warp(vtck,vtnck,rc,nc,qcten,ncten,tcs_odzq,tcs_orho,tcs_lane,kts,kte,ksed1[4],dt,r1);
if(__any_sync(tcs_all,tcs_qi)) tc_sed_warp<true>(vtik,vtnik,ri,ni,qiten,niten,tcs_odzq,tcs_orho,tcs_lane,kts,kte,ksed1[1],onstep[1],dt,r1,r2,pptice);
if(__any_sync(tcs_all,tcs_qs)) tc_sed_warp<false>(vtsk,vtsk,rs,rs,qsten,qsten,tcs_odzq,tcs_orho,tcs_lane,kts,kte,ksed1[2],onstep[2],dt,r1,r1,pptsnow);
if(__any_sync(tcs_all,tcs_qg)) tc_sed_warp<false>(vtgk,vtgk,rg,rg,qgten,qgten,tcs_odzq,tcs_orho,tcs_lane,kts,kte,ksed1[3],onstep[3],dt,r1,r1,pptgraul);
}
'''

_INDEX = "[0+tc_sub(k,(kts))*(1)]"
_LANE_LOOP = ("#pragma unroll\nfor(int tcs_i=0;tcs_i<TC_SED_PER_LANE;tcs_i++) {"
              "k=kts+tcs_lane+32*tcs_i;if(k>kte)continue;\n")
_INIT_HEADER = "for(k=tc_add(kte,1); k>=kts; k+=(-1)) {"
_CHAIN_HEADER = "for(k=kte; k>=kts; k+=(-1)) {"
_CLOUD_HEADER = "for(k=ksed1[0+tc_sub(5,(1))*(1)]; k>=kts; k+=(-1)) {"
_COPY = re.compile(r"^(\w+)\[0\+tc_sub\(k,\(kts\)\)\*\(1\)\] = \1\[0\+tc_sub\(tc_add\(k,1\),\(kts\)\)\*\(1\)\];$")


def sediment_source_digest(native: str) -> str:
    return hashlib.sha256(native.encode("utf-8")).hexdigest()


def _code_lines(text):
    return [line for line in text.splitlines() if line.strip() and not line.startswith("// MPAS ")]


def _top_level_ifs(body, body_end):
    """(start, brace, end, else_brace, else_end) of each if statement at the top of a loop body."""
    found, i = [], 0
    while i < len(body):
        if body.startswith("//", i):
            i = body.index("\n", i) + 1
            continue
        if body.startswith("if(", i) and (i == 0 or body[i - 1] == "\n"):
            brace = body.index("{", i)
            end = body_end(body, brace)
            rest = body[end:]
            else_brace = else_end = None
            if rest.startswith(" else {"):
                else_brace = end + len(" else ")
                else_end = body_end(body, else_brace)
            found.append((i, brace, end, else_brace, else_end))
            i = else_end or end
            continue
        if body[i] == "{":
            i = body_end(body, i)
            continue
        i += 1
    return found


def _chain_loop(body, body_end):
    """One species of the fall-speed pass, its levels in parallel, the chain and maxima exact."""
    ifs = _top_level_ifs(body, body_end)
    if len(ifs) < 2 or ifs[-1][3] is not None or ifs[-2][3] is None:
        raise ValueError("Thompson fall-speed loop: expected a compute/else-copy branch then a running maximum")
    chain, reduce_ = ifs[-2], ifs[-1]
    copies = [_COPY.match(line) for line in _code_lines(body[chain[3] + 1:chain[4] - 1])]
    if not copies or not all(copies):
        raise ValueError("Thompson fall-speed loop: the else branch is not a pure copy from the level above")
    arrays = [m.group(1) for m in copies]
    compute = body[:chain[3]] + "{\ntcs_cp[tcs_i]=true;\n}" + body[chain[4]:reduce_[0]]
    if _code_lines(body[reduce_[2]:]):
        raise ValueError("Thompson fall-speed loop: statements after the running maximum")
    first = _LANE_LOOP.replace("{k=", "{tcs_cp[tcs_i]=false;k=", 1)
    resolve = "tc_sed_chain_src(tcs_cp,tcs_src,tcs_lane,kte-kts+1);\n" + "".join(
        f"tc_sed_chain_copy({name},tcs_cp,tcs_src,tcs_lane,kte-kts+1);\n" for name in arrays)
    maxima = "nstep=tc_warp_max(nstep);for(int tcs_q=0;tcs_q<5;tcs_q++)ksed1[tcs_q]=tc_warp_max(ksed1[tcs_q]);\n"
    return (first + compute + "}\n__syncwarp();\n" + resolve + "__syncwarp();\n"
            + _LANE_LOOP + body[reduce_[0]:reduce_[2]] + "\n}\n" + maxima)


def _replace_loops(text, header, body_end, rewrite, expected):
    out, pos, count = [], 0, 0
    while True:
        at = text.find(header, pos)
        if at < 0:
            break
        brace = at + len(header) - 1
        end = body_end(text, brace)
        out.append(text[pos:at] + rewrite(text[brace + 1:end - 1]))
        pos, count = end, count + 1
    if count != expected:
        raise ValueError(f"Thompson fall-speed pass: {count} loops '{header}', expected {expected}")
    return "".join(out) + text[pos:]


_N0_RUNNING_MIN = "n0_min = tc_min(n0_exp,n0_min);\n"


def _graupel_prefix_warp(piece, body_end):
    """The graupel intercept pass (k_0, then n0_min running down the column), one warp per column.

    k_0 is an integer running maximum, taken across the warp. n0_min = MIN(n0_exp, n0_min) is a
    running minimum of clamped finite intercepts from the top down: every level's intercept is
    formed in parallel, the minimum over the levels at and above each level is a warp scan (min of
    finite values is exact in any order), then each level continues from n0_exp = n0_min with the
    native statements.
    """
    loops = []
    _replace_loops(piece, _CHAIN_HEADER, body_end, lambda body: loops.append(body) or "", 2)
    first, second = loops
    if _code_lines(first) != ["if((temp[0+tc_sub(k,(kts))*(1)]>=270.65f)) { k_0 = tc_max(k_0,k); }"]:
        raise ValueError("Thompson graupel pass: the k_0 search changed")
    if second.count(_N0_RUNNING_MIN) != 1 or "n0_min" in second.replace(_N0_RUNNING_MIN, "").replace("n0_exp = n0_min;", ""):
        raise ValueError("Thompson graupel pass: the n0_min running minimum changed")
    head, tail = second.split(_N0_RUNNING_MIN)
    bodies = iter((
        _LANE_LOOP + first + "\n}\nk_0=tc_warp_max(k_0);\n",
        "double tcs_n0[TC_SED_PER_LANE];\n" + _LANE_LOOP + head + "tcs_n0[tcs_i]=n0_exp;\n}\n"
        "tc_warp_running_min(tcs_n0,n0_min,tcs_lane,kte-kts+1);\n"
        + _LANE_LOOP + "n0_min=tcs_n0[tcs_i];\n" + tail + "\n}\nn0_min=__shfl_sync(0xffffffffu,tcs_n0[0],0);\n"))
    piece = _replace_loops(piece, _CHAIN_HEADER, body_end, lambda body: next(bodies), 2)
    return "const int tcs_lane=threadIdx.x&31;\n" + piece


def _sediment_warp(native, body_end):
    """Phase 12, module_mp_thompson.F:3229-3549, one warp per column.

    The fall-speed pass is the native text with its level loops spread over the lanes: the
    zeroing loop, the four compute/else-copy loops (rain, ice, snow, graupel: computed levels in
    parallel, then the copies resolved from the level above, then the running maxima of ksed1 and
    nstep taken across the warp) and the cloud loop. The cloud's 500 m search keeps its serial
    loop on every lane. The sub-steps are SEDIMENT_WARP, refused if their native text changed.
    """
    marker = _marker(SEDIMENT_SPLIT)
    fall, substeps = native[:native.index(marker)], native[native.index(marker):]
    if sediment_source_digest(substeps) != SEDIMENT_SOURCE_SHA256:
        raise ValueError("Thompson sedimentation transcription changed; SEDIMENT_WARP must be re-derived")
    fall = _replace_loops(fall, _INIT_HEADER, body_end,
                          lambda body: _LANE_LOOP.replace("TC_SED_PER_LANE;", "TC_SED_PER_LANE+1;", 1)
                          .replace("if(k>kte)", "if(k>kte+1)", 1) + body + "\n}\n__syncwarp();\n", 1)
    fall = _replace_loops(fall, _CHAIN_HEADER, body_end, lambda body: _chain_loop(body, body_end), 4)
    fall = _replace_loops(fall, _CLOUD_HEADER, body_end,
                          lambda body: _LANE_LOOP.replace("if(k>kte)", "if(k>ksed1[0+tc_sub(5,(1))*(1)])", 1)
                          + body + "\n}\n", 1)
    if "for(k=" in fall.replace("for(k=kts; k<=tc_sub(kte,1); k+=1) {", "", 1):
        raise ValueError("Thompson fall-speed pass: a level loop was left serial")
    head = ("const int tcs_lane=threadIdx.x&31;\n"
            "bool tcs_cp[TC_SED_PER_LANE];int tcs_src[TC_SED_PER_LANE];\n")
    return head + fall + "__syncwarp();\n" + SEDIMENT_WARP


def split_source(text, body_end):
    """Split a workspace-transformed TU, preserving each expression verbatim."""
    head = text.index("__device__ __forceinline__ void mp_thompson(")
    brace = text.index("{", head)
    end = body_end(text, brace)
    signature = text[head:brace].rstrip()
    body = text[brace + 1:end - 1]
    offsets = [body.index(_marker(n)) for n in BOUNDARIES] + [len(body)]
    declarations = body[:offsets[0]]
    original = [body[a:b] for a, b in zip(offsets, offsets[1:])]

    def part(piece, lo, hi=None):
        return piece[piece.index(_marker(lo)):piece.index(_marker(hi)) if hi else len(piece)]

    # Initialization and the snow/rain moments have no vertical dependency.
    # Keep only the graupel prefix minimum and its warm-level scan column serial.
    # The early-exit flag is a Boolean OR, not an arithmetic reduction. Derive
    # it after every cell has initialized its flags, without atomics or a race.
    early = ("no_micro=true;\nfor(k=kts;k<=kte;k+=1) {\n"
             "int j=k-kts;\n"
             "if(l_qc[j]||l_qi[j]||l_qr[j]||l_qs[j]||l_qg[j]||ssati[j]>0.0f)"
             "no_micro=false;\n}\n"
             "ws.carry[0]=no_micro?1.0f:0.0f;\n")
    pieces = [part(original[0], 1513, 1819), early,
              part(original[0], 1824, 1912) + "}\n",
              "if((!iiwarm)) {\n" + part(original[0], 1912, 1940),
              part(original[0], 1940), *original[1:5],
              part(original[5], 2931, 2992) + "}\n",
              "if((!iiwarm)) {\n" + part(original[5], 2992, 3020),
              part(original[5], 3020),
              _sediment_warp(original[6], body_end),
              original[7]]
    pieces[3], pieces[10] = _graupel_prefix_warp(pieces[3], body_end), _graupel_prefix_warp(pieces[10], body_end)
    declarations, pieces, _ = _localize_process_rates(declarations, pieces)
    declarations, pieces = _localize_initial_temporaries(declarations, pieces)
    pieces, _ = _defer_zero_stores(declarations, pieces)
    loop_counts = {0: (6, 0), 2: (1, 0), 4: (8, 1), 5: (1, 0),
                   6: (1, 0), 7: (1, 0), 8: (3, 0), 9: (2, 0),
                   11: (3, 1), 13: (2, 0)}
    methods = []
    for phase, piece in enumerate(pieces):
        if phase in FUSED_METHODS[1:]:
            continue
        if phase > 1:
            # These are the only scalar live-ins across the native loop boundaries.
            # Repeating their exact operations avoids a scalar-state memory round trip.
            piece = ("if(ws.carry[0]!=0.0f)return;\n"
                     "dtsave=dt;odt=tc_div(1.f,dt);odts=tc_div(1.f,dtsave);\n" + piece)
        if phase in METHOD_CELL_PHASES:
            piece, up = re.subn(r"for\(k=kts; k<=kte; k\+=1\)",
                                "for(k=phase_k; k<=phase_k; k+=1)", piece)
            piece, down = re.subn(r"for\(k=kte; k>=kts; k\+=\(-1\)\)",
                                  "for(k=phase_k; k>=phase_k; k+=(-1))", piece)
            if (up, down) != loop_counts[phase]:
                raise ValueError(f"Thompson phase {phase}: vertical loop count changed")
        methods.append(("if" if phase == 0 else "else if") +
                       f" constexpr(Phase=={phase}) {{\n" + piece + "\n}")
    signature = signature.replace("void mp_thompson(", "void mp_thompson_phase(")
    signature = "template<int Phase>\n" + signature[:-1] + ",int phase_k)"
    text = text[:head] + signature + "{\n" + declarations + "\n".join(methods) + "\n}" + text[end:]

    head = text.index('extern "C" __global__ void mpas_thompson_step(')
    brace = text.index("{", text.index(")", head))
    end = body_end(text, brace)
    signature = text[head:brace]
    body = text[brace + 1:end - 1]
    load = body.index("for(int k=0;k<nz;k++)")
    sums = body.index("float rain=0.0f,snow=0.0f,graupel=0.0f,ice=0.0f;")
    call = body.index("m.mp_thompson(", sums)
    commit = body.index("rainncv[col]=", call)
    prologue, inputs = body[:load], body[load:sums]
    arguments = body[call:commit].strip()
    if not arguments.endswith(");"):
        raise ValueError("Thompson driver call boundary changed")
    tail = body[commit:]
    kernels = []
    for group_index, (group, name) in enumerate(zip(PHASE_GROUPS, KERNELS)):
        prefix = prologue
        first = "int tc_local=blockIdx.x*blockDim.x+threadIdx.x;if(tc_local>=tc_n)return;int col=tc_col0+tc_local;"
        if prefix.count(first) != 1:
            raise ValueError("Thompson workspace prologue changed")
        if group_index in CELL_PHASES:
            index = ("int tc_index=blockIdx.x*blockDim.x+threadIdx.x;"
                     "int tc_local=tc_index%tc_n;int tc_k=tc_index/tc_n+1;"
                     "if(tc_k>nz)return;int col=tc_col0+tc_local;")
        elif group_index in WARP_GROUPS:
            # One warp per column; whole warps leave together, so every lane reaches each shuffle.
            index = ("int tc_index=blockIdx.x*blockDim.x+threadIdx.x;"
                     "int tc_local=tc_index>>5;if(tc_local>=tc_n)return;"
                     "int col=tc_col0+tc_local;int tc_k=0;")
        else:
            index = first + "int tc_k=0;"
        prefix = prefix.replace(first, index, 1)
        cell_inputs = inputs.replace("for(int k=0;k<nz;k++)",
                                     "for(int k=tc_k-1;k<tc_k;k++)")
        content = prefix + (cell_inputs if group == (0,) else "")
        if group == (TAIL_PHASE,):
            content += ("float rain=m.ws.carry[32],snow=m.ws.carry[64],"
                        "graupel=m.ws.carry[96],ice=m.ws.carry[128];\n" + tail)
        else:
            content += "float rain=0.0f,snow=0.0f,graupel=0.0f,ice=0.0f;\n"
            for phase in ((4,) if group == FUSED_METHODS else group):
                phase_call = arguments.replace("m.mp_thompson(", f"m.mp_thompson_phase<{phase}>(", 1)
                content += phase_call[:-2] + ",tc_k);\n"
            if group == (WARP_PHASE,):
                # every lane holds the same sums; lane 0 stores them
                content += ("if((threadIdx.x&31)==0){m.ws.carry[32]=rain;m.ws.carry[64]=snow;"
                            "m.ws.carry[96]=graupel;m.ws.carry[128]=ice;}\n")
        # Actual cu12 uses 102/104 registers in the fused processes and serial
        # sedimentation at four blocks/SM. Five blocks makes the compiler
        # budget match the <=96-register target without changing statements.
        # The warp phases are latency bound (one column a warp, shuffles between levels): twelve
        # resident blocks take sedimentation from 9.4 to 5.3 ms on the zoom globe at dt 90.
        min_blocks = 5 if group == (4, 5, 6, 7, 8, 9) else 12 if group_index in WARP_GROUPS else 4
        bounded_signature = signature.replace('__global__ void',
            f'__global__ __launch_bounds__(128, {min_blocks}) void')
        kernels.append(bounded_signature.replace("mpas_thompson_step(", name + "(", 1) +
                       "{\n" + content + "\n}")
    text = text[:head] + "\n".join(kernels) + text[end:]
    marker = "struct ThompsonColumn {"
    if text.count(marker) != 1:
        raise ValueError("Thompson column struct not found exactly once")
    text = text.replace(marker, _SEDIMENT_WARP_SUPPORT + marker, 1)
    # Five compact slab-interleaved scalars, outside the per-level array planes.
    text = text.replace("double* d; float* f; bool* b;",
                        "double* d; float* f; bool* b; float* carry;", 1)
    old = "(size_t)(local/TC_WS_LANES)*plane*((size_t)nd*8+(size_t)nf*4+(size_t)nb)"
    new = "(size_t)(local/TC_WS_LANES)*(plane*((size_t)nd*8+(size_t)nf*4+(size_t)nb)+" + str(CARRY_BYTES_PER_COLUMN) + "*TC_WS_LANES)"
    if text.count(old) != 1:
        raise ValueError("Thompson workspace stride changed")
    text = text.replace(old, new, 1)
    text = text.replace("w.b=(bool*)(block+plane*nd*8+plane*nf*4)+lane;",
                        "w.b=(bool*)(block+plane*nd*8+plane*nf*4)+lane;"
                        "w.carry=(float*)(block+plane*(nd*8+nf*4+nb))+lane;", 1)
    return _level_major_workspace(text)


def _level_major_workspace(text):
    """Level-major planes within each aligned 128-column block slab.

    A compile-time pitch keeps field/level offsets immediate, rather than
    retaining a separate dynamic address for every live scratch array. All
    four warps of a block access adjacent columns in the same level plane.
    The capacity rounds to 128 columns, including the partial final slab.
    """
    if text.count('#define TC_WS_LANES 32') != 1:
        raise ValueError('Thompson workspace lane definition changed')
    text = text.replace('#define TC_WS_LANES 32', '#define TC_WS_LANES 128', 1)
    # Replace all offsets in one pass so the old 32 -> 128 replacement cannot
    # be mistaken for the old 128 -> 512 offset on a later replacement.
    return re.sub(r'm\.ws\.carry\[(32|64|96|128)\]',
                  lambda match: f'm.ws.carry[{4*int(match[1])}]', text)
