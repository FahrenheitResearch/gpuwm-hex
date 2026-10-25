"""KISS jump-ahead used by the per-g-point McICA kernels (fix group G2).

mpas_lw_mcica_gpoint / mpas_sw_mcica_gpoint start each g-point from the state
the sequential native stream reaches there. This checks the jump arithmetic
(the CUDA helpers' exact integer algorithm, written out here) against stepping
the stream, including non-canonical multiply-with-carry words and both fixed
points. The device words themselves are checked by the native column replays.
"""
import random

from hexcore.mpasphys import rrtmg_lw_v842 as lw

M32 = 0xFFFFFFFF


def _kiss(s):
    s[0] = (69069 * s[0] + 1327217885) & M32
    s[1] ^= (s[1] << 13) & M32
    s[1] ^= s[1] >> 17
    s[1] ^= (s[1] << 5) & M32
    s[2] = (18000 * (s[2] & 65535) + (s[2] >> 16)) & M32
    s[3] = (30903 * (s[3] & 65535) + (s[3] >> 16)) & M32


def _lcg_jump(s, n):  # mpas_lcg_jump
    ra, rc, pa, pc = 1, 0, 69069, 1327217885
    while n:
        if n & 1:
            rc = (pa * rc + pc) & M32
            ra = (pa * ra) & M32
        pc = (pa * pc + pc) & M32
        pa = (pa * pa) & M32
        n >>= 1
    return (ra * s + rc) & M32


def _mwc_jump(s, a, n):  # mpas_mwc_jump
    m = a * 65536 - 1
    while n and s > m:
        s = (a * (s & 65535) + (s >> 16)) & M32
        n -= 1
    if not n or s == m or s == 0:
        return s
    r, base = 1, a
    while n:
        if n & 1:
            r = r * base % m
        base = base * base % m
        n >>= 1
    return s * r % m


def _xorshift_power(n):
    identity = [1 << i for i in range(32)]
    power, base = identity, lw._xorshift_step_columns()
    while n:
        if n & 1:
            power = lw._xorshift_compose(base, power)
        base = lw._xorshift_compose(base, base)
        n >>= 1
    return power


def test_jump_matches_the_sequential_stream():
    rng = random.Random(20261024)
    edge2 = [0, 18000 * 65536 - 1, 18000 * 65536, M32, 65535 * 65536 + 65535]
    edge3 = [0, 30903 * 65536 - 1, 30903 * 65536, M32]
    for trial in range(200):
        start = [rng.getrandbits(32) for _ in range(4)]
        if trial % 5 == 0:
            start[2] = edge2[trial % len(edge2)]
            start[3] = edge3[trial % len(edge3)]
        n = rng.choice([0, 1, 2, 3, 4, 57, 114, 139 * 57, 139 * 114, rng.randrange(40000)])
        stepped = list(start)
        for _ in range(n):
            _kiss(stepped)
        jumped = [_lcg_jump(start[0], n), lw._xorshift_apply(_xorshift_power(n), start[1]),
                  _mwc_jump(start[2], 18000, n), _mwc_jump(start[3], 30903, n)]
        assert jumped == stepped, (start, n)


def test_draws_per_gpoint_follow_the_overlap_draw_order():
    assert [lw.mcica_draws_per_gpoint(o, 57) for o in (1, 2, 3, 4, 5)] == [57, 57, 1, 114, 114]
