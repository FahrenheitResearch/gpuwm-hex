//! Native-source upper damping. ABI arrays use level-major contiguous storage.
macro_rules! operators {
    ($t:ty,$cam:ident,$ray:ident) => {
        #[no_mangle]
        pub unsafe extern "C" fn $cam(nlev:usize,ncols:usize,levels:usize,length:$t,coefficient:$t,values:*mut $t)->i32 {
            if levels==0 || levels>nlev || !length.is_finite() || length<=0.0 || !coefficient.is_finite() || coefficient<0.0 {return 1;}
            let Some(size)=nlev.checked_mul(ncols) else{return 1;};
            if values.is_null(){return 1;}
            let data=std::slice::from_raw_parts_mut(values,size);
            // mpas_atm_time_integration.F:6401-6403, native association.
            let base=((4.0 as $t * 2.0833 as $t)*length)*coefficient;
            for k in nlev-levels..nlev {
                let floor=base*(1.0 as $t - (nlev-1-k) as $t / levels as $t);
                for cell in 0..ncols {let index=k*ncols+cell;data[index]=data[index].max(floor);}
            }
            0
        }
        #[no_mangle]
        pub unsafe extern "C" fn $ray(nlev:usize,ncols:usize,solved:usize,levels:usize,days:$t,tendency:*mut $t,rho:*const $t,velocity:*const $t)->i32 {
            if levels==0 || levels>nlev || solved>ncols || !days.is_finite() || days<=0.0 {return 1;}
            let Some(size)=nlev.checked_mul(ncols) else{return 1;};
            if tendency.is_null() || rho.is_null() || velocity.is_null(){return 1;}
            let result=std::slice::from_raw_parts_mut(tendency,size);
            let density=std::slice::from_raw_parts(rho,size);let wind=std::slice::from_raw_parts(velocity,size);
            // F:6610-6627. Damping precedes the saved Euler addition.
            let inverse=1.0 as $t/(levels as $t*(days*86400.0 as $t));
            for k in nlev-levels..nlev {
                let coefficient=(k-(nlev-levels)+1) as $t*inverse;
                for edge in 0..solved {let index=k*ncols+edge;result[index]=result[index]-(density[index]*wind[index])*coefficient;}
            }
            0
        }
    }
}
operators!(f32,upper_cam_f32,upper_rayleigh_f32);
operators!(f64,upper_cam_f64,upper_rayleigh_f64);
