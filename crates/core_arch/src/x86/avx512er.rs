//! Exponential and Reciprocal Instructions (ER)
//!
//! The reference is [Intel 64 and IA-32 Architectures Software Developer's
//! Manual Volume 2: Instruction Set Reference, A-Z][intel64_ref] (p. 4-241).
//!
//! [intel64_ref]: http://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf

use crate::core_arch::simd::f64x8;

use crate::core_arch::x86::{__m512d, _mm512_setzero_pd, m512dExt};
use core::mem::transmute;

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.exp2.pd"]
    fn vexp2pd(src: f64x8, a: f64x8, m: i8, sae: i32) -> f64x8;
}

#[inline]
#[target_feature(enable = "avx512er,avx512dq")]
#[cfg_attr(test, assert_instr(vexp2pd))]
pub unsafe fn _mm512_exp2a23_pd(a: __m512d) -> __m512d {
    transmute(vexp2pd(
        a.as_f64x8(),
        _mm512_setzero_pd().as_f64x8(),
        -1_i8,
        4_i32,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "avx512er,avx512dq")]
    unsafe fn test_mm512_exp2a23_pd() {
        #[rustfmt::skip]
        let d = _mm512_set_pd(
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0
        );
        #[rustfmt::skip]
        let exp = _mm512_set_pd(
            1.0,
            4.0,
            9.0,
            16.0,
            25.0,
            36.0,
            49.0,
            64.0
        );
        let res = _mm512_exp2a23_pd(d);
        assert_eq_m512d(res, exp);
    }
}
