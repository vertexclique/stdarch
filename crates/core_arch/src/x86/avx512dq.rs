use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem::{self, transmute},
    ptr,
};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Compute the bitwise AND of packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_and_pd&expand=100,288)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandpd))]
pub unsafe fn _mm512_and_pd(a: __m512d, b: __m512d) -> __m512d {
    transmute(simd_and(a.as_f64x8(), b.as_f64x8()))
}

/// Compute the bitwise XOR of packed double-precision (64-bit) floating-point elements in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_xor_pd&expand=100,6160)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorpd))]
pub unsafe fn _mm512_xor_pd(a: __m512d, b: __m512d) -> __m512d {
    transmute(simd_xor(a.as_f64x8(), b.as_f64x8()))
}

/// Compute the bitwise NOT of packed double-precision (64-bit) floating-point elements in a and then AND with b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_andnot_pd&expand=100,326)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnpd))]
pub unsafe fn _mm512_andnot_pd(a: __m512d, b: __m512d) -> __m512d {
    _mm512_and_pd(_mm512_xor_pd(a, _mm512_set1_pd(f64::MAX)), b)
}

/// Compute the bitwise NOT of packed double-precision (64-bit) floating-point elements in a and then AND with b, and
/// store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_andnot_pd&expand=100,327)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnpd))]
pub unsafe fn _mm512_mask_andnot_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
) -> __m512d {
    let andnot = _mm512_andnot_pd(a, b).as_f64x8();
    transmute(simd_select_bitmask(k, andnot, src.as_f64x8()))
}

/// Compute the bitwise NOT of packed double-precision (64-bit) floating-point elements in a and then AND with b, and
/// store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_andnot_pd&expand=100,328)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnpd))]
pub unsafe fn _mm512_maskz_andnot_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    let andnot = _mm512_andnot_pd(a, b).as_f64x8();
    let zero = _mm512_setzero_pd().as_f64x8();
    transmute(simd_select_bitmask(k, andnot, zero))
}

/// Compute the bitwise AND of packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_and_ps&expand=100,297)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandps))]
pub unsafe fn _mm512_and_ps(a: __m512, b: __m512) -> __m512 {
    transmute(simd_and(a.as_f32x16(), b.as_f32x16()))
}

/// Compute the bitwise XOR of packed single-precision (32-bit) floating-point elements in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_xor_ps&expand=100,6169)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm512_xor_ps(a: __m512, b: __m512) -> __m512 {
    transmute(simd_xor(a.as_f32x16(), b.as_f32x16()))
}

/// Compute the bitwise NOT of packed single-precision (32-bit) floating-point elements in a and then AND with b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_andnot_ps&expand=100,335)
#[inline]
#[target_feature(enable = "avx512dq")]
#[cfg_attr(test, assert_instr(vandnps))]
pub unsafe fn _mm512_andnot_ps(a: __m512, b: __m512) -> __m512 {
    _mm512_and_ps(_mm512_xor_ps(a, _mm512_set1_ps(f32::MAX)), b)
}
