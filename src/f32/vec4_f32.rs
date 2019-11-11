#![allow(dead_code)]

use crate::f32::Vec3;
use wasm_bindgen::prelude::*;

#[cfg(feature = "rand")]
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};

use std::{f32, fmt, ops::*};

#[wasm_bindgen]
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug, Default)]
// if compiling with simd enabled assume alignment needs to match the simd type
#[cfg_attr(not(feature = "scalar-math"), repr(align(16)))]
#[repr(C)]
pub struct Vec4(f32, f32, f32, f32);

#[wasm_bindgen]
impl Vec4 {
    /// Creates a new `Vec4` with all elements set to `0.0`.
    pub fn zero() -> Self {
        Self(0.0, 0.0, 0.0, 0.0)
    }

    /// Creates a new `Vec4` with all elements set to `1.0`.
    pub fn one() -> Self {
        Self(1.0, 1.0, 1.0, 1.0)
    }

    /// Creates a new `Vec4`.
    #[wasm_bindgen(constructor)]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(x, y, z, w)
    }

    /// Creates a new `Vec4` with values `[x: 1.0, y: 0.0, z: 0.0, w: 0.0]`.
    pub fn unit_x() -> Self {
        Self(1.0, 0.0, 0.0, 0.0)
    }

    /// Creates a new `Vec4` with values `[x: 0.0, y: 1.0, z: 0.0, w: 0.0]`.
    pub fn unit_y() -> Self {
        Self(0.0, 1.0, 0.0, 0.0)
    }

    /// Creates a new `Vec4` with values `[x: 0.0, y: 0.0, z: 1.0, w: 0.0]`.
    pub fn unit_z() -> Self {
        Self(0.0, 0.0, 1.0, 0.0)
    }

    /// Creates a new `Vec4` with values `[x: 0.0, y: 0.0, z: 0.0, w: 1.0]`.
    pub fn unit_w() -> Self {
        Self(0.0, 0.0, 0.0, 1.0)
    }

    /// Creates a new `Vec4` with all elements set to `v`.
    pub fn splat(v: f32) -> Self {
        Self(v, v, v, v)
    }

    /// Creates a `Vec3` from the first three elements of `self`,
    /// removing `w`.
    pub fn truncate(self) -> Vec3 {
        Vec3::new(self.0, self.1, self.2)
    }

    /// Returns element `x`.
    pub fn x(&self) -> f32 {
        self.0
    }

    /// Returns element `y`.
    pub fn y(&self) -> f32 {
        self.1
    }

    /// Returns element `z`.
    pub fn z(&self) -> f32 {
        self.2
    }

    /// Returns element `w`.
    pub fn w(&self) -> f32 {
        self.3
    }

    /// Sets element `x`.
    pub fn set_x(&mut self, x: f32) {
        self.0 = x;
    }

    /// Sets element `y`.
    pub fn set_y(&mut self, y: f32) {
        self.1 = y;
    }

    /// Sets element `z`.
    pub fn set_z(&mut self, z: f32) {
        self.2 = z;
    }

    /// Sets element `w`.
    pub fn set_w(&mut self, w: f32) {
        self.3 = w;
    }

    /// Returns a `Vec4` with all elements set to the value of element `x`.
    pub(crate) fn dup_x(self) -> Self {
        Self(self.0, self.0, self.0, self.0)
    }

    /// Returns a `Vec4` with all elements set to the value of element `y`.
    pub(crate) fn dup_y(self) -> Self {
        Self(self.1, self.1, self.1, self.1)
    }

    /// Returns a `Vec4` with all elements set to the value of element `z`.
    pub(crate) fn dup_z(self) -> Self {
        Self(self.2, self.2, self.2, self.2)
    }

    /// Returns a `Vec4` with all elements set to the value of element `w`.
    pub(crate) fn dup_w(self) -> Self {
        Self(self.3, self.3, self.3, self.3)
    }

    /// Computes the 4D dot product of `self` and `other`.
    pub fn dot(self, other: Self) -> f32 {
        (self.0 * other.0) + (self.1 * other.1) + (self.2 * other.2) + (self.3 * other.3)
    }

    /// Computes the 4D length of `self`.
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Computes the squared 4D length of `self`.
    ///
    /// This is generally faster than `Vec4::length()` as it avoids a square
    /// root operation.
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Computes `1.0 / Vec4::length()`.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    pub fn length_reciprocal(self) -> f32 {
        1.0 / self.length()
    }

    /// Returns `self` normalized to length 1.0.
    ///
    /// For valid results, `self` must _not_ be of length zero.
    pub fn normalize(self) -> Self {
        self * self.length_reciprocal()
    }

    /// Returns the vertical minimum of `self` and `other`.
    ///
    /// In other words, this computes
    /// `[x: min(x1, x2), y: min(y1, y2), z: min(z1, z2), w: min(w1, w2)]`,
    /// taking the minimum of each element individually.
    pub fn min(self, other: Self) -> Self {
        Self(
            self.0.min(other.0),
            self.1.min(other.1),
            self.2.min(other.2),
            self.3.min(other.3),
        )
    }

    /// Returns the vertical maximum of `self` and `other`.
    ///
    /// In other words, this computes
    /// `[x: max(x1, x2), y: max(y1, y2), z: max(z1, z2), w: max(w1, w2)]`,
    /// taking the maximum of each element individually.
    pub fn max(self, other: Self) -> Self {
        Self(
            self.0.max(other.0),
            self.1.max(other.1),
            self.2.max(other.2),
            self.3.max(other.3),
        )
    }

    /// Returns the minimum of all four elements in `self`.
    ///
    /// In other words, this computes `min(x, y, z, w)`.
    pub fn min_element(self) -> f32 {
        self.0.min(self.1.min(self.2.min(self.3)))
    }

    /// Returns the maximum of all four elements in `self`.
    ///
    /// In other words, this computes `max(x, y, z, w)`.
    pub fn max_element(self) -> f32 {
        self.0.max(self.1.max(self.2.min(self.3)))
    }

    /// Performs a vertical `==` comparison between `self` and `other`,
    /// returning a `Vec4Mask` of the results.
    ///
    /// In other words, this computes `[x1 == x2, y1 == y2, z1 == z2, w1 == w2]`.
    pub fn cmpeq(self, other: Self) -> Vec4Mask {
        Vec4Mask::new(
            self.0.eq(&other.0),
            self.1.eq(&other.1),
            self.2.eq(&other.2),
            self.3.eq(&other.3),
        )
    }

    /// Performs a vertical `!=` comparison between `self` and `other`,
    /// returning a `Vec4Mask` of the results.
    ///
    /// In other words, this computes `[x1 != x2, y1 != y2, z1 != z2, w1 != w2]`.
    pub fn cmpne(self, other: Self) -> Vec4Mask {
        Vec4Mask::new(
            self.0.ne(&other.0),
            self.1.ne(&other.1),
            self.2.ne(&other.2),
            self.3.ne(&other.3),
        )
    }

    /// Performs a vertical `>=` comparison between `self` and `other`,
    /// returning a `Vec4Mask` of the results.
    ///
    /// In other words, this computes `[x1 >= x2, y1 >= y2, z1 >= z2, w1 >= w2]`.
    pub fn cmpge(self, other: Self) -> Vec4Mask {
        Vec4Mask::new(
            self.0.ge(&other.0),
            self.1.ge(&other.1),
            self.2.ge(&other.2),
            self.3.ge(&other.3),
        )
    }

    /// Performs a vertical `>` comparison between `self` and `other`,
    /// returning a `Vec4Mask` of the results.
    ///
    /// In other words, this computes `[x1 > x2, y1 > y2, z1 > z2, w1 > w2]`.
    pub fn cmpgt(self, other: Self) -> Vec4Mask {
        Vec4Mask::new(
            self.0.gt(&other.0),
            self.1.gt(&other.1),
            self.2.gt(&other.2),
            self.3.gt(&other.3),
        )
    }

    /// Performs a vertical `<=` comparison between `self` and `other`,
    /// returning a `Vec4Mask` of the results.
    ///
    /// In other words, this computes `[x1 <= x2, y1 <= y2, z1 <= z2, w1 <= w2]`.
    pub fn cmple(self, other: Self) -> Vec4Mask {
        Vec4Mask::new(
            self.0.le(&other.0),
            self.1.le(&other.1),
            self.2.le(&other.2),
            self.3.le(&other.3),
        )
    }

    /// Performs a vertical `<` comparison between `self` and `other`,
    /// returning a `Vec4Mask` of the results.
    ///
    /// In other words, this computes `[x1 < x2, y1 < y2, z1 < z2, w1 < w2]`.
    pub fn cmplt(self, other: Self) -> Vec4Mask {
        Vec4Mask::new(
            self.0.lt(&other.0),
            self.1.lt(&other.1),
            self.2.lt(&other.2),
            self.3.lt(&other.3),
        )
    }

    /// Creates a new `Vec4` from the first four values in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than four elements long.
    pub fn from_slice_unaligned(slice: &[f32]) -> Self {
        Self(slice[0], slice[1], slice[2], slice[3])
    }

    /// Writes the elements of `self` to the first four elements in `slice`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is less than four elements long.
    pub fn write_to_slice_unaligned(self, slice: &mut [f32]) {
        slice[0] = self.0;
        slice[1] = self.1;
        slice[2] = self.2;
        slice[3] = self.3;
    }

    pub(crate) fn mul_add(self, a: Self, b: Self) -> Self {
        Self(
            (self.0 * a.0) + b.0,
            (self.1 * a.1) + b.1,
            (self.2 * a.2) + b.2,
            (self.3 * a.3) + b.3,
        )
    }

    pub(crate) fn neg_mul_sub(self, a: Self, b: Self) -> Self {
        Self(
            b.0 - (self.0 * a.0),
            b.1 - (self.1 * a.1),
            b.2 - (self.2 * a.2),
            b.3 - (self.3 * a.3),
        )
    }

    pub fn abs(self) -> Self {
        Self(self.0.abs(), self.1.abs(), self.2.abs(), self.3.abs())
    }

    // Necessary methods for JS (trait methods don't work)
    // Don't consume self in any of these, otherwise JS can't use the value
    // afterward
    pub fn add(&self, other: &Vec4) -> Self {
        *self + *other
    }
    pub fn sub(&self, other: &Vec4) -> Self {
        *self - *other
    }
    pub fn mul(&self, other: f32) -> Self {
        *self * other
    }
    pub fn to_string(&self) -> String {
        format!("Vec4({}, {}, {}, {})", self.0, self.1, self.2, self.3)
    }
}

impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.0, self.1, self.2, self.3)
    }
}

impl Div<Vec4> for Vec4 {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        Self(
            self.0 / other.0,
            self.1 / other.1,
            self.2 / other.2,
            self.3 / other.3,
        )
    }
}

impl DivAssign<Vec4> for Vec4 {
    fn div_assign(&mut self, other: Self) {
        *self = Self(
            self.0 / other.0,
            self.1 / other.1,
            self.2 / other.2,
            self.3 / other.3,
        )
    }
}

impl Div<f32> for Vec4 {
    type Output = Self;
    fn div(self, other: f32) -> Self {
        Self(
            self.0 / other,
            self.1 / other,
            self.2 / other,
            self.3 / other,
        )
    }
}

impl DivAssign<f32> for Vec4 {
    fn div_assign(&mut self, other: f32) {
        *self = Self(
            self.0 / other,
            self.1 / other,
            self.2 / other,
            self.3 / other,
        )
    }
}

impl Mul<Vec4> for Vec4 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self(
            self.0 * other.0,
            self.1 * other.1,
            self.2 * other.2,
            self.3 * other.3,
        )
    }
}

impl MulAssign<Vec4> for Vec4 {
    fn mul_assign(&mut self, other: Self) {
        *self = Self(
            self.0 * other.0,
            self.1 * other.1,
            self.2 * other.2,
            self.3 * other.3,
        )
    }
}

impl Mul<f32> for Vec4 {
    type Output = Self;
    fn mul(self, other: f32) -> Self {
        Self(
            self.0 * other,
            self.1 * other,
            self.2 * other,
            self.3 * other,
        )
    }
}

impl MulAssign<f32> for Vec4 {
    fn mul_assign(&mut self, other: f32) {
        *self = Self(
            self.0 * other,
            self.1 * other,
            self.2 * other,
            self.3 * other,
        )
    }
}

impl Mul<Vec4> for f32 {
    type Output = Vec4;
    fn mul(self, other: Vec4) -> Vec4 {
        Vec4(
            self * other.0,
            self * other.1,
            self * other.2,
            self * other.3,
        )
    }
}

impl Add for Vec4 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(
            self.0 + other.0,
            self.1 + other.1,
            self.2 + other.2,
            self.3 + other.3,
        )
    }
}

impl AddAssign for Vec4 {
    fn add_assign(&mut self, other: Self) {
        *self = Self(
            self.0 + other.0,
            self.1 + other.1,
            self.2 + other.2,
            self.3 + other.3,
        )
    }
}

impl Sub for Vec4 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self(
            self.0 - other.0,
            self.1 - other.1,
            self.2 - other.2,
            self.3 - other.3,
        )
    }
}

impl SubAssign for Vec4 {
    fn sub_assign(&mut self, other: Self) {
        *self = Self(
            self.0 - other.0,
            self.1 - other.1,
            self.2 - other.2,
            self.3 - other.3,
        )
    }
}

impl Neg for Vec4 {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0, -self.1, -self.2, -self.3)
    }
}

impl From<(f32, f32, f32, f32)> for Vec4 {
    fn from(t: (f32, f32, f32, f32)) -> Self {
        Self(t.0, t.1, t.2, t.3)
    }
}

impl From<Vec4> for (f32, f32, f32, f32) {
    fn from(v: Vec4) -> Self {
        (v.0, v.1, v.2, v.3)
    }
}

impl From<[f32; 4]> for Vec4 {
    fn from(a: [f32; 4]) -> Self {
        Self(a[0], a[1], a[2], a[3])
    }
}

impl From<Vec4> for [f32; 4] {
    fn from(v: Vec4) -> Self {
        [v.0, v.1, v.2, v.3]
    }
}

#[cfg(feature = "rand")]
impl Distribution<Vec4> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec4 {
        rng.gen::<(f32, f32, f32, f32)>().into()
    }
}

/// A 4-dimensional vector mask.
///
/// This type is typically created by comparison methods on `Vec4`.  It is
/// essentially a vector of four boolean values.
#[wasm_bindgen]
#[derive(Clone, Copy, Default)]
// if compiling with simd enabled assume alignment needs to match the simd type
#[cfg_attr(not(feature = "scalar-math"), repr(align(16)))]
#[repr(C)]
pub struct Vec4Mask(u32, u32, u32, u32);

#[wasm_bindgen]
impl Vec4Mask {
    /// Creates a new `Vec4Mask`.
    pub fn new(x: bool, y: bool, z: bool, w: bool) -> Self {
        const MASK: [u32; 2] = [0, 0xff_ff_ff_ff];
        Self(
            MASK[x as usize],
            MASK[y as usize],
            MASK[z as usize],
            MASK[w as usize],
        )
    }

    /// Returns a bitmask with the lowest four bits set from the elements of
    /// the `Vec4Mask`.
    ///
    /// A true element results in a `1` bit and a false element in a `0` bit.
    /// Element `x` goes into the first lowest bit, element `y` into the
    /// second, etc.
    pub fn bitmask(&self) -> u32 {
        (self.0 & 0x1) | (self.1 & 0x1) << 1 | (self.2 & 0x1) << 2 | (self.3 & 0x1) << 3
    }

    /// Returns true if any of the elements are true, false otherwise.
    ///
    /// In other words: `x || y || z || w`.
    pub fn any(&self) -> bool {
        (self.0 != 0) || (self.1 != 0) || (self.2 != 0) || (self.3 != 0)
    }

    /// Returns true if all the elements are true, false otherwise.
    ///
    /// In other words: `x && y && z && w`.
    pub fn all(&self) -> bool {
        (self.0 != 0) && (self.1 != 0) && (self.2 != 0) && (self.3 != 0)
    }

    /// Creates a new `Vec4` from the elements in `if_true` and `if_false`,
    /// selecting which to use for each element based on the `Vec4Mask`.
    ///
    /// A true element in the mask uses the corresponding element from
    /// `if_true`, and false uses the element from `if_false`.
    pub fn select(self, if_true: Vec4, if_false: Vec4) -> Vec4 {
        Vec4(
            if self.0 != 0 { if_true.0 } else { if_false.0 },
            if self.1 != 0 { if_true.1 } else { if_false.1 },
            if self.2 != 0 { if_true.2 } else { if_false.2 },
            if self.3 != 0 { if_true.3 } else { if_false.3 },
        )
    }
}

impl BitAnd for Vec4Mask {
    type Output = Self;

    fn bitand(self, other: Self) -> Self {
        Self(
            self.0 & other.0,
            self.1 & other.1,
            self.2 & other.2,
            self.3 & other.3,
        )
    }
}

impl BitAndAssign for Vec4Mask {
    fn bitand_assign(&mut self, other: Self) {
        *self = *self & other
    }
}

impl BitOr for Vec4Mask {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(
            self.0 | other.0,
            self.1 | other.1,
            self.2 | other.2,
            self.3 | other.3,
        )
    }
}

impl BitOrAssign for Vec4Mask {
    fn bitor_assign(&mut self, other: Self) {
        *self = *self | other
    }
}

impl Not for Vec4Mask {
    type Output = Self;

    fn not(self) -> Self {
        Self(!self.0, !self.1, !self.2, !self.3)
    }
}
