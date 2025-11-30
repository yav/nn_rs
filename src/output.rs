use crate::common::{R,sigmoid,sigmoid_dy,sel};

/// How to compute and validate the output of the net.
pub trait Norm {
  /// Normalize the given results.
  fn normalize(ys: &mut [R]);

  /// Compute the error for the given results.
  fn error(actual: &[R], expected: &[R]) -> R;

  /// Compute the error gradient.   The error gradient is returned in
  /// the location where the expected values were provided.
  fn error_delta(actual: &[R], expected: &mut[R]);
}

/// Don't normalize the outputs, and use square error loss.
pub struct RVec {}

impl Norm for RVec {
  fn normalize(_xs: &mut [R]) {}

  fn error(actual: &[R], expected: &[R]) -> R { sel(actual,expected) }

  fn error_delta(actual: &[R], expected: &mut [R]) {
    for i in 0 .. actual.len() {
      expected[i] = actual[i] - expected[i];
    }
  }
}

/// Normalize the outputs with sigmoid, and use square error loss.
pub struct BitVec {}

impl Norm for BitVec {
  fn normalize(ys: &mut [R]) {
    for y in ys.iter_mut() {
      *y = sigmoid(*y);
    }
  }
  fn error(actual: &[R], expected: &[R]) -> R { sel(actual, expected) }

  fn error_delta(actual: &[R], expected: &mut[R]) {
    for i in 0 .. actual.len() {
      expected[i] = actual[i] - expected[i];
    } 
  }
}


/// Normalize outputs with softmax, and use cross-entropy loss.
/// This is suitable for functions that need to pick one out of some
/// options (i.e., classifiers).   Each result represents the likelihood that
/// the input belongs to the given class, and all outputs sum up to 1.
pub struct Classifier {}

impl Norm for Classifier {
  fn normalize(xs: &mut [R]) {
    let tot = xs.iter().map(|x| x.exp()).sum::<R>();
    for i in 0 .. xs.len() {
      xs[i] = xs[i].exp() / tot; 
    }
  }
  fn error(actual: &[R], expected: &[R]) -> R {
    - actual.iter().zip(expected.iter()).map(|(x,t)| t * x.ln()).sum::<R>()
  }
  fn error_delta(actual: &[R], expected: &mut [R]) {
    for i in 0 .. actual.len() {
      let a = actual[i];
      expected[i] = sigmoid_dy(a) * (a - expected[i]);
    }
  }
}

