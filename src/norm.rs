pub use crate::common::R;


// Activation function for a neuron.  Used to normalize each neuron's output.
fn sigmoid(x: R) -> R {
  1.0 / (1.0 + (-x).exp())
}

// Derivative of activation function.
// Note that this is in terms of the *output* of the function,
// rather than the input
fn sigmoid_dy(y: R) -> R {
  y * (1.0 - y)
}

fn sel(actual: &[R], expected: &[R]) -> R {
    expected.iter().zip(actual.iter())
    .map(|(a,b)| (a - b) * (a - b)).sum::<R>() * 0.5
  }

/// How to normalize values computed by the network.
pub trait Norm {
  /// How to normalize internal results of the net.
  type INorm: Internal;

  /// How to normalize the output of the net.
  type ONorm: Output;
  fn error(actual: &[R], expected: &[R]) -> R {
    Self::ONorm::error(actual,expected)
  }
}

impl<I: Internal, O: Output> Norm for (I,O) {
  type INorm = I;
  type ONorm = O;
}


/// How to normalize the internal results computed by the net.
pub trait Internal {
  /// Normalize an internally computed value.
  fn neuron_norm(x: R) -> R;

  /// The gradient of the normalization function, in terms of
  /// its **OUTPUT** at a point (i.e, we want to compute the gradient at `x`,
  /// but we only provide this function with `neuron_norm(x)` instead
  /// of giving it `x`). 
  fn neuron_norm_dy(y: R) -> R;
}

/// Normalize internal values using `sigmoid`.
#[derive(Copy,Clone)]
pub struct ISigmoid {}

impl Internal for ISigmoid {
  fn neuron_norm(x: R) -> R { sigmoid(x) }
  fn neuron_norm_dy(y: R) -> R { sigmoid_dy(y) }
}


/// How to compute and validate the output of the net.
pub trait Output {
  /// Normalize the given results.
  fn normalize(ys: &mut [R]);

  /// Compute the error for the given results.
  fn error(actual: &[R], expected: &[R]) -> R;

  /// Compute the error gradient.   The error gradient is returned in
  /// the location where the expected values were provided.
  fn error_delta(actual: &[R], expected: &mut[R]);
}

/// Don't normalize the outputs.
/// Error is computed as half of the sum of the squares of the individual errors.
#[derive(Copy,Clone)]
pub struct ORVec {}

impl Output for ORVec {
  fn normalize(_xs: &mut [R]) {}

  fn error(actual: &[R], expected: &[R]) -> R { sel(actual,expected) }

  fn error_delta(actual: &[R], expected: &mut [R]) {
    for i in 0 .. actual.len() {
      expected[i] = actual[i] - expected[i];
    }
  }
}

/// Normalize the outputs with `sigmoid`.
/// Error is computed as half of the sum of the squares of the individual errors.
#[derive(Copy,Clone)]
pub struct OBitVec {}

impl Output for OBitVec {
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


/// Normalize outputs with `softmax`, and use `cross-entropy` to estimate the error.
/// This is suitable for functions that need to pick one out of some
/// options (i.e., classifiers).   Each result represents the likelihood that
/// the input belongs to the given class, and all outputs sum up to 1.
#[derive(Copy,Clone)]
pub struct OClassifier {}

impl Output for OClassifier {
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

