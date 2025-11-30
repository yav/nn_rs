/// The type of net weights, inputs, and outputs.
pub type R = f32;

// Activation function for a neuron.  Used to normalize each neuron's output.
pub fn sigmoid(x: R) -> R {
  1.0 / (1.0 + (-x).exp())
}

// Derivative of activation function.
// Note that this is in terms of the *output* of the function,
// rather than the input
pub fn sigmoid_dy(y: R) -> R {
  y * (1.0 - y)
}

pub fn sel(actual: &[R], expected: &[R]) -> R {
    expected.iter().zip(actual.iter())
    .map(|(a,b)| (a - b) * (a - b)).sum::<R>() * 0.5
  }
