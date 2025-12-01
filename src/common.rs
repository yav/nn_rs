/// The type of net weights, inputs, and outputs.
pub type R = f32;
pub type Neuron = Vec<R>;       // defined by weights

// Assumes `xs` contains a bias input (1)
pub fn linear(ws: &[R], xs: &[R]) -> R {
  ws.iter().zip(xs).map(|(x,y)| x * y).sum::<R>()
}

// How error will change if we change the linear part of a neuron neuron.
//
// This assumes that inputs and outputs contain bias elements.
//
// `other` is the parameter that is fixed (e.g., inputs if changing weights),
// `y` is the output of the neuron at the point of interest,
// `d_err` indicate how the error will change wrt to this neuron (i.e., what it is connected to)
// `delta` is where we place the derivatives.
pub fn linear_delta(other: &[R], d_err: &[R], delta: &mut [R]) {
  for i in 0 .. other.len() {
    let x = other[i];
    delta[i] += d_err.iter().map(|d| x * *d).sum::<R>();  
  };
}

// Assumes `xs` and `ys` contains a bias element.
pub fn lin_layer_dw(xs: &[R], d_err: &[R], d_ns: &mut [Vec<R>]) {
  for i in 0 .. d_ns.len() {
    linear_delta(xs, d_err, d_ns[i].as_mut_slice());
  };
}

// Assumes `xs` contains a bias input
// In the last layer a neuron is connected only to a single input of the error function.
pub fn last_lin_layer_dw(xs: &[R], gs_dy: &[R], d_ns: &mut [Vec<R>]) {
  for i in 0 .. d_ns.len() {
    linear_delta(xs, &[gs_dy[i]], d_ns[i].as_mut_slice());
  };
}

// Compute derivative wrt to inputs
pub fn lin_layer_dx(ns: &[Neuron], d_err: &[R], d_xs: &mut [R]) {
  d_xs.fill(0.0);
  for i in 0 .. ns.len() {
    linear_delta(&ns[i].as_slice()[1..], d_err, d_xs);
  };
}

// Compute derivative wrt to inputs (not counting bias)
pub fn last_lin_layer_dx(ns: &[Neuron], d_err: &[R], d_xs: &mut [R]) {
  d_xs.fill(0.0);
  for i in 0 .. ns.len() {
    linear_delta(&ns[i].as_slice()[1..], &[d_err[i]], d_xs);
  };
}


