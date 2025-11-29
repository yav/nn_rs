use std::io::{Read, Write};
use rand::Rng;
use rand::distr::Uniform;

/// The type of net weights, inputs, and outputs.
pub type R = f32;

type Neuron = Vec<R>;       // defined by weights
type Layer  = Vec<Neuron>;  // a bunch of neurons sharing input


/// How to compute the output of the net, and check for error.
pub trait FinalLayer {
  /// Normalize the given results.
  /// First element of `ys` is bias.
  fn normalize(ys: &mut [R]);

  /// Compute the error for the given results.
  /// `actual` and `expected` DO  NOT contain bias.
  fn error(actual: &[R], expected: &[R]) -> R;

  /// Compute the error gradient.   The error gradient is returned in
  /// the location where the expected values were provided.
  /// First element of `actual` iss bias, `expected` does not have bias.
  fn error_delta(actual: &[R], expected: &mut[R]);
}

/// Don't normalize the outputs, and use square error loss.
pub struct Direct {}

impl FinalLayer for Direct {
  fn normalize(_xs: &mut [R]) {}

  fn error(actual: &[R], expected: &[R]) -> R {
    expected.iter().zip(actual.iter())
    .map(|(a,b)| (a - b) * (a - b)).sum::<R>() * 0.5
  }
  fn error_delta(actual: &[R], expected: &mut [R]) {
    for i in 1 .. actual.len() {
      expected[i - 1] = actual[i] - expected[i - 1];
    }
  }
}


/// Normalize outputs with softmax, and use cross-entropy loss.
/// This is suitable for functions that need to pick one out of some
/// options (i.e., classifiers).   Each result represents the likelihood that
/// the input belongs to the given class, and all outputs some up to 1.
pub struct Softmax {}

impl FinalLayer for Softmax {
  fn normalize(xs: &mut [R]) {
    let tot = xs.iter().skip(1).map(|x| x.exp()).sum::<R>();
    for i in 1 .. xs.len() {
      xs[i] = xs[i].exp() / tot; 
    }
  }
  fn error(actual: &[R], expected: &[R]) -> R {
    - actual.iter().zip(expected.iter()).map(|(x,t)| t * x.ln()).sum::<R>()
  }
  fn error_delta(actual: &[R], expected: &mut [R]) {
    for i in 1 .. actual.len() {
      expected[i - 1] = actual[i] - expected[i - 1];
    }
  }
}

/// Activation function for a neuron.  Used to normalize each neuron's
/// output.
fn sigmoid(x: R) -> R {
  1.0 / (1.0 + (-x).exp())
}

// Derivative of activation function.
// Note that this is in terms of the *output* of the function,
// rather than the input
fn sigmoid_dy(y: R) -> R {
  y * (1.0 - y)
}

/// Loss function.  We use half of the sum of squares of the pointwise errors.
pub fn loss(ys_actual: &[R], ys_expected: &[R]) -> R {
  ys_expected.iter().zip(ys_actual.iter())
    .map(|(a,b)| (a - b) * (a - b)).sum::<R>() * 0.5
}

// How the loos function changes when each of its inputs are changed.
// The first argument contains bias.
// The second argument takes the expected results,
// and returns the gradient.
fn loss_dx(ys_actual: &[R], ys_expected: &mut [R]) {
  for i in 1 .. ys_actual.len() {
    ys_expected[i - 1] = ys_actual[i] - ys_expected[i - 1];
  }
}

// Assumes `xs` contains a bias input (1)
fn linear(ws: &[R], xs: &[R]) -> R {
  ws.iter().zip(xs).map(|(x,y)| x * y).sum::<R>()
}

// Assumes `xs` contains a bias input (1)
fn neuron(ws: &[R], xs: &[R]) -> R {
  sigmoid(linear(ws,xs))
}

// How error will change if we change the linear part of a neuron neuron.
//
// This assumes that inputs and outputs contain bias elements.
//
// `other` is the parameter that is fixed (e.g., inputs if changing weights),
// `y` is the output of the neuron at the point of interest,
// `d_err` indicate how the error will change wrt to this neuron (i.e., what it is connected to)
// `delta` is where we place the derivatives.
fn neuron_lin_delta(other: &[R], d_err: &[R], delta: &mut [R]) {
  for i in 0 .. other.len() {
    let x = other[i];
    delta[i] += d_err.iter().map(|d| x * *d).sum::<R>();  
  };
}


// Assumes `xs` contains a bias input
// Produces an additional bias result in the first slot of the output
fn layer(ns: &[Neuron], xs: &[R], res: &mut [R]) {
  res[0] = 1.0;
  for (ws,tgt) in ns.iter().zip(res[1..].iter_mut()) {
    *tgt = neuron(ws.as_slice(), xs)
  }
}


// Update the error gradient to account for changes due to the actuators.
// `ys` is the results of the layer (i.e., the normalized value).
// The first element is bias
fn actuator_layer_delta(ys: &[R], gs_dy: &mut [R]) {
  for i in 1 .. ys.len() {
    gs_dy[i - 1] *= sigmoid_dy(ys[i]);
  }
}

// Assumes `xs` and `ys` contains a bias element.
fn lin_layer_dw(xs: &[R], d_err: &[R], d_ns: &mut [Vec<R>]) {
  for i in 0 .. d_ns.len() {
    neuron_lin_delta(xs, d_err, d_ns[i].as_mut_slice());
  };
}

// Assumes `xs` contains a bias input
// In the last layer a neuron is connected only to a single input of the error function.
fn last_lin_layer_dw(xs: &[R], gs_dy: &[R], d_ns: &mut [Vec<R>]) {
  for i in 0 .. d_ns.len() {
    neuron_lin_delta(xs, &[gs_dy[i]], d_ns[i].as_mut_slice());
  };
}

// Compute derivative wrt to inputs
fn lin_layer_dx(ns: &[Neuron], d_err: &[R], d_xs: &mut [R]) {
  d_xs.fill(0.0);
  for i in 0 .. ns.len() {
    neuron_lin_delta(&ns[i].as_slice()[1..], d_err, d_xs);
  };
  
}

// Compute derivative wrt to inputs (not counting bias)
fn last_lin_layer_dx(ns: &[Neuron], d_err: &[R], d_xs: &mut [R]) {
  
  d_xs.fill(0.0);
  for i in 0 .. ns.len() {
    neuron_lin_delta(&ns[i].as_slice()[1..], &[d_err[i]], d_xs);
  };
}


/// Geometry of a neural net.
#[derive(Copy,Clone,Debug)]
pub struct Dim {
  /// Number of inputs to the net
  pub inputs: usize,

  /// Number of results produced by the net
  pub outputs: usize,

  /// How many hidden layers we have
  pub hidden: usize,

  /// How big is each hidden layer
  pub hidden_size: usize
}

impl Dim {
  fn save(self, f: &mut impl Write) -> std::io::Result<()>{
    let mut save = |x| f.write_all(&(x as u64).to_le_bytes());
    save(self.inputs)?;
    save(self.outputs)?;
    save(self.hidden)?;
    save(self.hidden_size)
  }

  fn load(f: &mut impl Read) -> std::io::Result<Dim> {
    let mut load = || {
      let mut b = [0; size_of::<u64>()];
      if let Err(e) = f.read_exact(&mut b) { Err(e) }
      else { Ok(u64::from_le_bytes(b) as usize) }
    };
    Ok(Dim {
      inputs:       load()?,
      outputs:      load()?,
      hidden:       load()?,
      hidden_size:  load()?
    })
  }
}

/// The weights of a neural network.
pub struct Weights {
  layers: Vec<Layer>
}

impl Weights {
  /// Create a new net where all weights are a constant.
  pub fn new(dim: Dim, ini: R) -> Self {
    let mut layers  = Vec::with_capacity(2 + dim.hidden);
    let neuron      = |i| vec![ini; i + 1]; // bias + weights for inputs
    let mut layer   = |i,o| layers.push(vec![neuron(i); o]);
    
    layer(dim.inputs, dim.hidden_size);
    for _ in 0 .. dim.hidden {
      layer(dim.hidden_size,dim.hidden_size)
    }
    layer(dim.hidden_size, dim.outputs);
    Weights { layers }
  }

  fn iter(&self) -> impl DoubleEndedIterator<Item=&Layer> {
    self.layers.iter()
  }

  fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item=&mut Layer> {
    self.layers.iter_mut()
  }

  /// Get the dimension of this net
  pub fn dim(&self) -> Dim {
    Dim {
      inputs: self.input_size(),
      outputs: self.output_size(),
      hidden_size: self.hidden_size(),
      hidden: self.layer_num() - 2
    }
  }

  /// Number of inputs.
  fn input_size(&self) -> usize {
    self.layers[0][0].len() - 1
  }

  /// Number of outputs.
  fn output_size(&self) -> usize {
    self.layers.last().unwrap().len()
  }

  /// Size of hidden layers.
  fn hidden_size(&self) -> usize {
    self.layers[0].len()
  }

  /// Total number of layers in the network, including
  /// input layer, hidden layers, and output layer.
  fn layer_num(&self) -> usize {
    self.layers.len()
  }

  /// Save the weights to a file.
  pub fn save(&self, path: &str) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    self.dim().save(&mut f)?;
    for l in self.iter() {
      for n in l.iter() {
        for w in n.iter() {
          f.write_all(&w.to_le_bytes())?
        }
      }
    }
    Ok(())
  }

  /// Load some weights from a file.
  pub fn load(path: &str) -> std::io::Result<Weights> {
    let mut f = std::fs::File::open(path)?;
    let dim = Dim::load(&mut f)?;
    let mut net = Weights::new(dim, 0.0);
    for l in net.iter_mut() {
      for n in l.iter_mut() {
        for w in n.iter_mut() {
          let mut b = [0; size_of::<R>() ];
          f.read_exact(&mut b)?;
          *w = R::from_le_bytes(b);
        }
      }
    }
    Ok(net)
  }

  /// Set the weights to random numbers in the given range (inclusive)
  pub fn randomize(&mut self, lower: R, upper: R) {
    let mut rng = rand::rng();
    let d = Uniform::new(lower, upper).unwrap();
    for l in self.iter_mut() {
      for n in l.iter_mut() {
        for w in n.iter_mut() {
          *w = rng.sample(d);
        }
      }
    }
  }

  /// Print the network to stdout (for debugging).
  pub fn print(&self) {
    let last = self.layer_num() - 1;
    for (i,l) in self.iter().enumerate() {
      if i == 0 { println!("=== Input Layer ({} -> {}) ===", self.input_size(), self.hidden_size()) } else {
      if i == last { println!("=== Output Layer ({} -> {}) ===", self.hidden_size(), self.output_size()) } else {  
      println!("=== Hidden Layer {} ({} -> {}) ===", i, self.hidden_size(), self.hidden_size())
      }};
      
      for (i,n) in l.iter().enumerate() {
        print!("({:4})", i);
        for w in n.iter() {
          print!(" {:8.4}", *w);
        }
        println!("")
      }
      println!("")
    }
  }
}


/// Infrastructure for evaluating a net, without the weights.
pub struct RunnerEmpty { buf1: Vec<R>, buf2: Vec<R> }

/// A neural net that can be used to map inputs to outputs.
pub struct RunnerReady<'a> { net: &'a Weights, buf1: Vec<R>, buf2: Vec<R> }


impl RunnerEmpty {
  pub fn new() -> Self {
    RunnerEmpty { buf1: vec![], buf2: vec![] }
  }

  pub fn set_weights(mut self, net: &Weights) -> RunnerReady {
    let size = 1 + std::cmp::max(net.input_size(), std::cmp::max(net.hidden_size(), net.output_size()));
    self.buf1.resize(size, 0.0);
    self.buf2.resize(size, 0.0);
    RunnerReady { net:net, buf1: self.buf1, buf2: self.buf2 }
  } 
}

impl<'a> RunnerReady<'a> {

  pub fn clear_net(self) -> RunnerEmpty {
    RunnerEmpty { buf1: self.buf1, buf2: self.buf2 }
  }

  /// Get a reference to fill in the input to the net.
  pub fn set_input(&mut self) -> &mut[R] {
    let b = self.buf1.as_mut_slice();
    b[0] = 1.0;
    &mut b[1 ..= self.net.input_size()]
  }

  /// Evaluate the net on the current input.
  pub fn eval(&mut self) {
    for (i,l) in self.net.iter().enumerate() {
      let (rd,wt) = if i & 1 == 0 { (self.buf1.as_slice(), self.buf2.as_mut_slice()) }
                             else { (self.buf2.as_slice(), self.buf1.as_mut_slice()) };
      layer(l.as_slice(), rd, wt);
    }
  }

  /// Get the output of the net.
  pub fn get_output(&self) -> &[R] {
    let r = if self.net.layer_num() & 1 == 0 {
              self.buf1.as_slice()
            } else {
              self.buf2.as_slice()
            };
    &r[1 ..= self.net.output_size()]
  }

}

/// A neural net in training.
pub struct Learner {
  net:      Weights,            // weights
  d_layers: Weights,            // weight gradients
  batches: R,               // how many samples are in the (gradient for batching)

  /// Determines how much to change the net's state based on a batch
  /// of examples.
  pub learning_rate: R,

  // The first one is for input, the rest are for the outputs of the layers.
  // Note that all vectors are 1 longer to accommodate for bias
  buffers: Vec<Vec<R>>,
  
  // Intermediate layer gradient buffers (swap)
  gbuf1: Vec<R>,
  gbuf2: Vec<R>
}


impl Learner {

  /// Create a trainer for the given net.
  pub fn new(net: Weights) -> Self {
    let dim = net.dim();
    let size = std::cmp::max(dim.hidden, dim.outputs);
    let mut bufs = Vec::with_capacity(1 + net.layer_num()); // 1 extra fro input
    bufs.push(vec![0.0; dim.inputs + 1]);
    for _ in 0 ..= dim.hidden { bufs.push(vec![0.0; dim.hidden_size + 1]) }
    bufs.push(vec![0.0; dim.outputs + 1]);
    Learner {
      net:        net,
      d_layers:   Weights::new(dim, 0.0),
      batches:    0.0,

      learning_rate: 0.1,

      buffers:    bufs,
      gbuf1:      vec![0.0; size],
      gbuf2:      vec![0.0; size],
    }
  }

  /// Get a buffer to fill with the net's input.
  pub fn set_input(&mut self) -> &mut [R] {
    let b = self.buffers[0].as_mut_slice();
    b[0] = 1.0;
    let n = b.len();
    &mut b[1 .. n]
  }

  /// Get a buffer to fill a desired output.
  pub fn set_output(&mut self) -> &mut [R] {
    // Note that gradient buffers do not contain space for the bias inputs
    // as those don't change.
    let b = self.gbuf1.as_mut_slice();
    &mut b[0 .. self.net.output_size()]
  }

  /// Train based on the current input output pair.  For batch training,
  /// one can do multiple examples, and update the net as the average
  /// of the change from all examples.
  pub fn train(&mut self) {
    self.eval();
    self.backprop();
    self.batches += 1.0;
  }

  /// Update the net's state based on the examples in the current batch.
  pub fn finish_batch(&mut self) {
    let scale = self.learning_rate / self.batches;
    for (l,dl) in self.net.iter_mut().zip(self.d_layers.iter_mut()) {
      for (n,dn) in l.iter_mut().zip(dl.iter_mut()) {
        for (w,dw) in n.iter_mut().zip(dn.iter_mut()) {
          *w -= *dw * scale;
          *dw = 0.0;
        }
      }
    }
    self.batches = 0.0;
  }

  /// Evaluate the net.  After, the first buffer contains the net inputs,
  /// and the rest contain the outputs of the layers.
  fn eval(&mut self) {
    for(i,l) in self.net.layers.iter().enumerate() {
      let (xs,ys) = self.buffers.as_mut_slice().split_at_mut(i+1);
      let rd = xs.last().unwrap().as_slice();
      let wt = ys[0].as_mut_slice();
      layer(l, rd, wt);
    }
  }

  
  fn backprop(&mut self) {
    let ins       = self.buffers.iter().map(|x| x.as_slice()).rev();
    let outs      = self.buffers.iter().skip(1).map(|x| x.as_slice()).rev();
    let ls        = self.net.iter().rev();
    let dls       = self.d_layers.iter_mut().rev();
    let mut steps = ins.zip(outs).zip(ls).zip(dls);

    let (((last_is, last_os), last_ns), last_dns) = steps.next().unwrap();

    loss_dx(last_os, self.gbuf1.as_mut_slice());
    actuator_layer_delta(last_os, self.gbuf1.as_mut_slice());
    last_lin_layer_dw(last_is, self.gbuf1.as_slice(), last_dns.as_mut_slice());
    last_lin_layer_dx(last_ns.as_slice(), self.gbuf1.as_slice(), self.gbuf2.as_mut_slice());

    let mut swap = false;
    for (((xs,ys), ns), dns) in steps {
      let (cur,next) = if swap { (self.gbuf1.as_mut_slice(), self.gbuf2.as_mut_slice()) }
                          else { (self.gbuf2.as_mut_slice(), self.gbuf1.as_mut_slice()) };
      actuator_layer_delta(ys, cur);
      lin_layer_dw(xs,            cur, dns.as_mut_slice());
      lin_layer_dx(ns.as_slice(), cur, next);
      swap = !swap;
    }
  }


// r = phi(w1 + x*w2) * w3 + w4

  /// The current state of the net
  pub fn get_weights(&self) -> &Weights { &self.net }

  /// Extract net and destroy training infrastructure.
  pub fn complete(self) -> Weights { self.net }
}

