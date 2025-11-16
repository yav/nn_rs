use std::io::{Read, Write};

/// The type of net weights, inputs, and outputs.
pub type R = f32;

type Neuron = Vec<R>;       // defined by weights
type Layer  = Vec<Neuron>;  // a bunch of neurons sharing input


/// Activation function for a neuron.  Used to normalize each neuron's
/// output.
pub fn sigmoid(x: R) -> R {
  1.0 / (1.0 + (-x).exp())
}

// Derivative of activation function.
// Note that this is in terms of the *output* of the function,
// rather than the input
fn sigmoid_dy(y: R) -> R {
  y * (1.0 - y)
}

/// Square error loss function.  Used during training to estimate how
/// far off we are from a desired result.
pub fn sel(ys_actual: &[R], ys_expected: &[R]) -> R {
  ys_expected.iter().zip(ys_actual.iter())
    .map(|(a,b)| (a - b) * (a - b)).sum::<R>() * 0.5
}

// How SEL changes when each of its inputs are changed
fn sel_dx(ys_actual: &[R], ys_expected_in_out: &mut [R]) {
  for (e,x) in ys_actual.iter().zip(ys_expected_in_out.iter_mut()) {
    *x -= e;
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

// How error will change if we change a neuron, which is not in the last layer.
// The last layer is a bit different because the output of the neuron is
// connected to only one input of the loss function---in contrast, a neuron
// from an earlier layer sends its output to all inputs of the net layer.
// We use this for both computing the change wrt weights and inputs,
// but we provide different arguments in each case.
//
// This assumes that inputs and outputs contain bias elements.
// `other` is the parameter that is fixed (e.g., inputs if changing weights),
// `y` is the output of the neuron at the point of interest,
// `gs_dy` are the partial derivatives of the rest of the error function
//    (following layers and loss function) wrt to their inputs
//    this only counts derivates wrt actul inputs (i.e., no bias)
// `delta` is where we place the derivatives.
fn neuron_delta(other: &[R], y: R, gs_dy: &[R], delta: &mut [R]) {
  let dsig = sigmoid_dy(y);
  for (i,out) in other.iter().zip(delta.iter_mut()) {
    let dlocal = *i * dsig;  // the derivative of linear is just other
    *out += gs_dy.iter().map(|d| dlocal * *d).sum::<R>(); 
  }
}

// Each neuron in the last layer is connect only to one input of the loss function
fn neuron_delta_last(other: &[R], y: R, g_dy: R, delta: &mut [R]) {
  let grad = sigmoid_dy(y) * g_dy;
  for (i,out) in other.iter().zip(delta.iter_mut()) {
    *out += *i * grad
  }
}

// Assumes `xs` contains a bias input
// Produces an additional bias result in the first slot of the output
fn layer(ns: &[Neuron], xs: &[R], res: &mut [R]) {
  res[0] = 1.0;
  for (ws,tgt) in ns.iter().zip(res[1..].iter_mut()) {
    *tgt = neuron(ws.as_slice(), xs)
  }
}


// Assumes `xs` contains a bias input
fn layer_dw(xs: &[R], ys: &[R], gs_dy: &[R], d_ns: &mut [Vec<R>]) {
  for (y,dws) in ys.iter().zip(d_ns.iter_mut()) {
    neuron_delta(xs, *y, gs_dy, dws);
  }
}

// Assumes `xs` contains a bias input
fn last_layer_dw(xs: &[R], ys: &[R], gs_dy: &[R], d_ns: &mut [Vec<R>]) {
  for ((y,dws),g) in ys.iter().zip(d_ns.iter_mut()).zip(gs_dy) {
    neuron_delta_last(xs, *y, *g, dws);
  }
}

// Compute derivative wrt to inputs (not counting bias)
fn layer_dx(ns: &[Neuron], ys: &[R], gs_dy: &[R], d_xs: &mut [R]) {
  d_xs.fill(0.0);
  for (y,ws) in ys.iter().zip(ns.iter()) {
    neuron_delta(&ws.as_slice()[1..], *y, gs_dy, d_xs);
  }
}

// Compute derivative wrt to inputs (not counting bias)
fn layer_dx_last(ns: &[Neuron], ys: &[R], gs_dy: &[R], d_xs: &mut [R]) {
  d_xs.fill(0.0);
  for ((y,ws),g) in ys.iter().zip(ns.iter()).zip(gs_dy) {
    neuron_delta_last(&ws.as_slice()[1..], *y, *g, d_xs);
  }
}


/// Geometry of a neural net.
#[derive(Copy,Clone)]
pub struct NetDim {
  /// Number of inputs to the net
  pub inputs: usize,

  /// Number of results produced by the net
  pub outputs: usize,

  /// How many hidden layers we have
  pub hidden: usize,

  /// How big is each hidden layer
  pub hidden_size: usize
}

impl NetDim {
  fn save(self, f: &mut impl Write) -> std::io::Result<()>{
    let mut save = |x| f.write_all(&(x as u64).to_le_bytes());
    save(self.inputs)?;
    save(self.outputs)?;
    save(self.hidden)?;
    save(self.hidden_size)
  }

  fn load(f: &mut impl Read) -> std::io::Result<NetDim> {
    let mut load = || {
      let mut b = [0; size_of::<u64>()];
      if let Err(e) = f.read_exact(&mut b) { Err(e) }
      else { Ok(u64::from_le_bytes(b) as usize) }
    };
    Ok(NetDim {
      inputs:       load()?,
      outputs:      load()?,
      hidden:       load()?,
      hidden_size:  load()?
    })
  }
}

/// The weights of a neural network.
pub struct Net {
  layers: Vec<Layer>
}

impl Net {
  /// Create a new net where all weights are a constant.
  pub fn new(dim: &NetDim, ini: R) -> Self {
    let mut layers  = Vec::with_capacity(2 + dim.hidden);
    let neuron      = |i| vec![ini; i + 1]; // bias + weights for inputs
    let mut layer   = |i,o| layers.push(vec![neuron(i); o]);
    
    layer(dim.inputs, dim.hidden_size);
    for _ in 0 .. dim.hidden {
      layer(dim.hidden_size,dim.hidden_size)
    }
    layer(dim.hidden_size, dim.outputs);
    Net { layers }
  }

  fn iter(&self) -> impl DoubleEndedIterator<Item=&Layer> {
    self.layers.iter()
  }

  fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item=&mut Layer> {
    self.layers.iter_mut()
  }

  /// Get the dimension of this net
  pub fn dim(&self) -> NetDim {
    NetDim {
      inputs: self.input_size(),
      outputs: self.output_size(),
      hidden_size: self.hidden_size(),
      hidden: self.layer_num() - 2
    }
  }

  /// Number of inputs.
  fn input_size(&self) -> usize {
    self.layers[0].len() - 1
  }

  /// Number of outputs.
  fn output_size(&self) -> usize {
    self.layers.last().unwrap().len() - 1
  }

  /// Size of hidden layers.
  fn hidden_size(&self) -> usize {
    self.layers[1].len() - 1
  }

  /// Total number of layers in the network, including
  /// input layer, hidden layers, and output layer.
  fn layer_num(&self) -> usize {
    self.layers.len()
  }

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

  pub fn load(path: &str) -> std::io::Result<Net> {
    let mut f = std::fs::File::open(path)?;
    let dim = NetDim::load(&mut f)?;
    let mut net = Net::new(&dim, 0.0);
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
}


/// A neural net that can be used to map inputs to outpus.
pub struct NetRunner<'a> {
  net:    &'a Net,
  buf1:   Vec<R>,
  buf2:   Vec<R>
}

impl<'a> NetRunner<'a> {

  /// Create a new runner using the given weights.
  pub fn new(net: &'a Net) -> Self {
    let size = 1 + std::cmp::max(net.hidden_size(), net.output_size());
    NetRunner {
      net:  net,
      buf1: vec![0.0; size],
      buf2: vec![0.0; size]
    }
  }

  /// Set the weights for the runner.
  pub fn set_net(&mut self, net: &'a Net) {
    self.net = net;
    let size = 1 + std::cmp::max(net.hidden_size(), net.output_size());
    self.buf1.resize(size, 0.0);
    self.buf2.resize(size, 0.0);
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
              self.buf2.as_slice()
            } else {
              self.buf1.as_slice()
            };
    &r[1 ..= self.net.output_size()]
  }

}

/// A neural net in training.
pub struct NetLearner {
  net:      Net,            // weights
  d_layers: Net,            // weight gradients
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


impl NetLearner {

  /// Create a trainer for the given net.
  pub fn new(net: Net) -> Self {
    let dim = net.dim();
    let size = std::cmp::max(dim.hidden, dim.outputs);
    let mut bufs = Vec::with_capacity(dim.hidden + 2);  // layer outputs
    for _ in 0 ..= dim.hidden { bufs.push(vec![0.0; dim.hidden_size + 1]) }
    bufs.push(vec![0.0; dim.outputs + 1]);
    NetLearner {
      net:        net,
      d_layers:   Net::new(&dim, 0.0),
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
    // as they don;t chnage
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
    for (l,dl) in self.net.iter_mut().zip(self.d_layers.iter_mut()) {
      for (n,dn) in l.iter_mut().zip(dl.iter_mut()) {
        for (w,dw) in n.iter_mut().zip(dn.iter_mut()) {
          *w += *dw/self.batches * self.learning_rate;
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
    let ins       = self.buffers.iter().map(|x| x.as_slice());
    let outs      = self.buffers.iter().skip(1).map(|x| x.as_slice());
    let ls        = self.net.iter().rev();
    let dls       = self.d_layers.iter_mut().rev();
    let mut steps = ins.zip(outs).zip(ls).zip(dls);

    let (((last_is, last_os), last_ns), last_dns) = steps.next().unwrap();
    let last_os1 = &last_os[1..];

    sel_dx(last_os1, self.gbuf1.as_mut_slice());
    last_layer_dw(last_is, last_os1, self.gbuf1.as_slice(), last_dns.as_mut_slice());
    layer_dx_last(last_ns.as_slice(), last_os1, self.gbuf1.as_slice(), self.gbuf2.as_mut_slice());

    let mut swap = false;
    for (((xs,ys), ns), dns) in steps {
      let ys1 = &ys[1..];
      let (rd,wt) = if swap { (self.gbuf1.as_slice(), self.gbuf2.as_mut_slice()) }
                       else { (self.gbuf2.as_slice(), self.gbuf1.as_mut_slice()) };
      layer_dw(xs, ys1, rd, dns.as_mut_slice());
      layer_dx(ns.as_slice(), ys, rd, wt);
      swap = !swap;
    }
  }

  /// The current state of the net
  pub fn get_net(&self) -> &Net { &self.net }

  /// Extract net and destroy training infrastructure.
  pub fn complete(self) -> Net { self.net }
}

