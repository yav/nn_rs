//! Functions for training and evaluating neural nets.


use std::io::{Read, Write};
use std::marker::PhantomData;
use rand::Rng;
use rand::distr::Uniform;
use crate::common::*;
use crate::norm::{Norm,Internal,Output};


type Layer  = Vec<Neuron>;  // a bunch of neurons sharing input

/// The type of net weights, inputs, and outputs.
pub type R = crate::common::R;

mod common;

/// Specifies how to normalize and validate the result of a network.
pub mod norm;
pub mod agent;


// Assumes `xs` contains a bias input
// Produces an additional bias result in the first slot of the output
fn layer<I: Internal>(ns: &[Neuron], xs: &[R], res: &mut [R]) {
  res[0] = 1.0;
  for (ws,tgt) in ns.iter().zip(res[1..].iter_mut()) {
    *tgt = I::neuron_norm(linear(ws.as_slice(), xs));
  }
}

// There's no bias in the last layer's output, and we normalize
// the outputs in a custom way.
fn last_layer<O: Output>(ns: &[Neuron], xs: &[R], res: &mut [R]) {
  for (ws,tgt) in ns.iter().zip(res.iter_mut()) {
    *tgt = linear(ws.as_slice(), xs)
  }
  O::normalize(res);
}



// We need to know how to do this to run a net.
trait NetRunner : Norm {
  // How many layers are in the net
  fn layer_num(&self) -> usize;

  // Get the given layer, its input, and its output buffer.
  fn get_in_out(&mut self, i: usize) -> (&[Neuron], &[R], &mut [R]);
}


// Common code for evaluating a net.  Used both by the runner and the trainer.
fn eval<R: NetRunner>(r: &mut R) {
  let last = r.layer_num() - 1;
  for i in 0 .. last {
    let (l, rd, wt) = r.get_in_out(i);
    layer::<R::INorm>(l, rd, wt);
  }
  let (l, rd, wt) = r.get_in_out(last);
  last_layer::<R::ONorm>(l, rd, wt);
}


// Update the error gradient to account for changes due to the actuators.
// `ys` is the results of the layer (i.e., the normalized value).
// The first element is bias
fn actuator_layer_delta<I: Internal>(ys: &[R], gs_dy: &mut [R]) {
  for i in 1 .. ys.len() {
    gs_dy[i - 1] *= I::neuron_norm_dy(ys[i]);
  }
}


/// Geometry of a neural net.
#[derive(Copy,Clone,Debug)]
pub struct Dim {
  /// Number of inputs to the net.
  pub inputs: usize,

  /// Number of results produced by the net.
  pub outputs: usize,

  /// How many hidden layers we have.
  pub hidden: usize,

  /// How big is each hidden layer.
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
#[derive(Clone)]
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

  fn layer(&self, i: usize) -> &[Neuron] { self.layers[i].as_slice() }

  /// Get the dimension of this net.
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

  /// Print the network to stdout.  This is mostly useful for debugging.
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
pub struct RunnerState { buf1: Vec<R>, buf2: Vec<R> }

impl RunnerState {
  pub fn new() -> Self {
    RunnerState { buf1: vec![], buf2: vec![] }
  }

  pub fn set_weights<N: Norm>(mut self, net: &Weights) -> Runner<N> {
    let size = 1 + std::cmp::max(net.input_size(), std::cmp::max(net.hidden_size(), net.output_size()));
    self.buf1.resize(size, 0.0);
    self.buf2.resize(size, 0.0);
    Runner { net:net, state: self, _norm: PhantomData }
  } 
}

impl<'a, N: Norm> Norm for Runner<'a, N> {
  type INorm = N::INorm;
  type ONorm = N::ONorm;
}

/// A neural net that can be used to map inputs to outputs.
pub struct Runner<'a, N: Norm> { net: &'a Weights, _norm: PhantomData<N>, state: RunnerState }

impl<'a, N: Norm> NetRunner for Runner<'a, N> {

  fn layer_num(&self) -> usize { self.net.layer_num() }

  /// Get the given layer, its inputs and outputs.
  fn get_in_out(&mut self, i: usize) -> (&[Neuron], &[R], &mut [R]) {
    if i & 1 == 0 {
      (self.net.layer(i), self.state.buf1.as_slice(), self.state.buf2.as_mut_slice())
    } else {
      (self.net.layer(i), self.state.buf2.as_slice(), self.state.buf1.as_mut_slice())
    }
  }
}

impl<'a, N: Norm> Runner<'a, N> {

  /// Crate a new runner using the given weights.
  pub fn new(ws: &'a Weights) -> Self { RunnerState::new().set_weights(ws) }

  /// Forget the weights, just keep the runner state for later use.
  pub fn clear_net(self) -> RunnerState { self.state }

  /// Get a reference to fill in the input to the net.
  pub fn set_input(&mut self) -> &mut[R] {
    let b = self.state.buf1.as_mut_slice();
    b[0] = 1.0;
    &mut b[1 ..= self.net.input_size()]
  }

  /// Evaluate the net on the current input.
  pub fn eval(&mut self) { eval(self) }

  /// Get the output of the net.
  pub fn get_output(&self) -> &[R] {
    let r = if self.net.layer_num() & 1 == 0 {
              self.state.buf1.as_slice()
            } else {
              self.state.buf2.as_slice()
            };
    &r[0 .. self.net.output_size()]
  }

}

#[derive(Clone)]
/// A neural net in training.
pub struct Trainer<N: Norm> {
  net:      Weights,        // weights
  d_layers: Weights,        // weight gradients
  batches: R,               // how many samples are in the (gradient for batching)

  /// Determines how much to change the net's state based on a batch
  /// of examples.
  pub learning_rate: R,

  // The first one is for input, the rest are for the outputs of the layers.
  // Note that all vectors are 1 longer to accommodate for bias.
  // The last vector contains the output, without bias.
  buffers: Vec<Vec<R>>,
  
  // Error gradient buffers (swap)
  gbuf1: Vec<R>,
  gbuf2: Vec<R>,

  // How to normalize output
  _norm: PhantomData<N>
}

impl<N: Norm> Norm for Trainer<N> {
  type INorm = N::INorm;
  type ONorm = N::ONorm;
}

impl<N: Norm> NetRunner for Trainer<N> {

  fn layer_num(&self) -> usize { self.net.layer_num() }
  
  fn get_in_out(&mut self, i: usize) -> (&[Neuron], &[R], &mut [R]) {
    let (xs,ys) = self.buffers.as_mut_slice().split_at_mut(i+1);
    (self.net.layer(i), xs.last().unwrap().as_slice(), ys[0].as_mut_slice())
  }
}

impl<N: Norm> Trainer<N> {

  /// Create a trainer for the given net.
  pub fn new(net: Weights) -> Self {
    let dim = net.dim();
    let size = std::cmp::max(dim.hidden, dim.outputs);
    let mut bufs = Vec::with_capacity(1 + net.layer_num()); // 1 extra fro input
    bufs.push(vec![0.0; dim.inputs + 1]);
    for _ in 0 ..= dim.hidden { bufs.push(vec![0.0; dim.hidden_size + 1]) }
    bufs.push(vec![0.0; dim.outputs + 1]);
    Trainer {
      net:        net,
      d_layers:   Weights::new(dim, 0.0),
      batches:    0.0,

      learning_rate: 0.1,

      buffers:    bufs,
      gbuf1:      vec![0.0; size],
      gbuf2:      vec![0.0; size],
      _norm:      PhantomData
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

  /// Evaluate the net on the current input.
  pub fn eval(&mut self) { eval(self) }

  /// Get the results of evaluating the net.
  /// Only valid after calling [Self::eval] or [Self::train].
  pub fn get_output(&self) -> &[R] {
    let xs = self.buffers.last().unwrap().as_slice();
    &xs[0 .. self.net.output_size()]
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
  /// The amount of change depends on the [learning rate][`Trainer::learning_rate`].
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


  fn backprop(&mut self) {
    let ins       = self.buffers.iter().map(|x| x.as_slice()).rev();
    let outs      = self.buffers.iter().skip(1).map(|x| x.as_slice()).rev();
    let ls        = self.net.iter().rev();
    let dls       = self.d_layers.iter_mut().rev();
    let mut steps = ins.zip(outs).zip(ls).zip(dls);

    let (((last_is, last_os), last_ns), last_dns) = steps.next().unwrap();

    N::ONorm::error_delta(&last_os[0..self.net.output_size()], self.gbuf1.as_mut_slice());
    last_lin_layer_dw(last_is, self.gbuf1.as_slice(), last_dns.as_mut_slice());
    last_lin_layer_dx(last_ns.as_slice(), self.gbuf1.as_slice(), self.gbuf2.as_mut_slice());

    let mut swap = false;
    for (((xs,ys), ns), dns) in steps {
      let (cur,next) = if swap { (self.gbuf1.as_mut_slice(), self.gbuf2.as_mut_slice()) }
                          else { (self.gbuf2.as_mut_slice(), self.gbuf1.as_mut_slice()) };
      actuator_layer_delta::<N::INorm>(ys, cur);
      lin_layer_dw(xs,            cur, dns.as_mut_slice());
      lin_layer_dx(ns.as_slice(), cur, next);
      swap = !swap;
    }
  }

  /// The current state of the net.
  pub fn get_weights(&self) -> &Weights { &self.net }

  /// Extract net and destroy training infrastructure.
  pub fn complete(self) -> Weights { self.net }
}

