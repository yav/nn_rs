use nn;

fn set_input(buf: &mut [nn::R], x: usize) {
  for bit in 0 .. 8 {
    buf[bit] = if bit == x { 1.0 } else { 0.0 };
  }
}

fn show_net(run: nn::RunnerEmpty, net: &nn::Weights) -> nn::RunnerEmpty {
  let mut r = run.set_weights(net);
  for x in 0 .. 8 {
    //set_input(r.set_input(), x);
    print!(" ");
    r.eval();
    let y = r.get_output()[0];
    let expect = (x as nn::R) / 8.0;
    println!("Input: {} -> {}, err = {}", x, y, nn::sel(r.get_output(),&[expect]));
  }
  r.clear_net()
}

pub fn main() {
  let mut n = nn::Weights::new(nn::Dim { 
     inputs: 8,
     outputs: 1,
     hidden: 1,
     hidden_size: 5
  }, 0.5);
  println!("\n{:?}\n", n.dim());
  //n.randomize(0.0,1.0);
  n.print();

  let mut r = nn::RunnerEmpty::new();
  r = show_net(r, &n);
  println!("");

  let mut l = nn::Learner::new(n);
  l.learning_rate = 0.001;

  for e in 0 .. 1_000_000 {
    if e % 10000 == 0 { println!("{}", e); r = show_net(r, l.get_weights()); println!("---"); }
    for x in 0 .. 8 {
      set_input(l.set_input(),x);
      l.set_output()[0] = (x as nn::R) / 8.0;
      l.train();
      l.finish_batch();
    }
    
  }

  n = l.complete();
  show_net(r, &n);
  n.print();
  
}