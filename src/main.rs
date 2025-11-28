use nn;



fn check(buf: nn::RunnerEmpty, net: &nn::Learner) -> (nn::R, nn::RunnerEmpty) {
  let mut r = buf.set_weights(net.get_weights());
  r.set_input()[0] = 0.0;
  r.eval();
  (nn::loss(r.get_output(),&[0.0]), r.clear_net())
}

pub fn main() {
  let mut n = nn::Weights::new(nn::Dim { 
     inputs: 1,
     outputs: 1,
     hidden: 0,
     hidden_size: 1
  }, 0.0);
  println!("\n{:?}\n", n.dim());
  n.randomize(0.0,1.0);
  // n.print();

  let mut r = nn::RunnerEmpty::new();

  let mut l = nn::Learner::new(n);
  l.learning_rate = 1.0;

  
  for e in 0 .. 1000000 {
    //println!("Step {}", e);
    //l.get_weights().print();
    let (err1,r1) = check(r, &l);
    r = r1;
    l.set_input()[0] = 0.0;
    l.set_output()[0] = 0.0;
    l.train();
    l.finish_batch();
    let (err2, r1) = check(r, &l);
    r = r1;
    println!("[{}] {} -> {}", (if err2 > err1 { "!!" } else { "OK" }), err1, err2);
    
  }

  n = l.complete();
  n.print();
  let mut r = r.set_weights(&n);
  r.set_input()[0] = 0.0;
  r.eval();
  println!("RES: {}", r.get_output()[0]);
}