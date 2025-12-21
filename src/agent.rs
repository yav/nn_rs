pub use crate::common::R;
use crate::{Trainer};
use crate::norm::{ISigmoid,ORVec};

/// Serialize to a sequence of numbers.
pub trait Serialize {
  /// Encode the value in the given buffer and return how many entries we used up.
  fn serialize(&self, buf: &mut [R]) -> usize;
}

/// Stateful transition systems.
pub trait Stateful: Serialize {
  /// The type of actions for this transition system.
  type Action: Clone + Serialize;

  /// Transition to the next state, using the given action, and return some reward.
  fn next_state(&mut self, action: Self::Action) -> R;
}


type Norm = (ISigmoid, ORVec);

/// The state of an agent in training.
pub struct TrainingAgent {
    /// How much weight to give to new information.
    /// The value should be in the interval `[0,1]`, where 0 indicates that
    /// we ignore new information, and 1 indicates that we should completely
    /// ignore the current reward estimate.
    /// Note that this is different than the learning rate of neural net,
    /// which determines how far we go down a computed gradient.
    pub learning_rate: R,

    /// How much weight to give to future rewards.
    /// This value should be in the interval `[0,1]`, where 0 indicates that
    /// we should completely ignore future estimates and only consider the
    /// immediate reward.
    pub discount: R,          


    first: bool,              // True if we have not yet made any decisions
    decider: Trainer<Norm>,   // Use this to make decisions
    trainee: Trainer<Norm>    // Use this to learn
}

impl TrainingAgent {

  /// Create a new agent using the given neural net for training.
  pub fn new(d: Trainer<Norm>) -> TrainingAgent {
    TrainingAgent {
      learning_rate: 0.5, // XXX
      discount: 0.8, // XXX
      first: true,
      decider: d.clone(),
      trainee: d
    }
  }

  /// Pick one of the given actions and transitions to next state based on it.
  pub fn pick_action<S: Stateful>(&mut self, state: &mut S, actions: impl Iterator<Item=S::Action>) {
    let mut reward  = R::NEG_INFINITY;
    let mut act     = None;
    let inp_size    = state.serialize(self.decider.set_input());

    for a in actions {
      let d_act_inp = &mut self.decider.set_input()[inp_size .. ];
      a.serialize(d_act_inp);
      self.decider.eval();
      let new_reward = self.decider.get_output()[0];
      if new_reward >= reward {
        reward = new_reward;
        act = Some(a.clone());
      }
    }
    if !self.first {
      self.trainee.set_output()[0] += self.learning_rate * self.discount * reward;
      self.trainee.train();
    } else {
      self.first = false;
    }
    
    let t_inp = self.trainee.set_input();
    let (t_state_inp, t_act_inp) = t_inp.split_at_mut(inp_size);
    t_state_inp.copy_from_slice(&self.decider.set_input()[0..inp_size]);
    let choice = act.unwrap();
    choice.serialize(t_act_inp);
    let new_reward = state.next_state(choice);
    self.trainee.set_output()[0] = (1.0 - self.learning_rate) * reward + self.learning_rate * new_reward;
  }

  /// Notify agent that the run is finished.
  pub fn final_reward(&mut self, reward: R) {
    if self.first { return }
    self.trainee.set_output()[0] += self.learning_rate * self.discount * reward;
    self.trainee.train();
    self.trainee.finish_batch();
    std::mem::swap(&mut self.trainee, &mut self.decider);
    self.first = true;
  }

}