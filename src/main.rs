use std::arch::x86_64::*;

use rand::distributions::{Distribution, Uniform};
use std::fs::File;
use std::io::prelude::*;

mod cost;
mod dataset;
mod functions;
mod gradient;
mod matrix;
mod network;
mod trainer;

use crate::dataset::*;
use crate::functions::*;
use crate::network::*;
use crate::trainer::*;

fn main() {
    let mut topo = NetworkTopology::new(768 * 4, 1, vec![512]);
    // let mut data = load_dataset("../training_data/mantissa_self_play.epd").unwrap();
    // let mut data = load_dataset("/home/jtwright/training_data/mantissa_sep_self_play.epd").unwrap();
    // let mut data = load_dataset("/home/jtwright/chess/training_data/self_play2.epd").unwrap();
    let mut data = load_dataset("/home/jtwright/chess/tissa-trainer/foo.epd").unwrap();

    // let mut data = load_dataset("/home/jtwright/training_data/small.epd").unwrap();
    let mut net = Network::new(221016, topo);
    // let mut net = Network::load("nets/epoch-103.nnue").unwrap();
    let mut trainer = Trainer::new(net, data, 500);
    trainer.train();
}
