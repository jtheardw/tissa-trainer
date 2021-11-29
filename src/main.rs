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
    let mut topo = NetworkTopology::new(769, 1, vec![128]);
    let mut data = load_dataset("../epds/training_data.epd").unwrap();
    let mut net = Network::new(1234, topo);
    let mut trainer = Trainer::new(net, data, 500);
    trainer.train();
}
