use std::cmp;
use std::sync::mpsc::{self};
use std::sync::mpsc::{Sender, Receiver};
use std::thread;
use std::time::SystemTime;

use rand::Rng;

use crate::cost::*;
use crate::dataset::*;
use crate::functions::*;
use crate::network::*;

const NUM_THREADS: usize = 44;
// const BATCH_SIZE: usize = 16384;
const BATCH_SIZE: usize = 262144;

pub fn get_time_millis() -> u128 {
    match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
        Ok(n) => n.as_millis(),
        Err(_) => panic!("SystemTime failed!"),
    }
}

pub struct Sample {
    pub inputs: Vec<i16>
}

pub struct Trainer {
    pub nets: Vec<Network>,
    pub training_set: Vec<Data>,
    pub validation_set: Vec<Data>,
    pub epochs: i32,
    pub validation_costs: Vec<f32>,
    pub training_costs: Vec<f32>
}

fn cost_thread_handler(mut net: Network, batch: Vec<Data>, chan: Sender<f32>, net_chan: Sender<Network>) {
    let mut local_cost = 0.0;
    for d in 0..batch.len() {
        let data = &batch[d];

        let predicted = net.predict(&data.input, data.len);
        let cost = validation_cost(predicted, sigmoid(data.score as f32), data.outcome as f32 / 2.0);

        local_cost += cost;
    }
    chan.send(local_cost).unwrap();
    net_chan.send(net.copy()).unwrap();
}

fn epoch_thread_handler(mut net: Network, batch: Vec<Data>, chan: Sender<f32>, net_chan: Sender<Network>) {
    let mut local_cost = 0.0;
    for d in 0..batch.len() {
        let data = &batch[d];

        local_cost += net.train(&data.input, data.len, sigmoid(data.score as f32), data.outcome as f32 / 2.0);
    }
    chan.send(local_cost).unwrap();
    net_chan.send(net.copy()).unwrap();
}

impl Trainer {
    pub fn shuffle_dataset(dataset: &mut Vec<Data>) {
        for i in 0..dataset.len() {
            let j = rand::thread_rng().gen_range(0..dataset.len()) as usize;
            dataset.swap(i, j);
        }
    }

    pub fn new(network: Network, dataset: Vec<Data>, epochs: i32) -> Trainer {
        let validation_size = cmp::min(20 * dataset.len() / 100, 20000000) as usize;
        let validation_set = dataset[..validation_size].to_vec();
        let training_set = dataset[validation_size..].to_vec();

        let mut networks = Vec::new();
        for _ in 0..NUM_THREADS {
            networks.push(network.copy());
        }

        Trainer {
            nets: networks,
            training_set: training_set,
            validation_set: validation_set,
            epochs: epochs,
            training_costs: vec![0.0; epochs as usize],
            validation_costs: vec![0.0; epochs as usize]
        }
    }

    pub fn copy_nets(&mut self) {
        for i in 1..self.nets.len() {
            let mut new_weights = Vec::new();
            for w in &self.nets[0].weights {
                new_weights.push(w.copy());
            }
            let mut new_biases = Vec::new();
            for b in &self.nets[0].biases {
                new_biases.push(b.copy());
            }
            self.nets[i].weights = new_weights;
            self.nets[i].biases = new_biases;
        }
    }

    pub fn shuffle_training(&mut self) {
        for i in 0..self.training_set.len() {
            let j = rand::thread_rng().gen_range(0..self.training_set.len()) as usize;
            self.training_set.swap(i, j);
        }
    }

    pub fn sync_gradients(&mut self) {
        let num_nets = self.nets.len();
        let num_layers = self.nets[0].activations.len();
        for i in 0..num_nets {
            for j in 0..num_layers {
                let num_wgrads = self.nets[0].weight_gradients[j].size();
                for k in 0..num_wgrads {
                    let val = self.nets[i].weight_gradients[j].data[k].value;
                    self.nets[0].weight_gradients[j].data[k].update(val);
                    self.nets[i].weight_gradients[j].data[k].reset();
                }

                let num_bgrads = self.nets[0].bias_gradients[j].size();
                for k in 0..num_bgrads {
                    let val = self.nets[i].bias_gradients[j].data[k].value;
                    self.nets[0].bias_gradients[j].data[k].update(val);
                    self.nets[i].bias_gradients[j].data[k].reset();
                }
            }
        }
    }

    pub fn print_cost(&mut self) -> f32 {
        println!("Starting the validation of the epoch");
        let mut total_cost: f32 = 0.0;

        let thread_batch_size = self.validation_set.len() / NUM_THREADS;
        let (answer_tx, answer_rx) = mpsc::channel();
        let mut threads = Vec::new();
        let mut net_chans = Vec::new();

        for i in 0..NUM_THREADS {
            let mut batch = self.validation_set[i*thread_batch_size..(i+1)*thread_batch_size].to_vec();
            let tx = answer_tx.clone();
            let (chan_tx, chan_rx) = mpsc::channel();
            let mut net = self.nets[i].copy();
            threads.push(thread::spawn(|| {
                cost_thread_handler(net, batch, tx, chan_tx);
            }));
            net_chans.push(chan_rx);
        }

        for i in 0..NUM_THREADS {
            total_cost += answer_rx.recv().unwrap();
            self.nets[i] = net_chans[i].recv().unwrap();
        }

        for t in threads {
            t.join();
        }
        let avg_cost = total_cost / (self.validation_set.len() as f32);
        println!("Current validation cost is: {}", avg_cost);
        return avg_cost;
    }

    pub fn start_epoch(&mut self, start_time: u128) -> f32 {
        let mut batch_end = BATCH_SIZE;
        let mut samples = 0;
        let mut total_cost = 0.0;
        while batch_end < self.training_set.len() {
            let mut new_batch = self.training_set[(batch_end - BATCH_SIZE)..batch_end].to_vec();
            let thread_batch_size = new_batch.len() / NUM_THREADS;
            let (answer_tx, answer_rx) = mpsc::channel();
            let mut threads = Vec::new();
            let mut net_chans = Vec::new();

            // let mut nets = self.nets.iter_mut();
            for i in 0..NUM_THREADS {
                let mut batch = new_batch[i*thread_batch_size..(i+1)*thread_batch_size].to_vec();
                let tx = answer_tx.clone();
                let mut net = self.nets[i].copy();
                let (chan_tx, chan_rx) = mpsc::channel();
                threads.push(thread::spawn(|| {
                    epoch_thread_handler(net, batch, tx, chan_tx);
                }));
                net_chans.push(chan_rx);
            }

            for i in 0..NUM_THREADS {
                total_cost += answer_rx.recv().unwrap();
                self.nets[i] = net_chans[i].recv().unwrap();
                samples += thread_batch_size;
            }

            for t in threads {
                t.join();
            }

            let speed = (samples * 1000) as u128 / (get_time_millis() - start_time);
            print!("\rTrained on {} samples [ {} samples / second ]", samples, speed as u64);

            self.sync_gradients();
            self.nets[0].apply_gradients();
            self.copy_nets();
            batch_end += BATCH_SIZE;
        }

        return total_cost;
    }

    pub fn train(&mut self) {
        for epoch in 0..self.epochs {
            let start_time = get_time_millis();
            println!("Started Epoch {} at {}", epoch + 1, start_time / 1000);
            println!("Number of samples: {}", self.training_set.len());

            let total_cost = self.start_epoch(start_time);
            let finish_time = get_time_millis();
            println!("\nFinished Epoch {} at {}, approx. elapsed time {}s", epoch + 1, finish_time / 1000, (finish_time - start_time) / 1000);
            println!("Storing this Epoch {} network", epoch + 1);
            self.nets[0].save(&format!("nets/epoch-{}.nnue", epoch+1).as_str());
            self.nets[0].save_image(&format!("nets/epoch-{:03}", epoch+1).as_str());
            println!("Store this Epoch {} network", epoch + 1);

            let average_cost = total_cost / self.training_set.len() as f32;
            self.validation_costs[epoch as usize] = self.print_cost();
            self.training_costs[epoch as usize] = average_cost;

            println!("Current training cost is: {}", average_cost);
            println!("Training and Validation cost progression");
            println!("===================================================================================");
		    println!("Epoch\t\t\t\tTraining Cost\t\t\t\tValidation Cost");
            println!("===================================================================================");
            for e in 0..epoch+1 {
                println!("{}\t\t\t\t{}\t\t\t\t{}", e+1, self.training_costs[e as usize], self.validation_costs[e as usize]);
            }
            println!("===================================================================================");
            println!("Shuffling training dataset");
            self.shuffle_training();
        }
    }
}
