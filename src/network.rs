use std::fs::File;
use std::io::BufWriter;
use std::io::prelude::*;

use crate::cost::*;
use crate::dataset::*;
use crate::functions::*;
use crate::gradient::*;
use crate::matrix::*;


fn input_number(piece: i32, white: bool, rank: i32, file: i32) -> i16 {
    let piece_num = if white { piece } else { 6 + piece };
    let idx = rank * 8 + file;
    let num = (piece_num * 64 + idx) as i16;
    if num > 769 {
        panic!("num too big {}, {} {} {} {}", num, piece, white, rank, file);
    }
    return num;
}

fn flip_input(input: i16) -> i16 {
    // need to flip "color" and "rank"
    // flip square
    let idx = (input % 64) ^ 56;
    let orig_piece_num = input / 64;
    let piece_type = orig_piece_num % 6;
    // >= 6 meant it was black, so we need to undo that and vice versa
    let piece_color = orig_piece_num >= 6;
    let piece_num = if piece_color {piece_type} else {6 + piece_type};

    return piece_num * 64 + idx;
}

pub struct NetworkTopology {
    pub inputs: u32,
    pub outputs: u32,
    pub hidden: Vec<u32>
}

pub struct Network {
    pub id: u32,
    pub topology: NetworkTopology,
    pub weights: Vec<Matrix>,
    pub biases: Vec<Matrix>,
    pub sums: Matrix,
    pub activations: Vec<Matrix>,
    pub errors: Vec<Matrix>,
    pub weight_gradients: Vec<Gradients>,
    pub bias_gradients: Vec<Gradients>,
    flipped: bool
}

impl NetworkTopology {
    pub fn new(inputs: u32, outputs: u32, hidden: Vec<u32>) -> NetworkTopology {
        NetworkTopology {
            inputs: inputs,
            outputs: outputs,
            hidden: hidden
        }
    }

    pub fn copy(&self) -> NetworkTopology {
        NetworkTopology {
            inputs: self.inputs,
            outputs: self.outputs,
            hidden: self.hidden.to_vec()
        }
    }
}

impl Network {
    pub fn new(id: u32, topology: NetworkTopology) -> Network {
        let mut activations: Vec<Matrix> = Vec::new();
        let mut weights: Vec<Matrix> = Vec::new();
        let mut biases: Vec<Matrix> = Vec::new();
        let mut errors: Vec<Matrix> = Vec::new();
        let mut weight_gradients: Vec<Gradients> = Vec::new();
        let mut bias_gradients: Vec<Gradients> = Vec::new();

        let num_layers = topology.hidden.len() + 1;
        let mut input_size = topology.inputs as usize;

        for i in 0..num_layers {
            let output_size = if i == topology.hidden.len() {
                // last layer
                topology.outputs
            } else {
                topology.hidden[i]
            } as usize;

            weights.push(Matrix::new_random(output_size, input_size, topology.inputs as f32));
            biases.push(Matrix::new_random_column(output_size, topology.inputs as f32));
            weight_gradients.push(Gradients::new(output_size, input_size));
            bias_gradients.push(Gradients::new(output_size, 1));
            activations.push(Matrix::new_random_column(output_size, topology.inputs as f32));
            errors.push(Matrix::new_random_column(output_size, topology.inputs as f32));

            input_size = output_size;
        }

        Network {
            id: id,
            topology: topology.copy(),
            weights: weights,
            biases: biases,
            activations: activations,
            sums: Matrix::new_random_column(topology.inputs as usize, topology.inputs as f32),
            errors: errors,
            weight_gradients: weight_gradients,
            bias_gradients: bias_gradients,
            flipped: false
        }
    }

    pub fn copy(&self) -> Network {
        let mut activations: Vec<Matrix> = Vec::new();
        let mut errors: Vec<Matrix> = Vec::new();
        let mut weight_gradients: Vec<Gradients> = Vec::new();
        let mut bias_gradients: Vec<Gradients> = Vec::new();

        let num_layers = self.topology.hidden.len() + 1;
        let mut input_size = self.topology.inputs as usize;

        for i in 0..num_layers {
            let output_size = if i == self.topology.hidden.len() {
                // last layer
                self.topology.outputs
            } else {
                self.topology.hidden[i]
            } as usize;

            weight_gradients.push(self.weight_gradients[i].copy());
            bias_gradients.push(self.bias_gradients[i].copy());
            activations.push(self.activations[i].copy());
            errors.push(self.errors[i].copy());

            input_size = output_size;
        }

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for w in &self.weights {
            weights.push(w.copy());
        }
        for b in &self.biases {
            biases.push(b.copy());
        }
        Network {
            id: self.id,
            topology: self.topology.copy(),
            weights: weights,
            biases: biases,
            activations: activations,
            errors: errors,
            sums: self.sums.copy(),
            weight_gradients: weight_gradients,
            bias_gradients: bias_gradients,
            flipped: self.flipped
        }
    }

    pub fn save_image(&self, file: &str) -> std::io::Result<()> {
        let ss = 8; // supersampling
        let border = [0, 0, 0];
        let width  = (12*8*4)*ss + 11 * 4 + 3 * ss * 4;
        let neurons = 256;
        let height = 8*neurons*ss + 127;
        let mut w = BufWriter::new(File::create(format!("{}.ppm", file))?);
        writeln!(&mut w, "P6")?;
        writeln!(&mut w, "{} {}", width, height)?;
        writeln!(&mut w, "255")?;

        // for n in 0..128 {
        for n in 0..neurons {
            if n != 0 { for _ in 0..width { w.write(&border)?; } }
            let mut upper = 0.0;
            let mut lower = 0.0;
            for rank in (0..8).rev() {
                for _ in 0..ss {
                    let mut wid = 0;
                    for kr in 0..4 {
                        if kr != 0 {
                            for _ in 0..(ss*4) { wid += 1; w.write(&border)?; }
                        }

                        let kr_offset = kr * 768;
                        for i in 0..768 {
                            upper = self.weights[0].get(n, kr_offset + i).max(upper);
                            lower = self.weights[0].get(n, kr_offset + i).min(lower);
                        }
                        let scale = upper.max(-lower);

                        for side in [true, false] {
                            for piece in (0..6).rev() {
                                if piece != 5 || !side { wid += 1; w.write(&border)?; }
                                for file in 0..8 {
                                    // let x : usize = piece*64 + rank*8 + file;
                                    let inp = input_number(piece, side, rank, file) + kr_offset as i16;
                                    let normed = ((self.weights[0].get(n, inp as usize) / scale) + 1.0) / 2.0;
                                    // let normed = ((self.w1[n][x] / scale) + 1.0) / 2.0;
                                    debug_assert!(1.0 >= normed && normed >= 0.0, "out of range");
                                    let r = (normed * 255.0).round() as u8;
                                    let g = (normed * 255.0).round() as u8;
                                    let b = (32.0 + normed * 191.0).round() as u8;
                                    for _ in 0..ss { wid += 1; w.write(&[r, g, b])?; }
                                }
                            }
                        }
                    }
                }
            }
        }
        return Ok(());
    }

    // From zahak-trainer:
    // Binary specification for the NNUE file:
    // - All the data is stored in little-endian layout
    // - All the matrices are written in column-major
    // - The magic number/version consists of 4 bytes (int32):
    //   - 66 (which is the ASCII code for B), uint8
    //   - 90 (which is the ASCII code for Z), uint8
    //   - 2 The major part of the current version number, uint8
    //   - 0 The minor part of the current version number, uint8
    // - 4 bytes (int32) to denote the network ID
    // - 4 bytes (int32) to denote input size
    // - 4 bytes (int32) to denote output size
    // - 4 bytes (int32) number to represent the number of inputs
    // - 4 bytes (int32) for the size of each layer
    // - All weights for a layer, followed by all the biases of the same layer
    // - Other layers follow just like the above point
    pub fn save(&self, fname: &str) -> std::io::Result<()> {
        let mut file = File::create(fname)?;

        // write magic number
        file.write(&[66, 90, 2, 0])?;

        // write network ID
        file.write(&self.id.to_le_bytes())?;

        // write topology
        file.write(&self.topology.inputs.to_le_bytes())?;
        file.write(&self.topology.outputs.to_le_bytes())?;
        file.write(&(self.topology.hidden.len() as u32).to_le_bytes())?;
        for i in 0..self.topology.hidden.len() {
            file.write(&self.topology.hidden[i].to_le_bytes())?;
        }

        // write weights and biases
        for i in 0..self.activations.len() {
            for weight in &self.weights[i].data {
                file.write(&weight.to_le_bytes())?;
            }
            for bias in &self.biases[i].data {
                file.write(&bias.to_le_bytes())?;
            }
        }

        return Ok(());
    }

    pub fn load(fname: &str) -> std::io::Result<Network> {
        let mut file = File::open(fname)?;

        // We'll read one 4-byte "word" at a time
        let mut buf: [u8; 4] = [0; 4];

        // magic number
        file.read(&mut buf)?;
        if buf[0..2] != [66, 90] { panic!("Magic word does not match expected 'BZ'"); }
        if buf[2..4] != [2, 0] { panic!("Network binary format version is not supported."); }

        // network id
        file.read(&mut buf)?;
        let id = u32::from_le_bytes(buf);

        // topology information
        file.read(&mut buf)?;
        let inputs = u32::from_le_bytes(buf);
        file.read(&mut buf)?;
        let outputs = u32::from_le_bytes(buf);
        file.read(&mut buf)?;
        let layers = u32::from_le_bytes(buf);

        // hidden neuron counts in layers
        let mut hidden: Vec<u32> = Vec::new();
        for _ in 0..layers {
            file.read(&mut buf)?;
            hidden.push(u32::from_le_bytes(buf));
        }

        let network_topology = NetworkTopology::new(inputs, outputs, hidden);
        let mut network = Network::new(id, network_topology);

        for i in 0..network.activations.len() {
            for j in 0..network.weights[i].size() {
                file.read(&mut buf)?;
                network.weights[i].data[j] = f32::from_le_bytes(buf);
            }

            for j in 0..network.biases[i].size() {
                file.read(&mut buf)?;
                network.biases[i].data[j] = f32::from_le_bytes(buf);
            }
        }

        return Ok(network);
    }

    pub fn predict(&mut self, input: &[i16; 33], len: u8, wk_loc: i16, bk_loc: i16) -> f32 {
        // hidden layers are done using relu's.  The output
        // layer will be a sigmoid
        self.activations[0].clear();
        self.sums.clear();
        self.flipped = true;
        let output = &mut self.activations[0];
        let weight = &mut self.weights[0];
        let bias = &mut self.biases[0];

        // The first layer will be updated "sparsely".
        // With this network setup, only a small fraction of our inputs
        // can be non-zero at any given time. we don't have to do the whole
        // song-and-dance of the full matrix multiplication.
        let output_size = output.size();
        self.flipped = !(input[(len - 1) as usize] == -1);

        for i in 0..len {
            // the "inputs" here are actually the indexes of the features that are non-zero
            if input[i as usize] == -1 {
                continue;
            }
            let inp = input[i as usize] + 768 * wk_loc;
            let flipped_inp = flip_input(input[i as usize]) + 768 * bk_loc;
            // if !self.flipped && (input[i as usize] / 64) == 5 {
            //     println!("king idx {} kr {}", inp % 64, wk_loc);
            // } else if self.flipped && (input[i as usize] / 64) == 11 {
            //     println!("flipped king idx {} kr {}", flipped_inp % 64, bk_loc);
            // }
            let neurons = output_size / 2;
            for j in 0..output_size {
                let out_idx = j % (output_size / 2);
                let weight_to_get = if j >= output_size / 2 {
                    // we need to "flip"
                    flipped_inp as usize
                } else {
                    inp as usize
                };
                // println!("inp {} neuron {} weight_j {} weight_i {}", input[i as usize], j, weight_to_get.0, weight_to_get.1);
                if !self.flipped {
                    self.activations[0].data[j] += weight.get(out_idx, weight_to_get)
                } else {
                    // self.activations[0].data[j ^ 128] += weight.get(out_idx, weight_to_get);
                    self.activations[0].data[j ^ neurons] += weight.get(out_idx, weight_to_get);
                }
                // self.sums.data[j] += weight.get(out_idx, weight_to_get);
            }
        }

        // we've gathered up the total sums (besides the bias).
        // now we need to add in the bias and run it through our relu
        for j in 0..output_size {
            self.activations[0].data[j] = relu(self.activations[0].data[j] + bias.data[j % (output_size / 2)]);
            // if !self.flipped {
            //     self.activations[0].data[j] = self.sums.data[j];
            // } else {
            //     self.activations[0].data[j ^ 128] = self.sums.data[j];
            // }
        }

        let mut activation_fn: fn(f32) -> f32 = relu;
        // the other layers will be updated via matrix multiplication
        for layer in 1..self.activations.len() {
            // let input = &mut self.activations[layer-1];
            // let output = &mut self.activations[layer];
            // let weight = &mut self.weights[layer];
            // let bias = &mut self.biases[layer];
            if layer == self.activations.len() - 1 {
                activation_fn = sigmoid;
            }

            for i in 0..self.activations[layer].size() {
                self.activations[layer].data[i] = 0.0;
                for j in 0..self.activations[layer-1].size() {
                    self.activations[layer].data[i] += self.activations[layer-1].data[j] * self.weights[layer].get(i, j);
                }

                self.activations[layer].data[i] = activation_fn(self.activations[layer].data[i] + self.biases[layer].data[i]);
            }
        }

        return self.activations[self.activations.len() - 1].data[0];
    }

    pub fn find_errors(&mut self, output_gradient: f32) {
        let last = self.activations.len() - 1;
        self.errors[last].data[0] = output_gradient;

        for layer in (0..last).rev() {
            for i in 0..self.errors[layer].size() {
                self.errors[layer].data[i] = 0.0;
                for j in 0..self.errors[layer + 1].size() {
                    self.errors[layer].data[i] += self.errors[layer + 1].data[j] * self.weights[layer + 1].get(j, i) * relu_prime(self.activations[layer].data[i]);
                }
            }
        }
    }

    pub fn update_gradients(&mut self, input: &[i16; 33], len: u8, wk_loc: i16, bk_loc: i16) {

        // the first layer is handled sparesly, as described in "predict"
        for i in 0..len {
            if input[i as usize] == -1 { continue; }
            let inp = input[i as usize] + 768 * wk_loc;
            let output_size = self.errors[0].size();
            let flipped_inp = flip_input(input[i as usize]) + 768 * bk_loc;
            for j in 0..output_size {
                let out_idx = j % (output_size / 2);
                let weight_to_get = if self.flipped ^ (j >= output_size / 2) {
                    flipped_inp as usize
                } else {
                    inp as usize
                };
                self.weight_gradients[0].update(out_idx, weight_to_get, self.errors[0].data[j]);
            }
        }

        for i in 0..self.errors[0].size() {
            self.bias_gradients[0].data[i % (self.errors[0].size() / 2)].update(self.errors[0].data[i]);
        }

        for layer in 1..self.activations.len() {
            for i in 0..self.weight_gradients[layer].rows {
                let err = self.errors[layer].data[i];
                self.bias_gradients[layer].update(i, 0, err);
                for j in 0..self.weight_gradients[layer].cols {
                    let gradient = self.activations[layer - 1].data[j] * err;
                    self.weight_gradients[layer].update(i, j, gradient)
                }
            }
        }
    }

    pub fn train(&mut self, input: &[i16; 33], len: u8, wk_loc: i16, bk_loc: i16, eval_target: f32, result_target: f32) -> f32 {

        // First we attempt to get a result from the current net
        let last_output = self.predict(input, len, wk_loc, bk_loc);

        // Then we need to figure out how effective we were
        let output_gradient = cost_gradient(last_output, eval_target, result_target) * sigmoid_to_sigmoid_prime(last_output);

        // use the result to determine the errors on the inner layers
        self.find_errors(output_gradient);

        // determine updates
        self.update_gradients(input, len, wk_loc, bk_loc);

        let vc = validation_cost(last_output, eval_target, result_target);
        return vc;
    }

    pub fn apply_gradients(&mut self) {
        for i in 0..self.activations.len() {
            self.bias_gradients[i].apply(&mut self.biases[i]);
            self.weight_gradients[i].apply(&mut self.weights[i]);
        }
    }
}
