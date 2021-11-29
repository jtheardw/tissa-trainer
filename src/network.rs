use std::fs::File;
use std::io::prelude::*;

use crate::cost::*;
use crate::functions::*;
use crate::gradient::*;
use crate::matrix::*;

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
    pub activations: Vec<Matrix>,
    pub errors: Vec<Matrix>,
    pub weight_gradients: Vec<Gradients>,
    pub bias_gradients: Vec<Gradients>
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
            errors: errors,
            weight_gradients: weight_gradients,
            bias_gradients: bias_gradients
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
            weight_gradients: weight_gradients,
            bias_gradients: bias_gradients
        }
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

    pub fn predict(&mut self, input: &Vec<i16>) -> f32 {
        // hidden layers are done using relu's.  The output
        // layer will be a sigmoid
        self.activations[0].clear();
        let output = &mut self.activations[0];
        let weight = &mut self.weights[0];
        let bias = &mut self.biases[0];

        // The first layer will be updated "sparsely".
        // With this network setup, only a small fraction of our inputs
        // can be non-zero at any given time. we don't have to do the whole
        // song-and-dance of the full matrix multiplication.
        let output_size = output.size();
        for i in 0..input.len() {
            // the "inputs" here are actually the indexes of the features that are non-zero
            for j in 0..output_size {
                self.activations[0].data[j] += weight.get(j, input[i] as usize);
            }
        }

        // we've gathered up the total sums (besides the bias).
        // now we need to add in the bias and run it through our relu
        for j in 0..output_size {
            self.activations[0].data[j] = relu(self.activations[0].data[j] + bias.data[j]);
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

                // if layer != self.activations.len() - 1 {
                    self.activations[layer].data[i] = activation_fn(self.activations[layer].data[i] + self.biases[layer].data[i]);
                // } else {
                //     self.activations[layer].data[i] = self.activations[layer].data[i] + self.biases[layer].data[i];
                // }
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

    pub fn update_gradients(&mut self, input: &Vec<i16>) {

        // the first layer is handled sparesly, as described in "predict"
        for i in 0..input.len() {
            for j in 0..self.errors[0].size() {
                self.weight_gradients[0].update(j, input[i] as usize, self.errors[0].data[j]);
            }
        }

        for i in 0..self.errors[0].size() {
            self.bias_gradients[0].data[i].update(self.errors[0].data[i]);
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

    pub fn train(&mut self, input: &Vec<i16>, eval_target: f32, result_target: f32) -> f32 {

        // First we attempt to get a result from the current net
        let last_output = self.predict(input);

        // Then we need to figure out how effective we were
        let output_gradient = cost_gradient(last_output, eval_target, result_target) * sigmoid_to_sigmoid_prime(last_output);

        // use the result to determine the errors on the inner layers
        self.find_errors(output_gradient);

        // determine updates
        self.update_gradients(input);

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
