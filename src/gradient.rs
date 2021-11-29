use crate::matrix::*;

#[derive(Copy, Clone)]
pub struct Gradient {
    pub value: f32,
    pub m1: f32,
    pub m2: f32
}

pub struct Gradients {
    pub data: Vec<Gradient>,
    pub rows: usize,
    pub cols: usize
}

const BETA_1: f32 = 0.9;
const BETA_2: f32 = 0.999;
const LEARNING_RATE: f32 = 0.01;

impl Gradient {
    pub fn new() -> Gradient {
        Gradient {
            value: 0.0,
            m1: 0.0,
            m2: 0.0
        }
    }

    pub fn update(&mut self, delta: f32) {
        self.value += delta;
    }

    pub fn calculate(&mut self) -> f32 {
        if self.value == 0.0 { return 0.0; }

        self.m1 = self.m1 * BETA_1 + self.value * (1.0 - BETA_1);
        self.m2 = self.m2 * BETA_2 + (self.value * self.value) * (1.0 - BETA_2);

        return LEARNING_RATE * self.m1 / ((self.m2 as f64).sqrt() + 1e-8) as f32;
    }

    pub fn reset(&mut self) {
        self.value = 0.0;
    }

    pub fn apply(&mut self, elem: &mut f32) {
        *elem -= self.calculate();
        self.reset();
    }
}

impl Gradients {
    pub fn new(rows: usize, cols: usize) -> Gradients {
        Gradients {
            data: vec![Gradient::new(); cols*rows],
            rows: rows,
            cols: cols
        }
    }

    pub fn size(&self) -> usize {
        self.rows * self.cols
    }

    pub fn update(&mut self, row: usize, col: usize, gradient: f32) {
        self.data[col*self.rows + row].update(gradient);
    }

    pub fn apply(&mut self, m: &mut Matrix) {
        for i in 0..m.size() {
            self.data[i].apply(&mut m.data[i]);
        }
    }

    pub fn values(&self) -> Vec<f32> {
        let mut values = vec![0.0; self.size()];
        for i in 0..self.size() {
            values[i] = self.data[i].value;
        }
        return values;
    }

    pub fn copy(&self) -> Gradients {
        let mut g = Gradients::new(self.rows, self.cols);
        for i in 0..self.size() {
            g.data[i] = self.data[i];
        }
        return g;
    }

}
