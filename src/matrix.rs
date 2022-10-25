use rand::distributions::{Distribution, Uniform};

#[derive(Clone)]
pub struct Matrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Matrix {
        Matrix {
            data: data,
            rows: rows,
            cols: cols
        }
    }

    pub fn new_random(rows: usize, cols: usize, range_factor: f32) -> Matrix {
        let len = rows * cols;
        let data = random_array(len, range_factor);
        Matrix::new(rows, cols, data)
    }

    pub fn new_random_column(elements: usize, range_factor: f32) -> Matrix {
        Matrix::new_random(elements, 1, range_factor)
    }

    pub fn size(&self) -> usize {
        self.rows * self.cols
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[col*self.rows + row]
    }

    pub fn clear(&mut self) {
        for i in 0..self.size() {
            self.data[i] = 0.0;
        }
    }

    pub fn copy(&self) -> Matrix {
        let cp = Matrix {
            data: self.data.to_vec(),
            cols: self.cols,
            rows: self.rows
        };
        return cp;
    }
}

fn random_array(len: usize, range_factor: f32) -> Vec<f32> {
    let max = 2.0 / ((range_factor / 2.0) as f64).sqrt();
    let mut rng = rand::thread_rng();

    let mut data: Vec<f32> = vec![0.0; len];
    // let mut dist = Uniform::from(-max..max);
    let mut dist = Uniform::from((max/5.0)..max);
    for i in 0..len {
        let sign = if (Uniform::from(0.0..1.0).sample(&mut rng) as f32 >= 0.5) {1.0} else {-1.0};
        data[i] = (dist.sample(&mut rng) as f32) * sign;
    }

    return data;
}
