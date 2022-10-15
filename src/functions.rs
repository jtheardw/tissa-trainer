// const SIGMOID_SCALE: f64 = 3.5 / 1024.0;
const SIGMOID_SCALE: f64 = 2.6 / 512.0;

pub fn sigmoid(x: f32) -> f32 {
    // Classic s-curve function
    (1.0 / (1.0 + ((-x as f64) * SIGMOID_SCALE).exp())) as f32
}

pub fn sigmoid_prime(x: f32) -> f32 {
    // first derivative of sigmoid
    // x * (1.0 - x) * (SIGMOID_SCALE as f32)
    sigmoid(x) * (1.0 - sigmoid(x)) * SIGMOID_SCALE as f32
}

pub fn sigmoid_to_sigmoid_prime(x: f32) -> f32 {
    x * (1.0 - x) * (SIGMOID_SCALE as f32)
}

pub fn relu(x: f32) -> f32 {
    // Rectified Lineator Unit (ReLU)
    // essentially, return x if x above 0.
    // 0 otherwise
    x.max(0.0)
}

pub fn relu_prime(x: f32) -> f32 {
    // first derivative of relu
    if x > 0.0 { 1.0 } else { 0.0 }
}
