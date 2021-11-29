const COST_EVAL_WEIGHT: f32 = 0.5;
const COST_RESULT_WEIGHT: f32 = 0.5;

pub fn cost_gradient(output: f32, eval_target: f32, result_target: f32) -> f32 {
    2.0 * (COST_EVAL_WEIGHT * (output - eval_target) + COST_RESULT_WEIGHT * (output - result_target))
}

pub fn validation_cost(output: f32, eval_target: f32, result_target: f32) -> f32 {
    COST_EVAL_WEIGHT * ((output - eval_target) as f64).powi(2) as f32 + COST_RESULT_WEIGHT * ((output - result_target) as f64).powi(2) as f32
}
