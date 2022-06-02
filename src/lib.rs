extern crate wasm_bindgen;
use wasm_bindgen::prelude::*;
mod random_forest;
mod util;

#[wasm_bindgen]
pub fn regressorTrain(
    flatten_features: Vec<f64>,
    flatten_features_length: usize,
    target: Vec<f64>,
    file_name: &str,
) {
    let multiplexed_features =
        util::multiplexing_vector(&flatten_features, flatten_features_length);
    let regressor = random_forest::regressor::train(&multiplexed_features, &target).unwrap();
    random_forest::regressor::save_model(&file_name, &regressor).unwrap();
    return;
}

#[wasm_bindgen]
pub fn regressorTest(features: Vec<f64>, file_name: &str) -> f64 {
    let regressor = random_forest::regressor::load_model(file_name).unwrap();
    let result = random_forest::regressor::test(&features, &regressor).unwrap();
    return result;
}

#[wasm_bindgen]
pub fn classifierTrain(
    flatten_features: Vec<f64>,
    flatten_features_length: usize,
    target: Vec<f64>,
    file_name: &str,
) {
    let multiplexed_features =
        util::multiplexing_vector(&flatten_features, flatten_features_length);
    let classifier = random_forest::classifier::train(&multiplexed_features, &target).unwrap();
    random_forest::classifier::save_model(&file_name, &classifier).unwrap();
    return;
}

#[wasm_bindgen]
pub fn classifier_test(features: Vec<f64>, file_name: &str) -> f64 {
    let classifier = random_forest::classifier::load_model(file_name).unwrap();
    let result = random_forest::classifier::test(&features, &classifier).unwrap();
    return result;
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn regressor_test() {
        let file_name = "regressor_test.model";

        let features = vec![
            vec![0.0, 2.0, 1.0, 0.0],
            vec![0.0, 2.0, 1.0, 1.0],
            vec![1.0, 2.0, 1.0, 0.0],
            vec![2.0, 1.0, 1.0, 0.0],
            vec![2.0, 0.0, 0.0, 0.0],
            vec![2.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0],
            vec![2.0, 1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];
        let target = vec![
            25.0, 30.0, 46.0, 45.0, 52.0, 23.0, 43.0, 35.0, 38.0, 46.0, 48.0, 52.0,
        ];
        let regressor = random_forest::regressor::train(&features, &target).unwrap();
        let test_features = &[1.0, 2.0, 0.0, 0.0];
        random_forest::regressor::save_model(file_name, &regressor);
        // let new_regressor = random_forest::regressor::load_model(file_name).unwrap();
        // let result = random_forest::regressor::test(test_features, &new_regressor);

        // assert_eq!(result.unwrap(), 41.9785);
    }

    fn test_classifier() {
        // let file_name = "classifier_test.model";

        // let features = vec![
        //     vec![0.0, 2.0, 1.0, 0.0],
        //     vec![0.0, 2.0, 1.0, 1.0],
        //     vec![1.0, 2.0, 1.0, 0.0],
        //     vec![2.0, 1.0, 1.0, 0.0],
        //     vec![2.0, 0.0, 0.0, 0.0],
        //     vec![2.0, 0.0, 0.0, 1.0],
        //     vec![1.0, 0.0, 0.0, 1.0],
        // ];

        // let target = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        // let classifier = random_forest::classifier::train(&features, &target).unwrap();
        // let test_features = &[1.0, 2.0, 1.0, 0.0];

        // random_forest::classifier::save_model(file_name, &classifier);
        // let new_classifier = random_forest::classifier::load_model(file_name).unwrap();

        // let result = random_forest::classifier::test(test_features, &new_classifier);
        // assert_eq!(result.unwrap(), 1.0);
    }
}
