use randomforest::criterion::Mse;
use randomforest::table::{TableBuilder, TableError};
use randomforest::{RandomForestRegressor, RandomForestRegressorOptions};
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::panic;
extern crate console_error_panic_hook;

pub fn save_model(file_name: &str, regressor: &RandomForestRegressor) -> Result<(), TableError> {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    let mut file = File::create(file_name).unwrap();
    let writer = BufWriter::new(file);
    regressor.serialize(writer);
    return Ok(());
}

pub fn load_model(file_name: &str) -> Result<RandomForestRegressor, TableError> {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    let path = env::current_dir().unwrap();

    let mut file = File::open(file_name).unwrap();
    let reader = BufReader::new(file);

    let regressor_deserialized = RandomForestRegressor::deserialize(reader).unwrap();
    return Ok(regressor_deserialized);
}

pub fn test(features: &[f64], regressor: &RandomForestRegressor) -> Result<f64, TableError> {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    let result = regressor.predict(features);
    return Ok(result);
}

pub fn train(features: &[Vec<f64>], target: &[f64]) -> Result<RandomForestRegressor, TableError> {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    let mut table_builder = TableBuilder::new();
    for (xs, y) in features.iter().zip(target.iter()) {
        table_builder.add_row(xs, *y)?;
    }
    let table = table_builder.build()?;
    let regressor = RandomForestRegressorOptions::new().seed(0).fit(Mse, table);
    return Ok(regressor);
}
