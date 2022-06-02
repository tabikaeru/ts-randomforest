use randomforest::criterion::Gini;
use randomforest::table::{TableBuilder, TableError};
use randomforest::{RandomForestClassifier, RandomForestClassifierOptions};
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::io::BufWriter;
use std::panic;
extern crate console_error_panic_hook;

pub fn save_model(file_name: &str, classifier: &RandomForestClassifier) -> Result<(), TableError> {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    let mut file = File::create(file_name).unwrap();
    let writer = BufWriter::new(file);
    classifier.serialize(writer);
    return Ok(());
}

pub fn load_model(file_name: &str) -> Result<RandomForestClassifier, TableError> {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    let path = env::current_dir().unwrap();

    let mut file = File::open(file_name).unwrap();
    let reader = BufReader::new(file);

    let classifier_deserialized = RandomForestClassifier::deserialize(reader).unwrap();
    return Ok(classifier_deserialized);
}

pub fn test(features: &[f64], classifier: &RandomForestClassifier) -> Result<f64, TableError> {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    let result = classifier.predict(features);
    return Ok(result);
}

pub fn train(features: &[Vec<f64>], target: &[f64]) -> Result<RandomForestClassifier, TableError> {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    let mut table_builder = TableBuilder::new();
    for (xs, y) in features.iter().zip(target.iter()) {
        table_builder.add_row(xs, *y)?;
    }
    let table = table_builder.build()?;
    let classifier = RandomForestClassifierOptions::new()
        .seed(0)
        .fit(Gini, table);
    return Ok(classifier);
}
