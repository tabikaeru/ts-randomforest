pub fn multiplexing_vector(vector: &Vec<f64>, vector_length: usize) -> Vec<Vec<f64>> {
    let mut result = Vec::with_capacity(vector.len() / vector_length);
    for index in 0..vector.len() {
        if index % vector_length == 0 {
            let mut sub_vector = Vec::with_capacity(vector_length);
            for sub_index in index..index + vector_length {
                sub_vector.push(vector[sub_index]);
            }
            result.push(sub_vector);
        }
    }
    return result;
}
