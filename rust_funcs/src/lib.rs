use pyo3::{ prelude::*, types::PyLong };
use numpy::PyReadonlyArray1;
use std::iter::zip;
extern crate rayon;
use rayon::prelude::*;

// use ndarray;
// use ndarray::{ ArrayBase, Dim, OwnedRepr };
// const LOG10_TO_LN: f64 = 1.0/std::f64::consts::LOG10_E;

/*
fn log(arr: &PyReadonlyArray1<f64>) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> {
    arr.as_array().map(|x| x.ln())
}
fn log1(arr: &PyReadonlyArray1<f64>) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> {
    arr.as_array().map(|x| (1.0 - x).ln())
}
*/
/// Retorna o crossentropy de dois vetores de float64. O log é calculado no segundo vetor.
#[pyfunction]
fn crossentropy(y: PyReadonlyArray1<f64>, yhat: PyReadonlyArray1<f64>) -> f64 {
    let n = y.len() as f64;
    -zip(y.as_array(), yhat.as_array()).fold(
        0.0,
        |acc, (x, xhat)| acc + x * xhat.ln() + (1.0 - x) * (1.0 - xhat).ln()
    ) / n
}
#[pyfunction]
fn crossentropy_teste(y: PyReadonlyArray1<f64>, yhat: PyReadonlyArray1<f64>) -> f64 {
    let n = y.len() as f64;
    zip(y.as_array(), yhat.as_array()).fold(
        0.0,
        |acc, (x, xhat)| acc - x * xhat.ln() - (1.0 - x) * (1.0 - xhat).ln()
    ) / n
}
/// Calcula o logloss de um vetor y de classes 0.0 ou 1.0 e um vetor yhat de probabilidades da classe 1.0.
#[pyfunction]
fn logloss(y: PyReadonlyArray1<f64>, yhat: PyReadonlyArray1<f64>) -> f64 {
    let n = y.len() as f64;
    -zip(y.as_array(), yhat.as_array()).fold(0.0, |acc, (&x, xhat)| (
        if x == 1.0 {
            acc + xhat.ln()
        } else {
            acc + (1.0 - xhat).ln()
        }
    )) / n
}
#[pyfunction]
fn logloss_par(y: PyReadonlyArray1<f64>, yhat: PyReadonlyArray1<f64>) -> f64 {
    let n = y.len() as f64;
    -y
        .as_slice()
        .unwrap()
        .par_iter()
        .zip_eq(yhat.as_slice().unwrap().par_iter())
        .fold(
            || 0.0,
            |acc: f64, (&x, &xhat): (&f64, &f64)| (
                if x == 1.0_f64 {
                    acc + xhat.ln()
                } else {
                    acc + (1.0 - xhat).ln()
                }
            )
        )
        .sum::<f64>() / n
}
/*
use std::cmp::PartialEq;
use num::Num;
#[pyfunction]
fn logloss_naonecessariamentefloat<N: Num + numpy::Element + PartialEq>(y: PyReadonlyArray1<N>, yhat: PyReadonlyArray1<f64>) -> f64 {
    let n = yhat.len() as f64;
    -zip(y.as_array(), yhat.as_array()).fold(0.0, |acc, (&x, xhat)| (
        if x == N::one() {
            acc + xhat.ln()
        } else {
            acc + (1.0 - xhat).ln()
        }
    )) / n
}
*/

/*
#[pyfunction]
fn logloss_teste(y: PyReadonlyArray1<f64>, yhat: PyReadonlyArray1<f64>) -> f64 {
    let n = LOG10_TO_LN/y.len() as f64;
    -zip(y.as_array(), yhat.as_array()).fold(
        0.0,
        |acc, (x, xhat)| acc + x * xhat.log10() + (1.0 - x) * (1.0 - xhat).log10()
    )*n
    // -(log(&yhat).dot(&y.as_array()) + zip(log1(&yhat), y.as_array()).fold(0.0, |acc, (xhat, x)| acc + xhat*(1.0-x)))/n
}
*/

#[pyfunction]
fn quaderror(x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> f64 {
    let n: f64 = (2 * y.len()) as f64;
    zip(x.as_array(), y.as_array()).fold(0.0, |acc, (x1, y1)| acc + (x1 - y1) * (x1 - y1)) / n
}

fn fibonacci(n: usize, vec: &mut Vec<u128>) -> u128 {
    if vec.len() > n - 1 {
        return vec[n - 1];
    }
    let number = if n == 1 || n == 2 { 1 } else { fibonacci(n - 1, vec) + fibonacci(n - 2, vec) };

    if vec.len() == n - 1 {
        vec.push(number);
    }

    number
}
#[pyfunction]
fn calc_fibonacci() {
    use std::time::SystemTime;

    let now = SystemTime::now();

    let mut vec = Vec::new();

    let array: [usize; 186] = core::array::from_fn(|i| i + 1);

    for number in array {
        println!("The {}th fibonacci number is {}", number, fibonacci(number, &mut vec));
    }
    println!("It took {} miliseconds", now.elapsed().expect("aaaa").as_millis());
}
/// A Python module implemented in Rust. Version is 0.1.2.
#[pymodule]
fn rust_funcs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(logloss, m)?)?;
    m.add_function(wrap_pyfunction!(logloss_par, m)?)?;
    m.add_function(wrap_pyfunction!(crossentropy, m)?)?;
    m.add_function(wrap_pyfunction!(crossentropy_teste, m)?)?;
    m.add_function(wrap_pyfunction!(quaderror, m)?)?;
    m.add_function(wrap_pyfunction!(calc_fibonacci, m)?)?;
    Ok(())
}
