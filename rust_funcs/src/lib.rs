extern crate numpy;
use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyArray1, ToPyArray};
use numpy::PyReadonlyArray2;
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
/// Derivada da sigmoid, escrita como sig' = sig(1-sig).
/// Aceita um vetor de f64's no formato array[sig(x_0, sig(x_1), ...)] 
/// Retorna array[sig'(x_0, sig'(x_1), ...)]
#[pyfunction]
fn sig_pra_deriv<'py>(py: Python<'py>, sigs: PyReadonlyArray1<f64>) -> Py<PyArray1<f64>> {
    sigs.as_array().map(|x| x*(1.0-x)).to_pyarray(py).to_owned()
}
/// Derivada da sigmoid, escrita como sig' = 1/4*tanh'(x/2).
/// Aceita um vetor de f64's no formato array[x_0, x_1, ...)] 
/// Retorna array[sig'(x_0, sig'(x_1), ...)]
#[pyfunction]
fn sig_deriv<'py>(py: Python<'py>, arr: PyReadonlyArray1<f64>) -> Py<PyArray1<f64>> {
    arr.as_array().map(|x| 1.0/(((x/2.0).exp() + (-x/2.0).exp()).powi(2))).to_pyarray(py).to_owned()
}
#[pyfunction]
fn tanh_pra_deriv<'py>(py: Python<'py>, tans: PyReadonlyArray1<f64>) -> Py<PyArray1<f64>> {
    tans.as_array().map(|x| 1.0 - x.powi(2)).to_pyarray(py).to_owned()
}
#[pyfunction]
fn tanh_deriv<'py>(py: Python<'py>, arr: PyReadonlyArray1<f64>) -> Py<PyArray1<f64>> {
    arr.as_array().map(|x| 4.0/((x.exp() + (-x).exp()).powi(2))).to_pyarray(py).to_owned()
}
#[pyfunction]
fn matrix_vec<'py>(py: Python<'py>, mat: PyReadonlyArray2<f64>, vec: PyReadonlyArray1<f64>) -> Py<PyArray1<f64>>{
    let &[_n1, n2] = mat.shape() else { panic!("Não é uma matriz com duas dimensões.") };
    let (mat, vec) = (mat.as_array(), vec.as_array());
    let mut res = Vec::with_capacity(n2);
    for row in mat.rows() {
        res.push(zip(row.iter(), vec.iter()).fold(0.0, |acc: f64, (r, v)| acc + r*v))
    };
    PyArray1::from_vec(py, res).to_owned()
}
    // extern crate openblas_src;
    // extern crate nalgebra;
    // use nalgebra::{Vector1, DVector, DMatrix};
    // use numpy::{PyReadonlyArray2, ToPyArray};
    // let mat = mat.as_matrix().into_owned();
    // let n = vec.len();
    // let vec = DVector::from_row_slice(vec.as_slice().expect("vetor não contíguo"));
    // let mut res = DVector::from_element(n, 0.0);
    // res.gemv(1.0, &mat, &vec, 0.0);
    // let res = res.as_slice();
    // res.to_pyarray(py).to_owned()
#[pyfunction]
fn multiply_vec<'py>(py: Python<'py>,
 arr1: PyReadonlyArray1<f64>, arr2: PyReadonlyArray1<f64>
) -> Py<PyArray1<f64>> {
    PyArray1::from_iter(py, zip(arr1.as_slice().unwrap(), arr2.as_slice().unwrap()).map(
        |(&x1, &y1): (&f64, &f64)| x1*y1
        )
    ).to_owned()
}
#[pyfunction]
fn multiply_vec_par<'py>(py: Python<'py>,
 arr1: PyReadonlyArray1<f64>, arr2: PyReadonlyArray1<f64>
) -> Py<PyArray1<f64>> {
    PyArray1::from_vec(py, arr1.as_slice().unwrap().par_iter().zip_eq(arr2.as_slice().unwrap().par_iter()).map(
        |(&x1, &y1): (&f64, &f64)| x1*y1
        ).collect()
    ).to_owned()
}
/// Calcula entropia cruzada entre duas distribuições.
/// É "o mesmo" que logloss, mas o primeiro vetor é de f64, não i32.
#[pyfunction]
fn crossentropy(y: PyReadonlyArray1<f64>, yhat: PyReadonlyArray1<f64>) -> f64 {
    let n = y.len() as f64;
    -zip(y.as_array(), yhat.as_array()).fold(
        0.0,
        |acc, (x, xhat)| acc + x * xhat.ln() + (1.0 - x) * (1.0 - xhat).ln()
    ) / n
}
/// Calcula o logloss de um vetor y de classes 0 ou 1 (int32) e um vetor yhat de probabilidades da classe 1.0.
#[pyfunction]
fn logloss(y: PyReadonlyArray1<i32>, yhat: PyReadonlyArray1<f64>) -> f64 {
    let n = y.len() as f64;
    -zip(y.as_array(), yhat.as_array()).fold(0.0, |acc, (&x, xhat)| (
        if x == 1 {
            acc + xhat.ln()
        } else {
            acc + (1.0 - xhat).ln()
        }
    )) / n
}
/// Mesmo que logloss, mas faz pararelização utilizando a crate rayon.
/// Em 4 núcleros foi mais rápido a partir de n ~ 700.
#[pyfunction]
fn logloss_par(y: PyReadonlyArray1<i32>, yhat: PyReadonlyArray1<f64>) -> f64 {
    let n = y.len() as f64;
    -y
        .as_slice()
        .unwrap()
        .par_iter()
        .zip_eq(yhat.as_slice().unwrap().par_iter())
        .fold(
            || 0.0,
            |acc: f64, (&x, &xhat): (&i32, &f64)| (
                if x == 1 {
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
    let n2: f64 = (2 * y.len()) as f64;
    zip(x.as_array(), y.as_array()).fold(0.0, |acc, (x1, y1)| acc + (x1 - y1) * (x1 - y1)) / n2
}
/// A Python module implemented in Rust. Version is 0.1.2.
#[pymodule]
fn rust_funcs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sig_pra_deriv, m)?)?;
    m.add_function(wrap_pyfunction!(sig_deriv, m)?)?;
    m.add_function(wrap_pyfunction!(tanh_pra_deriv, m)?)?;
    m.add_function(wrap_pyfunction!(tanh_deriv, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_vec, m)?)?;
    m.add_function(wrap_pyfunction!(multiply_vec, m)?)?;
    m.add_function(wrap_pyfunction!(multiply_vec_par, m)?)?;
    m.add_function(wrap_pyfunction!(logloss, m)?)?;
    m.add_function(wrap_pyfunction!(logloss_par, m)?)?;
    m.add_function(wrap_pyfunction!(crossentropy, m)?)?;
    m.add_function(wrap_pyfunction!(quaderror, m)?)?;
    Ok(())
}
