use pyo3::prelude::*;

pub mod feature_extraction;
pub mod pipeline;
mod python_bridge;

pub use sfbinpack::chess;

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python_bridge::register(m)?;
    Ok(())
}
