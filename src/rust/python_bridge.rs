use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Mutex;

use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::feature_extraction::FeatureSet;
use crate::pipeline::{
    encode_packed_chunks, EncodedBatch, PackedEntryStream, PackedStreamConfig, PipelineError,
    SkipConfig,
};

#[pyclass(name = "PackedEntryStream")]
pub struct PyPackedEntryStream {
    stream: Mutex<Option<PackedEntryStream>>,
}

#[pyclass]
struct PyEncodedBatchOwner {
    batch: EncodedBatch,
}

#[pymethods]
impl PyPackedEntryStream {
    #[new]
    #[pyo3(signature = (
        filenames,
        total_threads=None,
        decode_threads=None,
        chunk_entries=8192,
        shuffle_buffer_entries=16384,
        seed=None,
        cyclic=false,
        filtered=false,
        random_fen_skipping=0,
        wld_filtered=false,
        early_fen_skipping=-1,
        simple_eval_skipping=0,
        param_index=0,
        pc_y1=1.0,
        pc_y2=2.0,
        pc_y3=1.0,
        rank=0,
        world_size=1
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        filenames: Vec<String>,
        total_threads: Option<usize>,
        decode_threads: Option<usize>,
        chunk_entries: usize,
        shuffle_buffer_entries: usize,
        seed: Option<u64>,
        cyclic: bool,
        filtered: bool,
        random_fen_skipping: u32,
        wld_filtered: bool,
        early_fen_skipping: i32,
        simple_eval_skipping: i32,
        param_index: i32,
        pc_y1: f64,
        pc_y2: f64,
        pc_y3: f64,
        rank: usize,
        world_size: usize,
    ) -> PyResult<Self> {
        let mut config =
            PackedStreamConfig::new(filenames.into_iter().map(PathBuf::from).collect());
        if let Some(total_threads) = total_threads {
            config.total_threads = total_threads;
        }
        config.decode_threads = decode_threads;
        config.chunk_entries = chunk_entries;
        config.shuffle_buffer_entries = shuffle_buffer_entries;
        config.seed = seed;
        config.cyclic = cyclic;
        config.rank = rank;
        config.world_size = world_size;
        config.skip_config = SkipConfig {
            filtered,
            random_fen_skipping,
            wld_filtered,
            early_fen_skipping,
            simple_eval_skipping,
            param_index,
            pc_y1,
            pc_y2,
            pc_y3,
        };

        let stream = PackedEntryStream::new(config).map_err(to_py_runtime_error)?;
        Ok(Self {
            stream: Mutex::new(Some(stream)),
        })
    }

    fn next_chunk<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let maybe_chunk = py.allow_threads(|| {
            let guard = self.stream.lock().unwrap();
            match guard.as_ref() {
                Some(stream) => stream.next_chunk(),
                None => Ok(None),
            }
        });

        match maybe_chunk {
            Ok(Some(chunk)) => Ok(Some(PyBytes::new(py, &chunk))),
            Ok(None) => Ok(None),
            Err(error) => Err(to_py_runtime_error(error)),
        }
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = {
            let guard = self.stream.lock().unwrap();
            match guard.as_ref() {
                Some(stream) => stream.stats(),
                None => {
                    return Err(PyRuntimeError::new_err(
                        "packed entry stream already closed",
                    ))
                }
            }
        };

        let dict = PyDict::new(py);
        dict.set_item("decoded_entries", stats.decoded_entries)?;
        dict.set_item("skipped_entries", stats.skipped_entries)?;
        dict.set_item("produced_chunks", stats.produced_chunks)?;
        dict.set_item("scanned_chunks", stats.scanned_chunks)?;
        dict.set_item("chunk_queue_len", stats.chunk_queue_len)?;
        Ok(dict)
    }

    fn close(&self) {
        let mut guard = self.stream.lock().unwrap();
        let _ = guard.take();
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        match self.next_chunk(py)? {
            Some(chunk) => Ok(chunk),
            None => Err(PyStopIteration::new_err("packed entry stream exhausted")),
        }
    }
}

#[pyfunction(name = "encode_packed_chunks")]
fn encode_packed_chunks_py<'py>(
    py: Python<'py>,
    feature_set: &str,
    chunks: Vec<Vec<u8>>,
    batch_size: usize,
    encode_threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let feature_set = FeatureSet::from_str(feature_set)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let batch = encode_packed_chunks(&feature_set, &chunks, batch_size, encode_threads)
        .map_err(to_py_runtime_error)?;
    batch_to_pydict(py, batch)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPackedEntryStream>()?;
    m.add_function(wrap_pyfunction!(encode_packed_chunks_py, m)?)?;
    Ok(())
}

fn batch_to_pydict<'py>(py: Python<'py>, batch: EncodedBatch) -> PyResult<Bound<'py, PyDict>> {
    let owner = Py::new(py, PyEncodedBatchOwner { batch })?;
    let owner_bound = owner.bind(py);
    let owner_ref = owner_bound.borrow();
    let batch = &owner_ref.batch;
    let rows = batch.len();
    let cols = batch.max_active_features();

    let is_white_view = ArrayView2::from_shape((rows, 1), batch.is_white_slice())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let outcome_view = ArrayView2::from_shape((rows, 1), batch.outcome_slice())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let score_view = ArrayView2::from_shape((rows, 1), batch.score_slice())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let white_view = ArrayView2::from_shape((rows, cols), batch.white_flat_slice())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let black_view = ArrayView2::from_shape((rows, cols), batch.black_flat_slice())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let psqt_indices_view = ArrayView1::from(batch.psqt_indices_slice());
    let layer_stack_indices_view = ArrayView1::from(batch.layer_stack_indices_slice());

    let is_white =
        unsafe { PyArray2::borrow_from_array(&is_white_view, owner_bound.clone().into_any()) };
    let outcome =
        unsafe { PyArray2::borrow_from_array(&outcome_view, owner_bound.clone().into_any()) };
    let score = unsafe { PyArray2::borrow_from_array(&score_view, owner_bound.clone().into_any()) };
    let white = unsafe { PyArray2::borrow_from_array(&white_view, owner_bound.clone().into_any()) };
    let black = unsafe { PyArray2::borrow_from_array(&black_view, owner_bound.clone().into_any()) };
    let psqt_indices =
        unsafe { PyArray1::borrow_from_array(&psqt_indices_view, owner_bound.clone().into_any()) };
    let layer_stack_indices = unsafe {
        PyArray1::borrow_from_array(&layer_stack_indices_view, owner_bound.clone().into_any())
    };

    let dict = PyDict::new(py);
    dict.set_item("num_inputs", batch.num_inputs())?;
    dict.set_item("size", rows)?;
    dict.set_item(
        "num_active_white_features",
        batch.num_active_white_features(),
    )?;
    dict.set_item(
        "num_active_black_features",
        batch.num_active_black_features(),
    )?;
    dict.set_item("max_active_features", cols)?;
    dict.set_item("is_white", is_white)?;
    dict.set_item("outcome", outcome)?;
    dict.set_item("score", score)?;
    dict.set_item("white", white)?;
    dict.set_item("black", black)?;
    dict.set_item("psqt_indices", psqt_indices)?;
    dict.set_item("layer_stack_indices", layer_stack_indices)?;
    Ok(dict)
}

fn to_py_runtime_error(error: PipelineError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}
