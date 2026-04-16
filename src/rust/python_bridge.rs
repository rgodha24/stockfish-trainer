use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Mutex;

use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::feature_extraction::FeatureSet;
use crate::pipeline::{
    default_batch_slab_count, BatchPipeline, PipelineConfig, PipelineError, PooledBatch,
    SkipConfig, ThreadOverride,
};

#[pyclass(name = "BatchStream")]
pub struct PyBatchStream {
    pipeline: Mutex<Option<BatchPipeline>>,
}

#[pyclass]
struct PyBatchOwner {
    batch: PooledBatch,
}

#[pymethods]
impl PyBatchStream {
    #[new]
    #[pyo3(signature = (
        feature_set,
        filenames,
        batch_size,
        total_threads=None,
        decode_threads=None,
        encode_threads=None,
        slab_count=None,
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
        feature_set: &str,
        filenames: Vec<String>,
        batch_size: usize,
        total_threads: Option<usize>,
        decode_threads: Option<usize>,
        encode_threads: Option<usize>,
        slab_count: Option<usize>,
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
        let feature_set = FeatureSet::from_str(feature_set)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let mut config = PipelineConfig::new(
            filenames.into_iter().map(PathBuf::from).collect(),
            feature_set,
            batch_size,
        );
        if let Some(total_threads) = total_threads {
            config.total_threads = total_threads;
        }
        if decode_threads.is_some() || encode_threads.is_some() {
            config.thread_override = Some(ThreadOverride {
                decode: decode_threads,
                encode: encode_threads,
            });
        }
        config.slab_count =
            slab_count.unwrap_or_else(|| default_batch_slab_count(config.total_threads));
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

        let pipeline = BatchPipeline::new(config).map_err(to_py_runtime_error)?;
        Ok(Self {
            pipeline: Mutex::new(Some(pipeline)),
        })
    }

    fn next_batch<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let maybe_batch = py.allow_threads(|| {
            let guard = self.pipeline.lock().unwrap();
            match guard.as_ref() {
                Some(pipeline) => pipeline.next_batch(),
                None => Ok(None),
            }
        });

        match maybe_batch {
            Ok(Some(batch)) => Ok(Some(batch_to_pydict(py, batch)?)),
            Ok(None) => Ok(None),
            Err(error) => Err(to_py_runtime_error(error)),
        }
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = {
            let guard = self.pipeline.lock().unwrap();
            match guard.as_ref() {
                Some(pipeline) => pipeline.stats(),
                None => {
                    return Err(PyRuntimeError::new_err(
                        "batch stream has already been closed",
                    ))
                }
            }
        };

        let dict = PyDict::new(py);
        dict.set_item("decoded_entries", stats.decoded_entries)?;
        dict.set_item("encoded_entries", stats.encoded_entries)?;
        dict.set_item("skipped_entries", stats.skipped_entries)?;
        dict.set_item("produced_batches", stats.produced_batches)?;
        dict.set_item("scanned_chunks", stats.scanned_chunks)?;
        dict.set_item("decoded_queue_len", stats.decoded_queue_len)?;
        dict.set_item("ready_queue_len", stats.ready_queue_len)?;
        dict.set_item("free_queue_len", stats.free_queue_len)?;
        Ok(dict)
    }

    fn close(&self) {
        let mut guard = self.pipeline.lock().unwrap();
        let _ = guard.take();
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        match self.next_batch(py)? {
            Some(batch) => Ok(batch),
            None => Err(PyStopIteration::new_err("batch stream is exhausted")),
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBatchStream>()?;
    Ok(())
}

fn batch_to_pydict<'py>(py: Python<'py>, batch: PooledBatch) -> PyResult<Bound<'py, PyDict>> {
    let owner = Py::new(py, PyBatchOwner { batch })?;
    let owner_bound = owner.bind(py);
    let owner_ref = owner_bound.borrow();
    let batch = owner_ref.batch.slab();
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
