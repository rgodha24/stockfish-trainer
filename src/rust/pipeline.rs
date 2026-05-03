use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender, TrySendError};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use sfbinpack::chess::{color::Color, piece::Piece, piecetype::PieceType};
use sfbinpack::{CompressedReaderError, CompressedTrainingDataEntryReader, TrainingDataEntry};

use crate::feature_extraction::{
    encode_training_entry_indices_only, FeatureSet, RowMetadata, SparseRow,
};

const VALUE_NONE: i16 = 32002;
const MAX_SKIP_RATE: f64 = 10.0;
const BINPACK_HEADER_SIZE: usize = 8;
const BINPACK_MAX_CHUNK_SIZE: u32 = 100 * 1024 * 1024;
const DECODER_COUNTER_FLUSH_INTERVAL: u64 = 4096;
const ENCODER_COUNTER_FLUSH_INTERVAL: u64 = 4096;
const POLL_INTERVAL: Duration = Duration::from_millis(50);
const DEFAULT_ACCUMULATION_ENTRIES: usize = 1024;
const WLD_MAX_PLY: usize = 240;
const WLD_SCORE_SCALE: f32 = 100.0 / 208.0;

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub files: Vec<PathBuf>,
    pub feature_set: FeatureSet,
    pub batch_size: usize,
    pub total_threads: usize,
    pub thread_override: Option<ThreadOverride>,
    pub slab_count: usize,
    pub shuffle_buffer_entries: usize,
    pub accumulation_entries: usize,
    pub cyclic: bool,
    pub skip_config: SkipConfig,
    pub seed: Option<u64>,
    pub rank: usize,
    pub world_size: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ThreadOverride {
    pub decode: Option<usize>,
    pub encode: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkipConfig {
    pub filtered: bool,
    pub random_fen_skipping: u32,
    pub wld_filtered: bool,
    pub early_fen_skipping: i32,
    pub simple_eval_skipping: i32,
    pub param_index: i32,
    pub pc_y1: f64,
    pub pc_y2: f64,
    pub pc_y3: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineStats {
    pub decoded_entries: u64,
    pub encoded_entries: u64,
    pub skipped_entries: u64,
    pub produced_batches: u64,
    pub scanned_chunks: u64,
    pub decoded_queue_len: usize,
    pub ready_queue_len: usize,
    pub free_queue_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineError {
    message: String,
}

pub struct BatchPipeline {
    ready_rx: Option<Receiver<HostBatchSlab>>,
    free_tx: Option<Sender<HostBatchSlab>>,
    error_rx: Option<Receiver<String>>,
    error: Mutex<Option<PipelineError>>,
    stats: Arc<PipelineCounters>,
    workers: Vec<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
}

pub struct PooledBatch {
    slab: Option<HostBatchSlab>,
    free_tx: Sender<HostBatchSlab>,
}

#[derive(Debug)]
pub struct HostBatchSlab {
    num_inputs: usize,
    max_active_features: usize,
    capacity: usize,
    size: usize,
    num_active_white_features: usize,
    num_active_black_features: usize,
    white_counts: Vec<u16>,
    black_counts: Vec<u16>,
    is_white: Vec<f32>,
    outcome: Vec<f32>,
    score: Vec<f32>,
    white: Vec<i32>,
    black: Vec<i32>,
    psqt_indices: Vec<i64>,
    layer_stack_indices: Vec<i64>,
}

#[derive(Default)]
struct PipelineCounters {
    decoded_entries: AtomicU64,
    encoded_entries: AtomicU64,
    skipped_entries: AtomicU64,
    produced_batches: AtomicU64,
    scanned_chunks: AtomicU64,
}

#[derive(Clone, Copy, Debug)]
struct ChunkTask {
    file_index: usize,
    offset: u64,
    size: u32,
}

struct ChunkScheduler {
    tasks: Arc<[ChunkTask]>,
    next_index: AtomicUsize,
    cyclic: bool,
}

#[derive(Default)]
struct DecoderWorkerCounters {
    decoded_entries: u64,
    skipped_entries: u64,
}

#[derive(Default)]
struct EncoderWorkerCounters {
    encoded_entries: u64,
}

#[derive(Default)]
struct WldParams {
    a: f32,
    inv_b: f32,
}

struct SkipDecider {
    enabled: bool,
    config: SkipConfig,
    random_skip_probability: f64,
    desired_piece_count_weights_total: f64,
    alpha: f64,
    piece_count_history_all: [f64; 33],
    piece_count_history_passed: [f64; 33],
    piece_count_history_all_total: f64,
    piece_count_history_passed_total: f64,
}

impl PipelineConfig {
    pub fn new(files: Vec<PathBuf>, feature_set: FeatureSet, batch_size: usize) -> Self {
        let available_threads = thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(2);
        let total_threads = available_threads.saturating_sub(1).max(1);

        Self {
            files,
            feature_set,
            batch_size,
            total_threads,
            thread_override: None,
            slab_count: default_batch_slab_count(total_threads),
            shuffle_buffer_entries: 65_536,
            accumulation_entries: DEFAULT_ACCUMULATION_ENTRIES,
            cyclic: false,
            skip_config: SkipConfig::default(),
            seed: None,
            rank: 0,
            world_size: 1,
        }
    }

    fn validate(&self) -> Result<(), PipelineError> {
        if self.files.is_empty() {
            return Err(PipelineError::new(
                "pipeline requires at least one input file",
            ));
        }
        if self.batch_size == 0 {
            return Err(PipelineError::new("batch_size must be greater than zero"));
        }
        if self.total_threads == 0 {
            return Err(PipelineError::new(
                "total_threads must be greater than zero",
            ));
        }
        if self.slab_count == 0 {
            return Err(PipelineError::new("slab_count must be greater than zero"));
        }
        if self.accumulation_entries == 0 {
            return Err(PipelineError::new(
                "accumulation_entries must be greater than zero",
            ));
        }
        if self.world_size == 0 {
            return Err(PipelineError::new("world_size must be greater than zero"));
        }
        if self.rank >= self.world_size {
            return Err(PipelineError::new("rank must be less than world_size"));
        }
        Ok(())
    }

    pub fn thread_counts(&self, task_count: usize) -> (usize, usize) {
        // Default split: ~5/8 decode, ~3/8 encode (empirically tuned; encode is
        // much heavier per-thread than decode for this feature set).
        let total = self.total_threads;
        let mut decode = ((total * 5) / 8).max(1);
        let mut encode = total.saturating_sub(decode).max(1);

        if let Some(ovr) = self.thread_override {
            if let Some(d) = ovr.decode {
                decode = d.max(1);
            }
            if let Some(e) = ovr.encode {
                encode = e.max(1);
            }
        }
        let available_tasks = task_count.max(1);
        decode = decode.min(available_tasks).max(1);
        encode = encode.max(1);
        (decode, encode)
    }
}

impl Default for SkipConfig {
    fn default() -> Self {
        Self {
            filtered: false,
            random_fen_skipping: 0,
            wld_filtered: false,
            early_fen_skipping: -1,
            simple_eval_skipping: 0,
            param_index: 0,
            pc_y1: 1.0,
            pc_y2: 2.0,
            pc_y3: 1.0,
        }
    }
}

impl BatchPipeline {
    pub fn new(config: PipelineConfig) -> Result<Self, PipelineError> {
        config.validate()?;

        let mut scanned_chunks = scan_chunk_tasks(&config.files, config.rank, config.world_size)?;
        if !scanned_chunks.is_empty() {
            let shuffle_seed = config.seed.unwrap_or_else(random_seed);
            let mut rng = SmallRng::seed_from_u64(shuffle_seed);
            scanned_chunks.shuffle(&mut rng);
        }
        let scanned_chunk_count = scanned_chunks.len() as u64;
        let (decode_threads, encode_threads) = config.thread_counts(scanned_chunks.len());

        let decoded_capacity = (decode_threads + encode_threads).saturating_mul(4).max(16);
        let (decoded_tx, decoded_rx) = bounded::<Vec<TrainingDataEntry>>(decoded_capacity);
        let (ready_tx, ready_rx) = bounded::<HostBatchSlab>(config.slab_count);
        let (free_tx, free_rx) = bounded::<HostBatchSlab>(config.slab_count);
        let (error_tx, error_rx) = bounded::<String>(1);
        let stats = Arc::new(PipelineCounters::default());
        stats
            .scanned_chunks
            .store(scanned_chunk_count, Ordering::Release);

        for _ in 0..config.slab_count {
            free_tx
                .send(HostBatchSlab::new(config.batch_size, &config.feature_set))
                .map_err(|_| PipelineError::new("failed to seed free slab queue"))?;
        }

        let files = Arc::new(config.files.clone());
        let chunk_scheduler = Arc::new(ChunkScheduler::new(scanned_chunks, config.cyclic));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut workers = Vec::with_capacity(decode_threads + encode_threads);

        for thread_index in 0..decode_threads {
            let files = Arc::clone(&files);
            let chunk_scheduler = Arc::clone(&chunk_scheduler);
            let decoded_tx = decoded_tx.clone();
            let error_tx = error_tx.clone();
            let thread_config = config.clone();
            let stats = Arc::clone(&stats);
            let shutdown = Arc::clone(&shutdown);
            workers.push(thread::spawn(move || {
                decoder_worker(
                    thread_index,
                    files,
                    chunk_scheduler,
                    thread_config,
                    decoded_tx,
                    error_tx,
                    stats,
                    shutdown,
                )
            }));
        }
        drop(decoded_tx);

        for thread_index in 0..encode_threads {
            let feature_set = config.feature_set.clone();
            let decoded_rx = decoded_rx.clone();
            let ready_tx = ready_tx.clone();
            let free_rx = free_rx.clone();
            let stats = Arc::clone(&stats);
            let shutdown = Arc::clone(&shutdown);
            workers.push(thread::spawn(move || {
                encoder_worker(
                    thread_index,
                    feature_set,
                    decoded_rx,
                    ready_tx,
                    free_rx,
                    stats,
                    shutdown,
                )
            }));
        }
        drop(ready_tx);

        Ok(Self {
            ready_rx: Some(ready_rx),
            free_tx: Some(free_tx),
            error_rx: Some(error_rx),
            error: Mutex::new(None),
            stats,
            workers,
            shutdown,
        })
    }

    pub fn next_batch(&self) -> Result<Option<PooledBatch>, PipelineError> {
        self.take_error_if_any()?;

        let ready_rx = self
            .ready_rx
            .as_ref()
            .expect("ready receiver present while pipeline is alive");
        loop {
            match ready_rx.recv_timeout(POLL_INTERVAL) {
                Ok(slab) => {
                    self.take_error_if_any()?;
                    return Ok(Some(PooledBatch {
                        slab: Some(slab),
                        free_tx: self
                            .free_tx
                            .as_ref()
                            .expect("free sender present while pipeline is alive")
                            .clone(),
                    }));
                }
                Err(RecvTimeoutError::Timeout) => self.take_error_if_any()?,
                Err(RecvTimeoutError::Disconnected) => {
                    self.take_error_if_any()?;
                    return Ok(None);
                }
            }
        }
    }

    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            decoded_entries: self.stats.decoded_entries.load(Ordering::Acquire),
            encoded_entries: self.stats.encoded_entries.load(Ordering::Acquire),
            skipped_entries: self.stats.skipped_entries.load(Ordering::Acquire),
            produced_batches: self.stats.produced_batches.load(Ordering::Acquire),
            scanned_chunks: self.stats.scanned_chunks.load(Ordering::Acquire),
            decoded_queue_len: 0,
            ready_queue_len: self.ready_rx.as_ref().map_or(0, Receiver::len),
            free_queue_len: self.free_tx.as_ref().map_or(0, Sender::len),
        }
    }

    fn take_error_if_any(&self) -> Result<(), PipelineError> {
        if let Some(error) = self.error.lock().unwrap().clone() {
            return Err(error);
        }

        if let Some(error_rx) = &self.error_rx {
            if let Ok(message) = error_rx.try_recv() {
                let error = PipelineError::new(message);
                *self.error.lock().unwrap() = Some(error.clone());
                return Err(error);
            }
        }

        Ok(())
    }
}

impl Drop for BatchPipeline {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);

        self.ready_rx.take();
        self.free_tx.take();
        self.error_rx.take();

        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl PooledBatch {
    pub fn slab(&self) -> &HostBatchSlab {
        self.slab.as_ref().expect("pooled batch always owns a slab")
    }
}

impl std::ops::Deref for PooledBatch {
    type Target = HostBatchSlab;

    fn deref(&self) -> &Self::Target {
        self.slab()
    }
}

impl Drop for PooledBatch {
    fn drop(&mut self) {
        if let Some(mut slab) = self.slab.take() {
            slab.reset();
            let _ = self.free_tx.try_send(slab);
        }
    }
}

impl HostBatchSlab {
    pub fn new(batch_size: usize, feature_set: &FeatureSet) -> Self {
        let max_active_features = feature_set.max_active_features();
        let flat_size = batch_size * max_active_features;

        Self {
            num_inputs: feature_set.inputs(),
            max_active_features,
            capacity: batch_size,
            size: 0,
            num_active_white_features: 0,
            num_active_black_features: 0,
            white_counts: vec![0; batch_size],
            black_counts: vec![0; batch_size],
            is_white: vec![0.0; batch_size],
            outcome: vec![0.0; batch_size],
            score: vec![0.0; batch_size],
            white: vec![-1; flat_size],
            black: vec![-1; flat_size],
            psqt_indices: vec![0; batch_size],
            layer_stack_indices: vec![0; batch_size],
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    pub fn max_active_features(&self) -> usize {
        self.max_active_features
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    pub fn num_active_white_features(&self) -> usize {
        self.num_active_white_features
    }

    pub fn num_active_black_features(&self) -> usize {
        self.num_active_black_features
    }

    pub fn is_white_slice(&self) -> &[f32] {
        &self.is_white[..self.size]
    }

    pub fn outcome_slice(&self) -> &[f32] {
        &self.outcome[..self.size]
    }

    pub fn score_slice(&self) -> &[f32] {
        &self.score[..self.size]
    }

    pub fn white_flat_slice(&self) -> &[i32] {
        &self.white[..self.size * self.max_active_features]
    }

    pub fn black_flat_slice(&self) -> &[i32] {
        &self.black[..self.size * self.max_active_features]
    }

    pub fn psqt_indices_slice(&self) -> &[i64] {
        &self.psqt_indices[..self.size]
    }

    pub fn layer_stack_indices_slice(&self) -> &[i64] {
        &self.layer_stack_indices[..self.size]
    }

    pub fn copy_row(&self, row: usize) -> SparseRow {
        let start = row * self.max_active_features;
        let end = start + self.max_active_features;
        let white = self.white[start..end].to_vec();
        let black = self.black[start..end].to_vec();
        let white_values = white
            .iter()
            .map(|&value| if value < 0 { 0.0 } else { 1.0 })
            .collect();
        let black_values = black
            .iter()
            .map(|&value| if value < 0 { 0.0 } else { 1.0 })
            .collect();

        SparseRow {
            num_inputs: self.num_inputs,
            max_active_features: self.max_active_features,
            is_white: self.is_white[row],
            outcome: self.outcome[row],
            score: self.score[row],
            white_count: self.white_counts[row] as usize,
            black_count: self.black_counts[row] as usize,
            white,
            black,
            white_values,
            black_values,
            psqt_indices: self.psqt_indices[row],
            layer_stack_indices: self.layer_stack_indices[row],
        }
    }

    fn push_entry(&mut self, entry: &TrainingDataEntry, feature_set: &FeatureSet) {
        let row = self.size;
        let start = row * self.max_active_features;
        let end = start + self.max_active_features;
        let metadata = encode_training_entry_indices_only(
            entry,
            feature_set,
            &mut self.white[start..end],
            &mut self.black[start..end],
        );

        self.write_metadata(row, metadata);
        self.size += 1;
    }

    fn reset(&mut self) {
        self.size = 0;
        self.num_active_white_features = 0;
        self.num_active_black_features = 0;
    }

    fn write_metadata(&mut self, row: usize, metadata: RowMetadata) {
        self.white_counts[row] = metadata.white_count as u16;
        self.black_counts[row] = metadata.black_count as u16;
        self.is_white[row] = metadata.is_white;
        self.outcome[row] = metadata.outcome;
        self.score[row] = metadata.score;
        self.psqt_indices[row] = metadata.psqt_indices;
        self.layer_stack_indices[row] = metadata.layer_stack_indices;
        self.num_active_white_features += metadata.white_count;
        self.num_active_black_features += metadata.black_count;
    }
}

impl PipelineError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PipelineError {}

impl SkipDecider {
    fn new(config: SkipConfig) -> Self {
        let enabled = config.filtered
            || config.random_fen_skipping > 0
            || config.wld_filtered
            || config.early_fen_skipping >= 0;
        let random_skip_probability = if config.random_fen_skipping == 0 {
            0.0
        } else {
            config.random_fen_skipping as f64 / (config.random_fen_skipping as f64 + 1.0)
        };
        let desired_piece_count_weights_total = (0..=32)
            .map(|piece_count| desired_piece_count_weight(config, piece_count))
            .sum();

        Self {
            enabled,
            config,
            random_skip_probability,
            desired_piece_count_weights_total,
            alpha: 1.0,
            piece_count_history_all: [0.0; 33],
            piece_count_history_passed: [0.0; 33],
            piece_count_history_all_total: 0.0,
            piece_count_history_passed_total: 0.0,
        }
    }

    fn should_skip(&mut self, entry: &TrainingDataEntry, rng: &mut SmallRng) -> bool {
        if !self.enabled {
            return false;
        }

        if entry.score == VALUE_NONE {
            return true;
        }
        if entry.ply as i32 <= self.config.early_fen_skipping {
            return true;
        }
        if self.config.random_fen_skipping > 0 && rng.gen_bool(self.random_skip_probability) {
            return true;
        }
        if self.config.wld_filtered {
            let keep_prob = score_result_prob(entry);
            if keep_prob <= 0.0 || (keep_prob < 1.0 && rng.gen::<f32>() >= keep_prob) {
                return true;
            }
        }
        if self.config.filtered && (is_capturing_move(entry) || is_in_check(entry)) {
            return true;
        }
        if self.config.simple_eval_skipping > 0
            && simple_eval(&entry.pos).abs() < self.config.simple_eval_skipping
        {
            return true;
        }

        let piece_count = entry.pos.occupied().count() as usize;
        self.piece_count_history_all[piece_count] += 1.0;
        self.piece_count_history_all_total += 1.0;

        if (self.piece_count_history_all_total as u64).is_multiple_of(10_000) {
            let mut pass =
                self.piece_count_history_all_total * self.desired_piece_count_weights_total;

            for i in 0usize..=32 {
                let weight = desired_piece_count_weight(self.config, i as i32);
                if weight > 0.0 && self.piece_count_history_all[i] > 0.0 {
                    let tmp = self.piece_count_history_all_total * weight
                        / (self.desired_piece_count_weights_total
                            * self.piece_count_history_all[i]);
                    if tmp < pass {
                        pass = tmp;
                    }
                }
            }

            self.alpha = 1.0 / (pass * MAX_SKIP_RATE);
        }

        let tmp = (self.alpha
            * self.piece_count_history_all_total
            * desired_piece_count_weight(self.config, piece_count as i32)
            / (self.desired_piece_count_weights_total * self.piece_count_history_all[piece_count]))
            .min(1.0);

        if rng.gen_bool((1.0 - tmp).clamp(0.0, 1.0)) {
            return true;
        }

        self.piece_count_history_passed[piece_count] += 1.0;
        self.piece_count_history_passed_total += 1.0;
        false
    }
}

fn decoder_worker(
    thread_index: usize,
    files: Arc<Vec<PathBuf>>,
    chunk_scheduler: Arc<ChunkScheduler>,
    config: PipelineConfig,
    decoded_tx: Sender<Vec<TrainingDataEntry>>,
    error_tx: Sender<String>,
    stats: Arc<PipelineCounters>,
    shutdown: Arc<AtomicBool>,
) {
    if chunk_scheduler.task_count() == 0 {
        return;
    }

    let base_seed = config.seed.unwrap_or_else(random_seed);
    let mut rng = SmallRng::seed_from_u64(base_seed ^ thread_index as u64);
    let mut skip_decider = SkipDecider::new(config.skip_config);
    let mut shuffle_buffer = Vec::with_capacity(config.shuffle_buffer_entries.max(1));
    let mut accumulation = Vec::with_capacity(config.accumulation_entries);
    let mut chunk_buffer = Vec::new();
    let mut counters = DecoderWorkerCounters::default();
    let mut file_handles = std::iter::repeat_with(|| None)
        .take(files.len())
        .collect::<Vec<Option<File>>>();

    while let Some(task) = chunk_scheduler.claim() {
        if let Err(error) =
            read_chunk_into_buffer(&mut file_handles, files.as_slice(), task, &mut chunk_buffer)
        {
            flush_decoder_counters(&mut counters, &stats);
            report_error(
                &error_tx,
                format!(
                    "failed to read chunk from {} at offset {}: {error}",
                    files[task.file_index].display(),
                    task.offset
                ),
            );
            return;
        }

        let cursor = Cursor::new(chunk_buffer.as_slice());
        let mut reader = match CompressedTrainingDataEntryReader::new(cursor) {
            Ok(reader) => reader,
            Err(CompressedReaderError::EndOfFile) => continue,
            Err(error) => {
                flush_decoder_counters(&mut counters, &stats);
                report_error(
                    &error_tx,
                    format!(
                        "failed to decode chunk from {} at offset {}: {error}",
                        files[task.file_index].display(),
                        task.offset
                    ),
                );
                return;
            }
        };

        while reader.has_next() {
            let entry = reader.next();
            counters.decoded_entries += 1;

            if skip_decider.should_skip(&entry, &mut rng) {
                counters.skipped_entries += 1;
                flush_decoder_counters_if_needed(&mut counters, &stats);
                continue;
            }

            if config.shuffle_buffer_entries == 0 {
                accumulation.push(entry);
            } else {
                shuffle_buffer.push(entry);
                if shuffle_buffer.len() >= config.shuffle_buffer_entries {
                    let index = rng.gen_range(0..shuffle_buffer.len());
                    accumulation.push(shuffle_buffer.swap_remove(index));
                }
            }

            if accumulation.len() >= config.accumulation_entries
                && publish_decoded_chunk(&decoded_tx, &mut accumulation, &error_tx, &shutdown)
                    .is_err()
            {
                flush_decoder_counters(&mut counters, &stats);
                return;
            }

            flush_decoder_counters_if_needed(&mut counters, &stats);
        }
    }

    while let Some(entry) = pop_shuffled_entry(&mut shuffle_buffer, &mut rng) {
        accumulation.push(entry);
        if accumulation.len() >= config.accumulation_entries
            && publish_decoded_chunk(&decoded_tx, &mut accumulation, &error_tx, &shutdown).is_err()
        {
            flush_decoder_counters(&mut counters, &stats);
            return;
        }
    }

    flush_decoder_counters(&mut counters, &stats);
    if !accumulation.is_empty() {
        let _ = publish_decoded_chunk(&decoded_tx, &mut accumulation, &error_tx, &shutdown);
    }
}

fn publish_decoded_chunk(
    decoded_tx: &Sender<Vec<TrainingDataEntry>>,
    accumulation: &mut Vec<TrainingDataEntry>,
    error_tx: &Sender<String>,
    shutdown: &Arc<AtomicBool>,
) -> Result<(), ()> {
    if accumulation.is_empty() {
        return Ok(());
    }

    let mut chunk = Vec::with_capacity(accumulation.len());
    std::mem::swap(accumulation, &mut chunk);

    loop {
        match decoded_tx.try_send(chunk) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(c)) => {
                chunk = c;
                if shutdown.load(Ordering::Relaxed) {
                    return Err(());
                }
                std::thread::sleep(POLL_INTERVAL);
            }
            Err(TrySendError::Disconnected(_)) => {
                report_error(error_tx, "decoded channel closed unexpectedly".to_string());
                return Err(());
            }
        }
    }
}

fn pop_shuffled_entry(
    buffer: &mut Vec<TrainingDataEntry>,
    rng: &mut SmallRng,
) -> Option<TrainingDataEntry> {
    if buffer.is_empty() {
        None
    } else {
        Some(buffer.swap_remove(rng.gen_range(0..buffer.len())))
    }
}

fn encoder_worker(
    _thread_index: usize,
    feature_set: FeatureSet,
    decoded_rx: Receiver<Vec<TrainingDataEntry>>,
    ready_tx: Sender<HostBatchSlab>,
    free_rx: Receiver<HostBatchSlab>,
    stats: Arc<PipelineCounters>,
    shutdown: Arc<AtomicBool>,
) {
    let mut current_slab: Option<HostBatchSlab> = None;
    let mut counters = EncoderWorkerCounters::default();

    while let Ok(entries) = decoded_rx.recv() {
        for entry in entries {
            if current_slab.is_none() {
                loop {
                    match free_rx.recv_timeout(POLL_INTERVAL) {
                        Ok(slab) => {
                            current_slab = Some(slab);
                            break;
                        }
                        Err(RecvTimeoutError::Timeout) => {
                            if shutdown.load(Ordering::Relaxed) {
                                flush_encoder_counters(&mut counters, &stats);
                                return;
                            }
                        }
                        Err(RecvTimeoutError::Disconnected) => {
                            flush_encoder_counters(&mut counters, &stats);
                            return;
                        }
                    }
                }
            }

            let slab = current_slab.as_mut().unwrap();
            slab.push_entry(&entry, &feature_set);
            counters.encoded_entries += 1;
            flush_encoder_counters_if_needed(&mut counters, &stats);

            if slab.is_full() {
                let ready_slab = current_slab.take().unwrap();
                if ready_tx.send(ready_slab).is_err() {
                    flush_encoder_counters(&mut counters, &stats);
                    return;
                }
                stats.produced_batches.fetch_add(1, Ordering::AcqRel);
            }
        }
    }

    flush_encoder_counters(&mut counters, &stats);
    if let Some(slab) = current_slab.take() {
        if slab.is_empty() {
            let _ = ready_tx;
        } else if ready_tx.send(slab).is_ok() {
            stats.produced_batches.fetch_add(1, Ordering::AcqRel);
        }
    }
}

fn report_error(error_tx: &Sender<String>, message: String) {
    match error_tx.try_send(message) {
        Ok(()) | Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {}
    }
}

impl ChunkScheduler {
    fn new(tasks: Vec<ChunkTask>, cyclic: bool) -> Self {
        Self {
            tasks: tasks.into(),
            next_index: AtomicUsize::new(0),
            cyclic,
        }
    }

    fn task_count(&self) -> usize {
        self.tasks.len()
    }

    fn claim(&self) -> Option<ChunkTask> {
        if self.tasks.is_empty() {
            return None;
        }

        let index = self.next_index.fetch_add(1, Ordering::AcqRel);
        if self.cyclic {
            Some(self.tasks[index % self.tasks.len()])
        } else {
            self.tasks.get(index).copied()
        }
    }
}

fn flush_decoder_counters_if_needed(
    counters: &mut DecoderWorkerCounters,
    stats: &PipelineCounters,
) {
    if counters.decoded_entries >= DECODER_COUNTER_FLUSH_INTERVAL {
        flush_decoder_counters(counters, stats);
    }
}

fn flush_decoder_counters(counters: &mut DecoderWorkerCounters, stats: &PipelineCounters) {
    if counters.decoded_entries > 0 {
        stats
            .decoded_entries
            .fetch_add(counters.decoded_entries, Ordering::AcqRel);
        counters.decoded_entries = 0;
    }
    if counters.skipped_entries > 0 {
        stats
            .skipped_entries
            .fetch_add(counters.skipped_entries, Ordering::AcqRel);
        counters.skipped_entries = 0;
    }
}

fn flush_encoder_counters_if_needed(
    counters: &mut EncoderWorkerCounters,
    stats: &PipelineCounters,
) {
    if counters.encoded_entries >= ENCODER_COUNTER_FLUSH_INTERVAL {
        flush_encoder_counters(counters, stats);
    }
}

fn flush_encoder_counters(counters: &mut EncoderWorkerCounters, stats: &PipelineCounters) {
    if counters.encoded_entries > 0 {
        stats
            .encoded_entries
            .fetch_add(counters.encoded_entries, Ordering::AcqRel);
        counters.encoded_entries = 0;
    }
}

fn scan_chunk_tasks(
    files: &[PathBuf],
    rank: usize,
    world_size: usize,
) -> Result<Vec<ChunkTask>, PipelineError> {
    let mut tasks = Vec::new();
    for (file_index, path) in files.iter().enumerate() {
        scan_file_chunk_tasks(path, file_index, &mut tasks)?;
    }

    if world_size == 1 {
        return Ok(tasks);
    }

    Ok(tasks
        .into_iter()
        .enumerate()
        .filter_map(|(index, task)| (index % world_size == rank).then_some(task))
        .collect())
}

// Sidecar chunk-index cache (.chunks files)
// Format (little-endian): magic[4] | version:u32 | source_file_size:u64 | num_chunks:u32 | [offset:u64, size:u32]*

const SIDECAR_MAGIC: &[u8; 4] = b"CHNK";
const SIDECAR_VERSION: u32 = 1;
const SIDECAR_HEADER_SIZE: usize = 4 + 4 + 8 + 4;
const SIDECAR_ENTRY_SIZE: usize = 8 + 4;

fn sidecar_path(binpack_path: &PathBuf) -> PathBuf {
    let mut p = binpack_path.as_os_str().to_owned();
    p.push(".chunks");
    PathBuf::from(p)
}

fn load_sidecar_cache(
    binpack_path: &PathBuf,
    expected_file_size: u64,
    file_index: usize,
) -> Option<Vec<ChunkTask>> {
    let data = std::fs::read(&sidecar_path(binpack_path)).ok()?;

    if data.len() < SIDECAR_HEADER_SIZE
        || &data[0..4] != SIDECAR_MAGIC
        || u32::from_le_bytes(data[4..8].try_into().ok()?) != SIDECAR_VERSION
        || u64::from_le_bytes(data[8..16].try_into().ok()?) != expected_file_size
    {
        return None;
    }

    let num_chunks = u32::from_le_bytes(data[16..20].try_into().ok()?) as usize;
    if data.len() != SIDECAR_HEADER_SIZE + num_chunks * SIDECAR_ENTRY_SIZE {
        return None;
    }

    let mut tasks = Vec::with_capacity(num_chunks);
    for i in 0..num_chunks {
        let base = SIDECAR_HEADER_SIZE + i * SIDECAR_ENTRY_SIZE;
        let offset = u64::from_le_bytes(data[base..base + 8].try_into().ok()?);
        let size = u32::from_le_bytes(data[base + 8..base + 12].try_into().ok()?);
        tasks.push(ChunkTask {
            file_index,
            offset,
            size,
        });
    }

    Some(tasks)
}

fn write_sidecar_cache(binpack_path: &PathBuf, file_size: u64, tasks: &[ChunkTask]) {
    let cache_path = sidecar_path(binpack_path);
    let mut buf = Vec::with_capacity(SIDECAR_HEADER_SIZE + tasks.len() * SIDECAR_ENTRY_SIZE);

    buf.extend_from_slice(SIDECAR_MAGIC);
    buf.extend_from_slice(&SIDECAR_VERSION.to_le_bytes());
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(&(tasks.len() as u32).to_le_bytes());
    for task in tasks {
        buf.extend_from_slice(&task.offset.to_le_bytes());
        buf.extend_from_slice(&task.size.to_le_bytes());
    }

    // Atomic write-then-rename so concurrent writers don't produce corrupt reads.
    let tmp_path = {
        let mut p = cache_path.as_os_str().to_owned();
        p.push(&format!(".tmp.{}", std::process::id()));
        PathBuf::from(p)
    };
    if std::fs::write(&tmp_path, &buf).is_ok() {
        if std::fs::rename(&tmp_path, &cache_path).is_err() {
            let _ = std::fs::remove_file(&tmp_path);
        }
    }
}

fn read_chunk_header_at(
    file: &mut File,
    offset: u64,
    header: &mut [u8; BINPACK_HEADER_SIZE],
) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let mut filled = 0;
        while filled < header.len() {
            let read = file.read_at(&mut header[filled..], offset + filled as u64)?;
            if read == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "unexpected EOF while reading binpack chunk header",
                ));
            }
            filled += read;
        }
        Ok(())
    }

    #[cfg(not(unix))]
    {
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(header)
    }
}

fn scan_file_chunk_tasks(
    path: &PathBuf,
    file_index: usize,
    tasks: &mut Vec<ChunkTask>,
) -> Result<(), PipelineError> {
    let file_len = std::fs::metadata(path)
        .map_err(|error| {
            PipelineError::new(format!(
                "failed to read metadata for {}: {error}",
                path.display()
            ))
        })?
        .len();

    if let Some(cached) = load_sidecar_cache(path, file_len, file_index) {
        eprintln!(
            "chunk cache hit: {} ({} chunks)",
            path.display(),
            cached.len()
        );
        tasks.extend(cached);
        return Ok(());
    }

    eprintln!("chunk cache miss, scanning: {}", path.display());
    let start_len = tasks.len();

    let mut file = File::open(path).map_err(|error| {
        PipelineError::new(format!("failed to open {}: {error}", path.display()))
    })?;
    let mut header = [0u8; BINPACK_HEADER_SIZE];
    let mut offset = 0u64;

    while offset < file_len {
        read_chunk_header_at(&mut file, offset, &mut header).map_err(|error| {
            PipelineError::new(format!(
                "failed to read chunk header from {} at offset {}: {error}",
                path.display(),
                offset
            ))
        })?;

        if &header[..4] != b"BINP" {
            return Err(PipelineError::new(format!(
                "invalid binpack chunk magic in {} at offset {}",
                path.display(),
                offset
            )));
        }

        let size = u32::from_le_bytes(header[4..8].try_into().unwrap());
        if size > BINPACK_MAX_CHUNK_SIZE {
            return Err(PipelineError::new(format!(
                "chunk in {} at offset {} exceeds supported size",
                path.display(),
                offset
            )));
        }

        let next_offset = offset + BINPACK_HEADER_SIZE as u64 + size as u64;
        if next_offset > file_len {
            return Err(PipelineError::new(format!(
                "chunk in {} at offset {} is truncated",
                path.display(),
                offset
            )));
        }

        tasks.push(ChunkTask {
            file_index,
            offset,
            size,
        });

        offset = next_offset;
    }

    let new_tasks = &tasks[start_len..];
    eprintln!(
        "writing chunk cache: {} ({} chunks)",
        path.display(),
        new_tasks.len()
    );
    write_sidecar_cache(path, file_len, new_tasks);

    Ok(())
}

fn read_chunk_into_buffer(
    file_handles: &mut [Option<File>],
    files: &[PathBuf],
    task: ChunkTask,
    buffer: &mut Vec<u8>,
) -> std::io::Result<()> {
    if file_handles[task.file_index].is_none() {
        file_handles[task.file_index] = Some(File::open(&files[task.file_index])?);
    }

    let file = file_handles[task.file_index].as_mut().unwrap();
    buffer.resize(BINPACK_HEADER_SIZE + task.size as usize, 0);
    file.seek(SeekFrom::Start(task.offset))?;
    file.read_exact(buffer)?;
    Ok(())
}

fn desired_piece_count_weight(config: SkipConfig, piece_count: i32) -> f64 {
    let x = piece_count as f64;
    let x1 = 0.0;
    let y1 = config.pc_y1;
    let x2 = 16.0;
    let y2 = config.pc_y2;
    let x3 = 32.0;
    let y3 = config.pc_y3;
    let l1 = (x - x2) * (x - x3) / ((x1 - x2) * (x1 - x3));
    let l2 = (x - x1) * (x - x3) / ((x2 - x1) * (x2 - x3));
    let l3 = (x - x1) * (x - x2) / ((x3 - x1) * (x3 - x2));

    l1 * y1 + l2 * y2 + l3 * y3
}

fn is_capturing_move(entry: &TrainingDataEntry) -> bool {
    let to = entry.mv.to();
    let from = entry.mv.from();

    if to == sfbinpack::chess::coords::Square::NONE
        || from == sfbinpack::chess::coords::Square::NONE
    {
        return false;
    }

    let captured = entry.pos.piece_at(to);
    let moving = entry.pos.piece_at(from);
    captured != Piece::none() && captured.color() != moving.color()
}

fn is_in_check(entry: &TrainingDataEntry) -> bool {
    entry.pos.is_checked(entry.pos.side_to_move())
}

fn simple_eval(pos: &sfbinpack::chess::position::Position) -> i32 {
    let side_to_move_sign = if pos.side_to_move() == Color::White {
        1
    } else {
        -1
    };
    side_to_move_sign
        * (208 * material_count(pos, Color::White, PieceType::Pawn)
            - 208 * material_count(pos, Color::Black, PieceType::Pawn)
            + 781 * material_count(pos, Color::White, PieceType::Knight)
            - 781 * material_count(pos, Color::Black, PieceType::Knight)
            + 825 * material_count(pos, Color::White, PieceType::Bishop)
            - 825 * material_count(pos, Color::Black, PieceType::Bishop)
            + 1276 * material_count(pos, Color::White, PieceType::Rook)
            - 1276 * material_count(pos, Color::Black, PieceType::Rook)
            + 2538 * material_count(pos, Color::White, PieceType::Queen)
            - 2538 * material_count(pos, Color::Black, PieceType::Queen))
}

fn material_count(
    pos: &sfbinpack::chess::position::Position,
    color: Color,
    piece_type: PieceType,
) -> i32 {
    pos.pieces_bb_color(color, piece_type).count() as i32
}

fn score_result_prob(entry: &TrainingDataEntry) -> f32 {
    let params = &wld_params()[entry.ply.min(WLD_MAX_PLY as u16) as usize];
    let x = (entry.score as f32 * WLD_SCORE_SCALE).clamp(-2000.0, 2000.0);

    if entry.result > 0 {
        1.0 / (1.0 + ((params.a - x) * params.inv_b).exp())
    } else if entry.result < 0 {
        1.0 / (1.0 + ((params.a + x) * params.inv_b).exp())
    } else {
        let w = 1.0 / (1.0 + ((params.a - x) * params.inv_b).exp());
        let l = 1.0 / (1.0 + ((params.a + x) * params.inv_b).exp());
        (1.0 - w - l).max(0.0)
    }
}

fn wld_params() -> &'static [WldParams] {
    static WLD_PARAMS: std::sync::OnceLock<Box<[WldParams]>> = std::sync::OnceLock::new();
    WLD_PARAMS.get_or_init(|| {
        let as_ = [-3.683_893_f32, 30.070_66, -60.528_79, 149.533_78];
        let bs = [-2.018_185_6_f32, 15.856_851, -29.834_52, 47.590_79];

        (0..=WLD_MAX_PLY)
            .map(|ply| {
                let m = ply as f32 / 64.0;
                let a = (((as_[0] * m + as_[1]) * m + as_[2]) * m) + as_[3];
                let b = ((((bs[0] * m + bs[1]) * m + bs[2]) * m) + bs[3]) * 1.5;
                WldParams { a, inv_b: 1.0 / b }
            })
            .collect()
    })
}

fn random_seed() -> u64 {
    rand::thread_rng().gen()
}

pub fn default_batch_slab_count(total_threads: usize) -> usize {
    (total_threads / 2).clamp(8, 16)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::str::FromStr;

    use sfbinpack::chess::{
        coords::Square,
        piece::Piece,
        position::Position,
        r#move::{Move, MoveType},
    };
    use sfbinpack::CompressedTrainingDataEntryWriter;

    use super::*;
    use crate::feature_extraction::{build_sparse_row_for_feature_set, encode_training_entry};

    #[test]
    fn feature_index_cross_validation_matches_legacy_encoding() {
        let feature_sets = [
            FeatureSet::from_str("HalfKAv2_hm").unwrap(),
            FeatureSet::from_str("Full_Threats").unwrap(),
            FeatureSet::from_str("Full_Threats+HalfKAv2_hm").unwrap(),
        ];
        let positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bq1rk1/pp2bppp/2n2n2/2pp4/2P5/2NP1NP1/PP2PPBP/R1BQ1RK1 w - - 0 9",
            "8/2k5/8/2p5/2P5/2K5/8/8 w - - 0 1",
            "r3k2r/pppq1ppp/2npbn2/3Np3/2B1P3/2N5/PPP2PPP/R1BQ1RK1 b kq - 2 10",
        ];

        for feature_set in feature_sets {
            for fen in positions {
                let entry = make_entry(fen, Move::normal(sq("a1"), sq("a1")), 17, 12, -1);
                let max = feature_set.max_active_features();
                let mut legacy_white = vec![-1; max];
                let mut legacy_white_values = vec![0.0; max];
                let mut legacy_black = vec![-1; max];
                let mut legacy_black_values = vec![0.0; max];
                let legacy = encode_training_entry(
                    &entry,
                    &feature_set,
                    &mut legacy_white,
                    &mut legacy_white_values,
                    &mut legacy_black,
                    &mut legacy_black_values,
                );

                let mut white = vec![-1; max];
                let mut black = vec![-1; max];
                let actual = encode_training_entry_indices_only(
                    &entry,
                    &feature_set,
                    &mut white,
                    &mut black,
                );

                assert_eq!(actual, legacy);
                assert_eq!(white, legacy_white);
                assert_eq!(black, legacy_black);
                assert_valid_row(&white, actual.white_count, feature_set.inputs());
                assert_valid_row(&black, actual.black_count, feature_set.inputs());
            }
        }
    }

    #[test]
    fn pipeline_round_trip_matches_direct_encoding_without_shuffle() {
        let feature_set = FeatureSet::from_str("Full_Threats+HalfKAv2_hm").unwrap();
        let entries = sample_entries();
        let file = write_entries(&entries);
        let mut config = PipelineConfig::new(vec![file.clone()], feature_set.clone(), 2);
        config.total_threads = 2;
        config.slab_count = 2;
        config.shuffle_buffer_entries = 0;
        config.seed = Some(0);

        let pipeline = BatchPipeline::new(config).unwrap();
        let actual = collect_pipeline_rows(&pipeline).unwrap();
        let expected = entries
            .iter()
            .map(|entry| {
                build_sparse_row_for_feature_set(
                    &entry.pos,
                    entry.score as i32,
                    entry.result as i32,
                    &feature_set,
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
        let _ = fs::remove_file(file);
    }

    #[test]
    fn shuffle_output_is_a_permutation_and_changes_with_seed() {
        let feature_set = FeatureSet::from_str("HalfKAv2_hm").unwrap();
        let entries = repeated_entries(24);
        let file = write_entries(&entries);
        let original = expected_keys(&entries, &feature_set);
        let mut orders = Vec::new();

        for seed in [1_u64, 7_u64] {
            let mut config = PipelineConfig::new(vec![file.clone()], feature_set.clone(), 4);
            config.total_threads = 2;
            config.slab_count = 2;
            config.shuffle_buffer_entries = 8;
            config.seed = Some(seed);

            let pipeline = BatchPipeline::new(config).unwrap();
            let rows = collect_pipeline_rows(&pipeline).unwrap();
            let keys = rows.iter().map(row_key).collect::<Vec<_>>();
            let mut actual_sorted = keys.clone();
            let mut expected_sorted = original.clone();
            actual_sorted.sort();
            expected_sorted.sort();
            assert_eq!(actual_sorted, expected_sorted);
            orders.push(keys);
        }

        assert!(orders.iter().any(|order| order != &original));
        assert_ne!(orders[0], orders[1]);
        let _ = fs::remove_file(file);
    }

    #[test]
    fn skip_filter_removes_known_entries() {
        let feature_set = FeatureSet::from_str("HalfKAv2_hm").unwrap();
        let entries = vec![
            make_entry(
                "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                Move::normal(sq("e1"), sq("e2")),
                VALUE_NONE,
                10,
                0,
            ),
            make_entry(
                "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                Move::normal(sq("e1"), sq("e2")),
                50,
                1,
                0,
            ),
            make_entry(
                "4k3/8/8/8/8/8/4p3/4K3 w - - 0 1",
                Move::normal(sq("e1"), sq("e2")),
                50,
                10,
                0,
            ),
            make_entry(
                "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                Move::normal(sq("e1"), sq("d1")),
                200,
                20,
                1,
            ),
        ];
        let file = write_entries(&entries);
        let mut config = PipelineConfig::new(vec![file.clone()], feature_set, 8);
        config.total_threads = 2;
        config.shuffle_buffer_entries = 0;
        config.skip_config = SkipConfig {
            filtered: true,
            random_fen_skipping: 0,
            wld_filtered: false,
            early_fen_skipping: 4,
            simple_eval_skipping: 0,
            param_index: 0,
            pc_y1: 1.0,
            pc_y2: 1.0,
            pc_y3: 1.0,
        };

        let pipeline = BatchPipeline::new(config).unwrap();
        let rows = collect_pipeline_rows(&pipeline).unwrap();
        let keys = rows.iter().map(row_key).collect::<Vec<_>>();
        let skipped = entries[..3]
            .iter()
            .map(|entry| {
                row_key(&build_sparse_row_for_feature_set(
                    &entry.pos,
                    entry.score as i32,
                    entry.result as i32,
                    &FeatureSet::halfka(),
                ))
            })
            .collect::<Vec<_>>();
        assert!(skipped.iter().all(|key| !keys.contains(key)));
        let _ = fs::remove_file(file);
    }

    #[test]
    fn ddp_sharding_produces_disjoint_outputs() {
        let feature_set = FeatureSet::from_str("HalfKAv2_hm").unwrap();
        let files = vec![
            write_entries(&repeated_entries_with_offset(6, 0)),
            write_entries(&repeated_entries_with_offset(6, 6)),
        ];

        let mut rank0 = PipelineConfig::new(files.clone(), feature_set.clone(), 4);
        rank0.total_threads = 2;
        rank0.shuffle_buffer_entries = 0;
        rank0.rank = 0;
        rank0.world_size = 2;

        let mut rank1 = PipelineConfig::new(files.clone(), feature_set.clone(), 4);
        rank1.total_threads = 2;
        rank1.shuffle_buffer_entries = 0;
        rank1.rank = 1;
        rank1.world_size = 2;

        let rows0 = collect_pipeline_rows(&BatchPipeline::new(rank0).unwrap()).unwrap();
        let rows1 = collect_pipeline_rows(&BatchPipeline::new(rank1).unwrap()).unwrap();
        let keys0 = rows0
            .iter()
            .map(row_key)
            .collect::<std::collections::BTreeSet<_>>();
        let keys1 = rows1
            .iter()
            .map(row_key)
            .collect::<std::collections::BTreeSet<_>>();

        assert!(keys0.is_disjoint(&keys1));

        for file in files {
            let _ = fs::remove_file(file);
        }
    }

    #[test]
    #[ignore = "manual throughput smoke test"]
    fn throughput_smoke_test() {
        let feature_set = FeatureSet::from_str("HalfKAv2_hm").unwrap();
        let entries = repeated_entries(20_000);
        let file = write_entries(&entries);
        let mut config = PipelineConfig::new(vec![file.clone()], feature_set, 1024);
        config.total_threads = 4;
        config.shuffle_buffer_entries = 0;
        let start = std::time::Instant::now();
        let pipeline = BatchPipeline::new(config).unwrap();
        let rows = collect_pipeline_rows(&pipeline).unwrap();
        let elapsed = start.elapsed().as_secs_f64();
        let throughput = rows.len() as f64 / elapsed;

        assert!(throughput > 10_000.0);
        let _ = fs::remove_file(file);
    }

    fn collect_pipeline_rows(pipeline: &BatchPipeline) -> Result<Vec<SparseRow>, PipelineError> {
        let mut rows = Vec::new();
        while let Some(batch) = pipeline.next_batch()? {
            for row in 0..batch.len() {
                rows.push(batch.copy_row(row));
            }
        }
        Ok(rows)
    }

    fn write_entries(entries: &[TrainingDataEntry]) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "stockfish_trainer_pipeline_test_{}_{}.binpack",
            std::process::id(),
            random_seed()
        ));
        let writer_file = fs::File::create(&path).unwrap();
        let mut writer = CompressedTrainingDataEntryWriter::new(writer_file).unwrap();
        for entry in entries {
            writer.write_entry(entry).unwrap();
        }
        writer.flush_and_end();
        path
    }

    fn sample_entries() -> Vec<TrainingDataEntry> {
        vec![
            make_entry(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                Move::normal(sq("e2"), sq("e4")),
                12,
                1,
                0,
            ),
            make_entry(
                "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
                Move::castle(Square::E1, Square::H1),
                -44,
                8,
                1,
            ),
            make_entry(
                "4k3/1q6/8/3P4/8/8/6Q1/4K3 w - - 0 1",
                Move::normal(sq("g2"), sq("g7")),
                87,
                17,
                -1,
            ),
            make_entry(
                "rnbq1bnr/ppppkppp/8/4p3/3P4/2N5/PPP1PPPP/R1BQKBNR b KQ - 3 4",
                Move::normal(sq("e7"), sq("e6")),
                -120,
                9,
                0,
            ),
        ]
    }

    fn repeated_entries(count: usize) -> Vec<TrainingDataEntry> {
        repeated_entries_with_offset(count, 0)
    }

    fn repeated_entries_with_offset(count: usize, offset: usize) -> Vec<TrainingDataEntry> {
        let base = sample_entries();
        (0..count)
            .map(|index| {
                let mut entry = base[(index + offset) % base.len()];
                entry.score = (((index + offset) as i32 * 37) % 400 - 200) as i16;
                entry.ply = (10 + (index + offset) % 80) as u16;
                entry.pos.set_ply(entry.ply);
                entry.result = match (index + offset) % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                };
                entry
            })
            .collect()
    }

    fn expected_keys(entries: &[TrainingDataEntry], feature_set: &FeatureSet) -> Vec<String> {
        entries
            .iter()
            .map(|entry| {
                row_key(&build_sparse_row_for_feature_set(
                    &entry.pos,
                    entry.score as i32,
                    entry.result as i32,
                    feature_set,
                ))
            })
            .collect()
    }

    fn row_key(row: &SparseRow) -> String {
        format!("{row:?}")
    }

    fn assert_valid_row(features: &[i32], active_count: usize, num_inputs: usize) {
        assert!(features[..active_count]
            .iter()
            .all(|&feature| (0..num_inputs as i32).contains(&feature)));
        assert!(features[active_count..]
            .iter()
            .all(|&feature| feature == -1));
        let mut seen = features[..active_count].to_vec();
        seen.sort_unstable();
        for pair in seen.windows(2) {
            assert_ne!(pair[0], pair[1]);
        }
    }

    fn make_entry(fen: &str, mv: Move, score: i16, ply: u16, result: i16) -> TrainingDataEntry {
        let mut pos = Position::from_fen(fen).unwrap();
        pos.set_ply(ply);
        TrainingDataEntry {
            pos,
            mv,
            score,
            ply,
            result,
        }
    }

    fn sq(name: &str) -> Square {
        Square::from_string(name).unwrap()
    }

    #[allow(dead_code)]
    fn _promotion_entry() -> TrainingDataEntry {
        make_entry(
            "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1",
            Move::new(
                sq("e7"),
                Square::E8,
                MoveType::Promotion,
                Piece::WHITE_QUEEN,
            ),
            300,
            30,
            1,
        )
    }
}
