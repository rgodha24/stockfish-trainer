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
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use sfbinpack::chess::{
    bitboard::Bitboard,
    castling_rights::CastlingRights,
    color::Color,
    coords::{FlatSquareOffset, Rank, Square},
    piece::Piece,
    piecetype::PieceType,
    position::Position,
    r#move::{Move, MoveType},
};
use sfbinpack::{CompressedReaderError, CompressedTrainingDataEntryReader, TrainingDataEntry};

use crate::feature_extraction::{encode_training_entry_indices_only, FeatureSet, RowMetadata};

pub const PACKED_ENTRY_BYTES: usize = 32;

const VALUE_NONE: i16 = 32002;
const MAX_SKIP_RATE: f64 = 10.0;
const BINPACK_HEADER_SIZE: usize = 8;
const BINPACK_MAX_CHUNK_SIZE: u32 = 100 * 1024 * 1024;
const DECODER_COUNTER_FLUSH_INTERVAL: u64 = 4096;
const POLL_INTERVAL: Duration = Duration::from_millis(50);
const DEFAULT_CHUNK_ENTRIES: usize = 8192;
const WLD_MAX_PLY: usize = 240;
const WLD_SCORE_SCALE: f32 = 100.0 / 208.0;

#[derive(Clone, Debug)]
pub struct PackedStreamConfig {
    pub files: Vec<PathBuf>,
    pub total_threads: usize,
    pub decode_threads: Option<usize>,
    pub chunk_entries: usize,
    pub shuffle_buffer_entries: usize,
    pub cyclic: bool,
    pub skip_config: SkipConfig,
    pub seed: Option<u64>,
    pub rank: usize,
    pub world_size: usize,
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
pub struct PackedStreamStats {
    pub decoded_entries: u64,
    pub skipped_entries: u64,
    pub produced_chunks: u64,
    pub scanned_chunks: u64,
    pub chunk_queue_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineError {
    message: String,
}

pub struct PackedEntryStream {
    chunk_rx: Option<Receiver<Vec<u8>>>,
    error_rx: Option<Receiver<String>>,
    error: Mutex<Option<PipelineError>>,
    stats: Arc<PipelineCounters>,
    workers: Vec<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    pub chunk_entries: usize,
}

#[derive(Debug, Clone)]
pub struct EncodedBatch {
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
    skipped_entries: AtomicU64,
    produced_chunks: AtomicU64,
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
    produced_chunks: u64,
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

impl PackedStreamConfig {
    pub fn new(files: Vec<PathBuf>) -> Self {
        let available_threads = thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(2);

        Self {
            files,
            total_threads: available_threads.saturating_sub(1).max(1),
            decode_threads: None,
            chunk_entries: DEFAULT_CHUNK_ENTRIES,
            shuffle_buffer_entries: 16_384,
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
        if self.total_threads == 0 {
            return Err(PipelineError::new(
                "total_threads must be greater than zero",
            ));
        }
        if self.chunk_entries == 0 {
            return Err(PipelineError::new(
                "chunk_entries must be greater than zero",
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

    fn skip_heavy(&self) -> bool {
        self.skip_config.filtered
            || self.skip_config.wld_filtered
            || self.skip_config.random_fen_skipping > 0
            || self.skip_config.early_fen_skipping >= 0
            || self.skip_config.simple_eval_skipping > 0
    }

    fn decode_thread_count(&self, task_count: usize) -> usize {
        let available_tasks = task_count.max(1);
        if let Some(count) = self.decode_threads {
            return count.max(1).min(available_tasks);
        }
        let (decode, _) = default_thread_counts(self.total_threads, task_count, self.skip_heavy());
        decode.min(available_tasks).max(1)
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

impl PackedEntryStream {
    pub fn new(config: PackedStreamConfig) -> Result<Self, PipelineError> {
        config.validate()?;
        let scanned_chunks = scan_chunk_tasks(&config.files, config.rank, config.world_size)?;
        let scanned_chunk_count = scanned_chunks.len() as u64;
        let decode_threads = config.decode_thread_count(scanned_chunks.len());
        let chunk_capacity = decode_threads.saturating_mul(4).max(16);
        let (chunk_tx, chunk_rx) = bounded::<Vec<u8>>(chunk_capacity);
        let (error_tx, error_rx) = bounded::<String>(1);

        let stats = Arc::new(PipelineCounters::default());
        stats
            .scanned_chunks
            .store(scanned_chunk_count, Ordering::Release);

        let files = Arc::new(config.files.clone());
        let chunk_scheduler = Arc::new(ChunkScheduler::new(scanned_chunks, config.cyclic));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut workers = Vec::with_capacity(decode_threads);

        for thread_index in 0..decode_threads {
            let files = Arc::clone(&files);
            let chunk_scheduler = Arc::clone(&chunk_scheduler);
            let chunk_tx = chunk_tx.clone();
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
                    chunk_tx,
                    error_tx,
                    stats,
                    shutdown,
                )
            }));
        }
        drop(chunk_tx);

        Ok(Self {
            chunk_rx: Some(chunk_rx),
            error_rx: Some(error_rx),
            error: Mutex::new(None),
            stats,
            workers,
            shutdown,
            chunk_entries: config.chunk_entries,
        })
    }

    pub fn next_batch(&self, batch_entries: usize) -> Result<Option<Vec<u8>>, PipelineError> {
        if batch_entries == 0 {
            return Err(PipelineError::new("batch_entries must be greater than zero"));
        }
        if batch_entries % self.chunk_entries != 0 {
            return Err(PipelineError::new(
                "batch_entries must be divisible by chunk_entries",
            ));
        }

        let chunks_per_batch = batch_entries / self.chunk_entries;
        let expected_chunk_bytes = self.chunk_entries * PACKED_ENTRY_BYTES;
        let mut batch = Vec::with_capacity(batch_entries * PACKED_ENTRY_BYTES);

        for _ in 0..chunks_per_batch {
            let Some(chunk) = self.next_chunk()? else {
                return Ok(None);
            };
            if chunk.len() != expected_chunk_bytes {
                return Err(PipelineError::new(format!(
                    "packed chunk length {} does not match expected {}",
                    chunk.len(),
                    expected_chunk_bytes,
                )));
            }
            batch.extend_from_slice(&chunk);
        }

        Ok(Some(batch))
    }

    pub fn next_chunk(&self) -> Result<Option<Vec<u8>>, PipelineError> {
        self.take_error_if_any()?;

        let chunk_rx = self
            .chunk_rx
            .as_ref()
            .expect("chunk receiver present while stream is alive");

        loop {
            match chunk_rx.recv_timeout(POLL_INTERVAL) {
                Ok(chunk) => {
                    self.take_error_if_any()?;
                    return Ok(Some(chunk));
                }
                Err(RecvTimeoutError::Timeout) => self.take_error_if_any()?,
                Err(RecvTimeoutError::Disconnected) => {
                    self.take_error_if_any()?;
                    return Ok(None);
                }
            }
        }
    }

    pub fn stats(&self) -> PackedStreamStats {
        PackedStreamStats {
            decoded_entries: self.stats.decoded_entries.load(Ordering::Acquire),
            skipped_entries: self.stats.skipped_entries.load(Ordering::Acquire),
            produced_chunks: self.stats.produced_chunks.load(Ordering::Acquire),
            scanned_chunks: self.stats.scanned_chunks.load(Ordering::Acquire),
            chunk_queue_len: self.chunk_rx.as_ref().map_or(0, Receiver::len),
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

impl Drop for PackedEntryStream {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.chunk_rx.take();
        self.error_rx.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl EncodedBatch {
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

    fn push_entry(
        &mut self,
        entry: &TrainingDataEntry,
        feature_set: &FeatureSet,
    ) -> Result<(), PipelineError> {
        if self.size >= self.capacity {
            return Err(PipelineError::new("encoded batch overflow"));
        }

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
        Ok(())
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

pub fn encode_packed_bytes(
    feature_set: &FeatureSet,
    packed_entries: &[u8],
    batch_size: usize,
    encode_threads: usize,
) -> Result<EncodedBatch, PipelineError> {
    if batch_size == 0 {
        return Err(PipelineError::new("batch_size must be greater than zero"));
    }
    if packed_entries.len() % PACKED_ENTRY_BYTES != 0 {
        return Err(PipelineError::new(
            "packed entries length must be a multiple of 32 bytes",
        ));
    }
    let total_entries = packed_entries.len() / PACKED_ENTRY_BYTES;
    if total_entries != batch_size {
        return Err(PipelineError::new(format!(
            "packed bytes contain {total_entries} entries but batch_size is {batch_size}",
        )));
    }

    let max_active_features = feature_set.max_active_features();
    let threads = encode_threads.max(1);
    let encoded_rows = if threads == 1 {
        packed_entries
            .chunks_exact(PACKED_ENTRY_BYTES)
            .map(|packed| {
                let entry = unpack_training_entry(packed)?;
                let mut white = vec![-1; max_active_features];
                let mut black = vec![-1; max_active_features];
                let metadata =
                    encode_training_entry_indices_only(&entry, feature_set, &mut white, &mut black);
                Ok::<_, PipelineError>((metadata, white, black))
            })
            .collect::<Vec<_>>()
    } else {
        static ENCODE_POOL: std::sync::OnceLock<rayon::ThreadPool> =
            std::sync::OnceLock::new();
        let pool = ENCODE_POOL.get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("failed to build encode pool")
        });
        pool.install(|| {
            packed_entries
                .par_chunks_exact(PACKED_ENTRY_BYTES)
                .map(|packed| {
                    let entry = unpack_training_entry(packed)?;
                    let mut white = vec![-1; max_active_features];
                    let mut black = vec![-1; max_active_features];
                    let metadata = encode_training_entry_indices_only(
                        &entry,
                        feature_set,
                        &mut white,
                        &mut black,
                    );
                    Ok::<_, PipelineError>((metadata, white, black))
                })
                .collect::<Vec<_>>()
        })
    };

    let mut batch = EncodedBatch::new(batch_size, feature_set);
    for (row, row_data) in encoded_rows.into_iter().enumerate() {
        let (metadata, white, black) = row_data?;
        let start = row * max_active_features;
        let end = start + max_active_features;
        batch.white[start..end].copy_from_slice(&white);
        batch.black[start..end].copy_from_slice(&black);
        batch.write_metadata(row, metadata);
    }
    batch.size = batch_size;

    Ok(batch)
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
        if self.config.filtered && (is_capturing_move(entry) || is_in_check(entry)) {
            return true;
        }
        if self.config.wld_filtered {
            let keep_prob = score_result_prob(entry);
            if keep_prob <= 0.0 || (keep_prob < 1.0 && rng.gen::<f32>() >= keep_prob) {
                return true;
            }
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
            let current_weight = desired_piece_count_weight(self.config, piece_count as i32);
            let mut pass =
                self.piece_count_history_all_total * self.desired_piece_count_weights_total;

            for _ in 0..=32 {
                if current_weight > 0.0 {
                    let tmp = self.piece_count_history_all_total * current_weight
                        / (self.desired_piece_count_weights_total
                            * self.piece_count_history_all[piece_count]);
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
    config: PackedStreamConfig,
    chunk_tx: Sender<Vec<u8>>,
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
    let mut accumulation = Vec::with_capacity(config.chunk_entries);
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

            if accumulation.len() >= config.chunk_entries
                && publish_packed_chunk(
                    &chunk_tx,
                    &mut accumulation,
                    &error_tx,
                    &shutdown,
                    &mut counters,
                )
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
        if accumulation.len() >= config.chunk_entries
            && publish_packed_chunk(
                &chunk_tx,
                &mut accumulation,
                &error_tx,
                &shutdown,
                &mut counters,
            )
            .is_err()
        {
            flush_decoder_counters(&mut counters, &stats);
            return;
        }
    }

    if !accumulation.is_empty() {
        let _ = publish_packed_chunk(
            &chunk_tx,
            &mut accumulation,
            &error_tx,
            &shutdown,
            &mut counters,
        );
    }

    flush_decoder_counters(&mut counters, &stats);
}

fn publish_packed_chunk(
    chunk_tx: &Sender<Vec<u8>>,
    accumulation: &mut Vec<TrainingDataEntry>,
    error_tx: &Sender<String>,
    shutdown: &Arc<AtomicBool>,
    counters: &mut DecoderWorkerCounters,
) -> Result<(), ()> {
    if accumulation.is_empty() {
        return Ok(());
    }

    let mut entries = Vec::with_capacity(accumulation.len());
    std::mem::swap(accumulation, &mut entries);

    let mut chunk = Vec::with_capacity(entries.len() * PACKED_ENTRY_BYTES);
    for entry in entries {
        chunk.extend_from_slice(&pack_training_entry(&entry));
    }

    loop {
        match chunk_tx.try_send(chunk) {
            Ok(()) => {
                counters.produced_chunks += 1;
                return Ok(());
            }
            Err(TrySendError::Full(c)) => {
                chunk = c;
                if shutdown.load(Ordering::Relaxed) {
                    return Err(());
                }
                thread::sleep(POLL_INTERVAL);
            }
            Err(TrySendError::Disconnected(_)) => {
                report_error(
                    error_tx,
                    "packed chunk channel closed unexpectedly".to_string(),
                );
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
    if counters.produced_chunks > 0 {
        stats
            .produced_chunks
            .fetch_add(counters.produced_chunks, Ordering::AcqRel);
        counters.produced_chunks = 0;
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

fn pack_training_entry(entry: &TrainingDataEntry) -> [u8; PACKED_ENTRY_BYTES] {
    let mut data = [0u8; PACKED_ENTRY_BYTES];

    let mut offset = 0;
    write_compressed_position(&entry.pos, &mut data[offset..offset + 24]);
    offset += 24;

    let packed_move = compress_move(entry.mv);
    data[offset] = (packed_move >> 8) as u8;
    offset += 1;
    data[offset] = packed_move as u8;
    offset += 1;

    let encoded_score = signed_to_unsigned(entry.score);
    data[offset] = (encoded_score >> 8) as u8;
    offset += 1;
    data[offset] = encoded_score as u8;
    offset += 1;

    let pr = entry.ply | (signed_to_unsigned(entry.result) << 14);
    data[offset] = (pr >> 8) as u8;
    offset += 1;
    data[offset] = pr as u8;
    offset += 1;

    data[offset] = (entry.pos.rule50_counter() >> 8) as u8;
    offset += 1;
    data[offset] = entry.pos.rule50_counter() as u8;

    data
}

fn unpack_training_entry(data: &[u8]) -> Result<TrainingDataEntry, PipelineError> {
    if data.len() != PACKED_ENTRY_BYTES {
        return Err(PipelineError::new("packed entry must be exactly 32 bytes"));
    }

    let mut offset = 0;
    let mut pos = read_compressed_position(&data[offset..offset + 24]);
    offset += 24;

    let packed_move = ((data[offset] as u16) << 8) | data[offset + 1] as u16;
    let mv = decompress_move(packed_move);
    offset += 2;

    let score_u = ((data[offset] as u16) << 8) | data[offset + 1] as u16;
    let score = unsigned_to_signed(score_u);
    offset += 2;

    let pr = ((data[offset] as u16) << 8) | data[offset + 1] as u16;
    let ply = pr & 0x3FFF;
    let result = unsigned_to_signed(pr >> 14);
    offset += 2;

    let rule50 = ((data[offset] as u16) << 8) | data[offset + 1] as u16;

    pos.set_ply(ply);
    pos.set_rule50_counter(rule50);

    Ok(TrainingDataEntry {
        pos,
        mv,
        score,
        ply,
        result,
    })
}

fn write_compressed_position(pos: &Position, out: &mut [u8]) {
    debug_assert!(out.len() >= 24);
    let occupied = pos.occupied().bits();
    out[0] = (occupied >> 56) as u8;
    out[1] = ((occupied >> 48) & 0xFF) as u8;
    out[2] = ((occupied >> 40) & 0xFF) as u8;
    out[3] = ((occupied >> 32) & 0xFF) as u8;
    out[4] = ((occupied >> 24) & 0xFF) as u8;
    out[5] = ((occupied >> 16) & 0xFF) as u8;
    out[6] = ((occupied >> 8) & 0xFF) as u8;
    out[7] = (occupied & 0xFF) as u8;

    let mut packed_state = [0u8; 16];
    let mut idx = 0usize;
    for (nibble_idx, sq) in pos.occupied().iter().enumerate() {
        let nibble = pack_position_nibble(pos, sq);
        if nibble_idx % 2 == 0 {
            packed_state[idx] = nibble;
        } else {
            packed_state[idx] |= nibble << 4;
            idx += 1;
        }
    }

    out[8..24].copy_from_slice(&packed_state);
}

fn read_compressed_position(data: &[u8]) -> Position {
    debug_assert!(data.len() >= 24);
    let occupied = ((data[0] as u64) << 56)
        | ((data[1] as u64) << 48)
        | ((data[2] as u64) << 40)
        | ((data[3] as u64) << 32)
        | ((data[4] as u64) << 24)
        | ((data[5] as u64) << 16)
        | ((data[6] as u64) << 8)
        | (data[7] as u64);

    let mut packed_state = [0u8; 16];
    packed_state.copy_from_slice(&data[8..24]);

    let mut pos = Position::empty();
    pos.set_castling_rights(CastlingRights::NONE);

    let mut squares_iter = Bitboard::new(occupied).iter();
    for chunk in packed_state {
        if let Some(sq) = squares_iter.next() {
            unpack_position_nibble(&mut pos, sq, chunk & 0xF);
        } else {
            break;
        }

        if let Some(sq) = squares_iter.next() {
            unpack_position_nibble(&mut pos, sq, chunk >> 4);
        } else {
            break;
        }
    }

    pos
}

fn pack_position_nibble(pos: &Position, sq: Square) -> u8 {
    let piece = pos.piece_at(sq);

    if piece.piece_type() == PieceType::Pawn {
        let ep_sq = pos.ep_square();
        if ep_sq != Square::NONE
            && ((piece.color() == Color::White
                && sq.rank() == Rank::FOURTH
                && ep_sq == sq + FlatSquareOffset::new(0, -1))
                || (piece.color() == Color::Black
                    && sq.rank() == Rank::FIFTH
                    && ep_sq == sq + FlatSquareOffset::new(0, 1)))
        {
            return 12;
        }
    }

    if piece == Piece::WHITE_ROOK
        && ((sq == Square::A1
            && pos
                .castling_rights()
                .contains(CastlingRights::WHITE_QUEEN_SIDE))
            || (sq == Square::H1
                && pos
                    .castling_rights()
                    .contains(CastlingRights::WHITE_KING_SIDE)))
    {
        return 13;
    }

    if piece == Piece::BLACK_ROOK
        && ((sq == Square::A8
            && pos
                .castling_rights()
                .contains(CastlingRights::BLACK_QUEEN_SIDE))
            || (sq == Square::H8
                && pos
                    .castling_rights()
                    .contains(CastlingRights::BLACK_KING_SIDE)))
    {
        return 14;
    }

    if piece == Piece::BLACK_KING && pos.side_to_move() == Color::Black {
        return 15;
    }

    piece.id() as u8
}

fn unpack_position_nibble(pos: &mut Position, sq: Square, nibble: u8) {
    match nibble {
        0..=11 => pos.place(Piece::from_id(nibble as i32), sq),
        12 => {
            let rank = sq.rank();
            if rank == Rank::FOURTH {
                pos.place(Piece::WHITE_PAWN, sq);
                pos.set_ep_square_unchecked(sq + FlatSquareOffset::new(0, -1));
            } else {
                pos.place(Piece::BLACK_PAWN, sq);
                pos.set_ep_square_unchecked(sq + FlatSquareOffset::new(0, 1));
            }
        }
        13 => {
            pos.place(Piece::WHITE_ROOK, sq);
            if sq == Square::A1 {
                pos.add_castling_rights(CastlingRights::WHITE_QUEEN_SIDE);
            } else {
                pos.add_castling_rights(CastlingRights::WHITE_KING_SIDE);
            }
        }
        14 => {
            pos.place(Piece::BLACK_ROOK, sq);
            if sq == Square::A8 {
                pos.add_castling_rights(CastlingRights::BLACK_QUEEN_SIDE);
            } else {
                pos.add_castling_rights(CastlingRights::BLACK_KING_SIDE);
            }
        }
        15 => {
            pos.place(Piece::BLACK_KING, sq);
            pos.set_side_to_move(Color::Black);
        }
        _ => unreachable!(),
    }
}

fn compress_move(mv: Move) -> u16 {
    if mv.from() == mv.to() {
        return 0;
    }

    let mut packed = ((mv.mtype() as u16) << 14)
        | ((mv.from().index() as u16) << 8)
        | ((mv.to().index() as u16) << 2);
    if mv.mtype() == MoveType::Promotion {
        packed |= (mv.promoted_piece().piece_type() as u16) - (PieceType::Knight as u16);
    }
    packed
}

fn decompress_move(packed: u16) -> Move {
    if packed == 0 {
        return Move::null();
    }

    let move_type = MoveType::from_ordinal((packed >> 14) as u8);
    let from = Square::new(((packed >> 8) & 0b11_1111) as u32);
    let to = Square::new(((packed >> 2) & 0b11_1111) as u32);
    let promoted_piece = if move_type == MoveType::Promotion {
        let color = if to.rank() == Rank::FIRST {
            Color::Black
        } else {
            Color::White
        };
        let piece_type = PieceType::from_ordinal((packed as u8 & 0b11) + PieceType::Knight as u8);
        Piece::new(piece_type, color)
    } else {
        Piece::none()
    };

    Move::new(from, to, move_type, promoted_piece)
}

#[inline(always)]
fn unsigned_to_signed(value: u16) -> i16 {
    let mut v = value.rotate_right(1);
    if v & 0x8000 != 0 {
        v ^= 0x7FFF;
    }
    v as i16
}

#[inline(always)]
fn signed_to_unsigned(value: i16) -> u16 {
    let mut v = i16::cast_unsigned(value);
    if v & 0x8000 != 0 {
        v ^= 0x7FFF;
    }
    v.rotate_left(1)
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

    if to == Square::NONE || from == Square::NONE {
        return false;
    }

    let captured = entry.pos.piece_at(to);
    let moving = entry.pos.piece_at(from);
    captured != Piece::none() && captured.color() != moving.color()
}

fn is_in_check(entry: &TrainingDataEntry) -> bool {
    entry.pos.is_checked(entry.pos.side_to_move())
}

fn simple_eval(pos: &Position) -> i32 {
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

fn material_count(pos: &Position, color: Color, piece_type: PieceType) -> i32 {
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

pub fn default_thread_counts(
    total_threads: usize,
    task_count: usize,
    skip_heavy: bool,
) -> (usize, usize) {
    if total_threads <= 1 {
        return (1, 1);
    }

    let available_tasks = task_count.max(1);

    let (decode_threads, encode_threads) = if skip_heavy {
        let decode_threads = ((total_threads * 3) / 4).max(1);
        let encode_threads = total_threads.saturating_sub(decode_threads).max(1);
        (decode_threads, encode_threads)
    } else {
        let decode_threads = (total_threads / 4).max(1).min(8);
        let encode_threads = total_threads.saturating_sub(decode_threads).max(1);
        (decode_threads, encode_threads)
    };

    (decode_threads.min(available_tasks).max(1), encode_threads)
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

    #[test]
    fn packed_entry_round_trip_preserves_training_data() {
        let entry = make_entry(
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            Move::castle(Square::E1, Square::H1),
            -44,
            8,
            1,
        );

        let packed = pack_training_entry(&entry);
        let unpacked = unpack_training_entry(&packed).unwrap();
        assert_eq!(entry, unpacked);
        assert_eq!(packed, pack_training_entry(&unpacked));
        assert_eq!(entry.pos.ply(), unpacked.pos.ply());
        assert_eq!(entry.pos.rule50_counter(), unpacked.pos.rule50_counter());
    }

    #[test]
    fn packed_entry_round_trip_preserves_special_position_markers() {
        let entries = vec![
            make_entry(
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                Move::null(),
                5,
                6,
                0,
            ),
            make_entry(
                "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
                Move::normal(sq("d5"), sq("e4")),
                -9,
                11,
                -1,
            ),
            make_entry(
                "4k3/8/8/8/8/8/8/4K3 b - - 0 1",
                Move::normal(sq("e8"), sq("e7")),
                2,
                3,
                1,
            ),
        ];

        for entry in entries {
            let packed = pack_training_entry(&entry);
            let unpacked = unpack_training_entry(&packed).unwrap();
            assert_eq!(entry, unpacked);
            assert_eq!(packed, pack_training_entry(&unpacked));
        }
    }

    #[test]
    fn packed_entry_bytes_match_sfbinpack_writer_for_single_entry() {
        let entries = vec![
            make_entry(
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                Move::castle(Square::E1, Square::H1),
                50,
                6,
                1,
            ),
            make_entry(
                "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
                Move::normal(sq("d5"), sq("e4")),
                -9,
                11,
                -1,
            ),
            make_entry(
                "1r3rk1/p2qnpb1/6pp/P1p1p3/3nN3/2QP2P1/R3PPBP/2B2RK1 b - - 2 20",
                Move::new(
                    Square::new(61),
                    Square::new(58),
                    MoveType::Normal,
                    Piece::none(),
                ),
                -127,
                39,
                0,
            ),
        ];

        for entry in entries {
            let expected = extract_single_stem_from_sfbinpack_file(&entry);
            let actual = pack_training_entry(&entry);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn packed_entry_bytes_match_sfbinpack_writer_for_many_entries() {
        for entry in repeated_entries(64) {
            let expected = extract_single_stem_from_sfbinpack_file(&entry);
            let actual = pack_training_entry(&entry);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn packed_entry_bytes_are_accepted_by_sfbinpack_reader() {
        let entries = vec![
            make_entry(
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                Move::null(),
                5,
                6,
                0,
            ),
            make_entry(
                "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
                Move::normal(sq("d5"), sq("e4")),
                -9,
                11,
                -1,
            ),
            make_entry(
                "1r3rk1/p2qnpb1/6pp/P1p1p3/3nN3/2QP2P1/R3PPBP/2B2RK1 b - - 2 20",
                Move::new(
                    Square::new(61),
                    Square::new(58),
                    MoveType::Normal,
                    Piece::none(),
                ),
                -127,
                39,
                0,
            ),
        ];

        for entry in entries {
            let packed = pack_training_entry(&entry);
            let blob = build_single_entry_binpack_blob(&packed);
            let mut reader = CompressedTrainingDataEntryReader::new(Cursor::new(blob)).unwrap();
            assert!(reader.has_next());
            let decoded = reader.next();
            assert_eq!(decoded, entry);
            assert!(!reader.has_next());
        }
    }

    #[test]
    fn packed_entry_known_fixture_matches_sfbinpack_format() {
        let packed = [
            98, 121, 192, 21, 24, 76, 241, 100, 100, 106, 0, 4, 8, 48, 2, 17, 17, 145, 19, 117,
            247, 0, 0, 0, 61, 232, 0, 253, 0, 39, 0, 2,
        ];

        let expected = TrainingDataEntry {
            pos: Position::from_fen(
                "1r3rk1/p2qnpb1/6pp/P1p1p3/3nN3/2QP2P1/R3PPBP/2B2RK1 b - - 2 20",
            )
            .unwrap(),
            mv: Move::new(
                Square::new(61),
                Square::new(58),
                MoveType::Normal,
                Piece::none(),
            ),
            score: -127,
            ply: 39,
            result: 0,
        };

        let unpacked = unpack_training_entry(&packed).unwrap();
        assert_eq!(unpacked, expected);
        assert_eq!(pack_training_entry(&expected), packed);
    }

    #[test]
    fn stream_produces_packed_chunks() {
        let entries = repeated_entries(33);
        let file = write_entries(&entries);

        let mut config = PackedStreamConfig::new(vec![file.clone()]);
        config.total_threads = 2;
        config.decode_threads = Some(1);
        config.chunk_entries = 8;
        config.shuffle_buffer_entries = 0;

        let stream = PackedEntryStream::new(config).unwrap();
        let mut total_entries = 0usize;
        while let Some(chunk) = stream.next_chunk().unwrap() {
            assert_eq!(chunk.len() % PACKED_ENTRY_BYTES, 0);
            total_entries += chunk.len() / PACKED_ENTRY_BYTES;
        }

        assert_eq!(total_entries, entries.len());
        let _ = fs::remove_file(file);
    }

    #[test]
    fn encode_packed_chunks_builds_exact_batch() {
        let feature_set = FeatureSet::from_str("HalfKAv2_hm").unwrap();
        let entries = repeated_entries(16);

        let mut chunk0 = Vec::new();
        let mut chunk1 = Vec::new();
        for entry in &entries[..8] {
            chunk0.extend_from_slice(&pack_training_entry(entry));
        }
        for entry in &entries[8..] {
            chunk1.extend_from_slice(&pack_training_entry(entry));
        }

        let batch = encode_packed_chunks(&feature_set, &[chunk0, chunk1], 16, 1).unwrap();
        assert_eq!(batch.len(), 16);
        assert_eq!(
            batch.white_flat_slice().len(),
            16 * feature_set.max_active_features()
        );
        assert_eq!(
            batch.black_flat_slice().len(),
            16 * feature_set.max_active_features()
        );
    }

    #[test]
    fn packed_stream_round_trip_matches_entries_without_shuffle() {
        let entries = repeated_entries(97);
        let file = write_entries(&entries);

        let mut config = PackedStreamConfig::new(vec![file.clone()]);
        config.total_threads = 2;
        config.decode_threads = Some(1);
        config.chunk_entries = 11;
        config.shuffle_buffer_entries = 0;

        let stream = PackedEntryStream::new(config).unwrap();
        let mut actual = Vec::new();
        while let Some(chunk) = stream.next_chunk().unwrap() {
            for packed in chunk.chunks_exact(PACKED_ENTRY_BYTES) {
                actual.push(unpack_training_entry(packed).unwrap());
            }
        }

        assert_eq!(actual, entries);
        let _ = fs::remove_file(file);
    }

    #[test]
    fn encode_packed_chunks_matches_direct_encoding() {
        let entries = repeated_entries(32);

        let chunks = entries.chunks(8).map(pack_entry_group).collect::<Vec<_>>();

        for feature_name in ["HalfKAv2_hm", "Full_Threats", "Full_Threats+HalfKAv2_hm"] {
            let feature_set = FeatureSet::from_str(feature_name).unwrap();
            let actual = encode_packed_chunks(&feature_set, &chunks, entries.len(), 1).unwrap();
            let expected = encode_entries_direct(&feature_set, &entries);
            assert_batches_equal(&actual, &expected);
        }
    }

    #[test]
    fn split_pipeline_first_batch_matches_direct_encoding() {
        let feature_set = FeatureSet::from_str("Full_Threats+HalfKAv2_hm").unwrap();
        let entries = repeated_entries(96);
        let file = write_entries(&entries);

        let chunk_entries = 8;
        let batch_size = 32;
        let chunks_per_batch = batch_size / chunk_entries;

        let mut config = PackedStreamConfig::new(vec![file.clone()]);
        config.total_threads = 2;
        config.decode_threads = Some(1);
        config.chunk_entries = chunk_entries;
        config.shuffle_buffer_entries = 0;

        let stream = PackedEntryStream::new(config).unwrap();
        let mut first_batch_chunks = Vec::new();
        for _ in 0..chunks_per_batch {
            first_batch_chunks.push(stream.next_chunk().unwrap().unwrap());
        }

        let actual =
            encode_packed_chunks(&feature_set, &first_batch_chunks, batch_size, 1).unwrap();
        let expected = encode_entries_direct(&feature_set, &entries[..batch_size]);
        assert_batches_equal(&actual, &expected);
        let _ = fs::remove_file(file);
    }

    #[test]
    fn split_pipeline_all_full_batches_match_direct_encoding() {
        let entries = repeated_entries(130);
        let file = write_entries(&entries);

        let chunk_entries = 8;
        let batch_size = 32;
        let chunks_per_batch = batch_size / chunk_entries;

        let mut config = PackedStreamConfig::new(vec![file.clone()]);
        config.total_threads = 2;
        config.decode_threads = Some(1);
        config.chunk_entries = chunk_entries;
        config.shuffle_buffer_entries = 0;

        let mut chunk_groups = Vec::new();
        let stream = PackedEntryStream::new(config).unwrap();
        let full_chunk_bytes = chunk_entries * PACKED_ENTRY_BYTES;

        'outer: loop {
            let mut chunk_group = Vec::new();
            for _ in 0..chunks_per_batch {
                match stream.next_chunk().unwrap() {
                    Some(chunk) if chunk.len() == full_chunk_bytes => chunk_group.push(chunk),
                    Some(_) | None => break 'outer,
                }
            }

            if chunk_group.is_empty() {
                break;
            }
            chunk_groups.push(chunk_group);
        }

        assert_eq!(chunk_groups.len(), entries.len() / batch_size);

        for feature_name in ["HalfKAv2_hm", "Full_Threats", "Full_Threats+HalfKAv2_hm"] {
            let feature_set = FeatureSet::from_str(feature_name).unwrap();
            for (batch_index, chunk_group) in chunk_groups.iter().enumerate() {
                let actual =
                    encode_packed_chunks(&feature_set, chunk_group, batch_size, 1).unwrap();
                let start = batch_index * batch_size;
                let end = start + batch_size;
                let expected = encode_entries_direct(&feature_set, &entries[start..end]);
                assert_batches_equal(&actual, &expected);
            }
        }

        let _ = fs::remove_file(file);
    }

    #[test]
    fn skip_filter_removes_known_entries_from_stream() {
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

        let mut config = PackedStreamConfig::new(vec![file.clone()]);
        config.total_threads = 2;
        config.decode_threads = Some(1);
        config.chunk_entries = 8;
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

        let stream = PackedEntryStream::new(config).unwrap();
        let kept = collect_stream_entries(&stream);
        let kept_keys = kept
            .iter()
            .map(entry_key)
            .collect::<std::collections::BTreeSet<_>>();
        let skipped_keys = entries[..3]
            .iter()
            .map(entry_key)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(kept_keys.is_disjoint(&skipped_keys));

        let _ = fs::remove_file(file);
    }

    #[test]
    fn ddp_sharding_produces_disjoint_packed_streams() {
        let files = vec![
            write_entries(&repeated_entries_with_offset(24, 0)),
            write_entries(&repeated_entries_with_offset(24, 24)),
        ];

        let mut rank0 = PackedStreamConfig::new(files.clone());
        rank0.total_threads = 2;
        rank0.decode_threads = Some(1);
        rank0.chunk_entries = 8;
        rank0.shuffle_buffer_entries = 0;
        rank0.rank = 0;
        rank0.world_size = 2;

        let mut rank1 = PackedStreamConfig::new(files.clone());
        rank1.total_threads = 2;
        rank1.decode_threads = Some(1);
        rank1.chunk_entries = 8;
        rank1.shuffle_buffer_entries = 0;
        rank1.rank = 1;
        rank1.world_size = 2;

        let rank0_entries = collect_stream_entries(&PackedEntryStream::new(rank0).unwrap());
        let rank1_entries = collect_stream_entries(&PackedEntryStream::new(rank1).unwrap());

        let rank0_keys = rank0_entries
            .iter()
            .map(entry_key)
            .collect::<std::collections::BTreeSet<_>>();
        let rank1_keys = rank1_entries
            .iter()
            .map(entry_key)
            .collect::<std::collections::BTreeSet<_>>();

        assert!(rank0_keys.is_disjoint(&rank1_keys));
        assert_eq!(rank0_entries.len() + rank1_entries.len(), 48);

        for file in files {
            let _ = fs::remove_file(file);
        }
    }

    fn encode_entries_direct(
        feature_set: &FeatureSet,
        entries: &[TrainingDataEntry],
    ) -> EncodedBatch {
        let mut batch = EncodedBatch::new(entries.len(), feature_set);
        for entry in entries {
            batch.push_entry(entry, feature_set).unwrap();
        }
        batch
    }

    fn collect_stream_entries(stream: &PackedEntryStream) -> Vec<TrainingDataEntry> {
        let mut entries = Vec::new();
        while let Some(chunk) = stream.next_chunk().unwrap() {
            for packed in chunk.chunks_exact(PACKED_ENTRY_BYTES) {
                entries.push(unpack_training_entry(packed).unwrap());
            }
        }
        entries
    }

    fn pack_entry_group(entries: &[TrainingDataEntry]) -> Vec<u8> {
        let mut chunk = Vec::with_capacity(entries.len() * PACKED_ENTRY_BYTES);
        for entry in entries {
            chunk.extend_from_slice(&pack_training_entry(entry));
        }
        chunk
    }

    fn assert_batches_equal(actual: &EncodedBatch, expected: &EncodedBatch) {
        assert_eq!(actual.num_inputs(), expected.num_inputs());
        assert_eq!(actual.max_active_features(), expected.max_active_features());
        assert_eq!(actual.len(), expected.len());
        assert_eq!(
            actual.num_active_white_features(),
            expected.num_active_white_features()
        );
        assert_eq!(
            actual.num_active_black_features(),
            expected.num_active_black_features()
        );
        assert_eq!(actual.is_white_slice(), expected.is_white_slice());
        assert_eq!(actual.outcome_slice(), expected.outcome_slice());
        assert_eq!(actual.score_slice(), expected.score_slice());
        assert_eq!(actual.white_flat_slice(), expected.white_flat_slice());
        assert_eq!(actual.black_flat_slice(), expected.black_flat_slice());
        assert_eq!(actual.psqt_indices_slice(), expected.psqt_indices_slice());
        assert_eq!(
            actual.layer_stack_indices_slice(),
            expected.layer_stack_indices_slice()
        );
    }

    fn write_entries(entries: &[TrainingDataEntry]) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "stockfish_trainer_pipeline_test_{}_{}.binpack",
            std::process::id(),
            random_seed()
        ));
        let writer_file = std::fs::File::create(&path).unwrap();
        let mut writer = CompressedTrainingDataEntryWriter::new(writer_file).unwrap();
        for entry in entries {
            writer.write_entry(entry).unwrap();
        }
        writer.flush_and_end();
        path
    }

    fn extract_single_stem_from_sfbinpack_file(
        entry: &TrainingDataEntry,
    ) -> [u8; PACKED_ENTRY_BYTES] {
        let file = write_entries(&[*entry]);
        let bytes = fs::read(&file).unwrap();
        let _ = fs::remove_file(file);

        assert!(bytes.len() >= BINPACK_HEADER_SIZE + PACKED_ENTRY_BYTES + 2);
        assert_eq!(&bytes[0..4], b"BINP");
        let chunk_size = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        assert_eq!(bytes.len(), BINPACK_HEADER_SIZE + chunk_size);
        assert!(chunk_size >= PACKED_ENTRY_BYTES + 2);

        let mut packed = [0u8; PACKED_ENTRY_BYTES];
        packed
            .copy_from_slice(&bytes[BINPACK_HEADER_SIZE..BINPACK_HEADER_SIZE + PACKED_ENTRY_BYTES]);
        packed
    }

    fn build_single_entry_binpack_blob(packed: &[u8; PACKED_ENTRY_BYTES]) -> Vec<u8> {
        let payload_size = PACKED_ENTRY_BYTES + 2;
        let mut blob = Vec::with_capacity(BINPACK_HEADER_SIZE + payload_size);
        blob.extend_from_slice(b"BINP");
        blob.extend_from_slice(&(payload_size as u32).to_le_bytes());
        blob.extend_from_slice(packed);
        blob.extend_from_slice(&0_u16.to_be_bytes());
        blob
    }

    fn repeated_entries(count: usize) -> Vec<TrainingDataEntry> {
        let base = sample_entries();
        (0..count)
            .map(|index| {
                let mut entry = base[index % base.len()];
                entry.score = ((index as i32 * 37) % 400 - 200) as i16;
                entry.ply = (10 + index % 80) as u16;
                entry.result = match index % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                };
                entry.pos.set_ply(entry.ply);
                entry
            })
            .collect()
    }

    fn repeated_entries_with_offset(count: usize, offset: usize) -> Vec<TrainingDataEntry> {
        let base = sample_entries();
        (0..count)
            .map(|index| {
                let mut entry = base[(index + offset) % base.len()];
                entry.score = (((index + offset) as i32 * 37) % 400 - 200) as i16;
                entry.ply = (10 + (index + offset) % 80) as u16;
                entry.result = match (index + offset) % 3 {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                };
                entry.pos.set_ply(entry.ply);
                entry
            })
            .collect()
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

    fn entry_key(entry: &TrainingDataEntry) -> String {
        format!("{:?}", entry)
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
