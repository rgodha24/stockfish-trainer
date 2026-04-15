use std::str::FromStr;
use std::sync::OnceLock;

use sfbinpack::chess::{
    attacks, bitboard::Bitboard, color::Color, coords::Square, piece::Piece, piecetype::PieceType,
    position::Position,
};
use sfbinpack::TrainingDataEntry;

pub const HALFKA_INPUTS: usize = 24_576;
pub const HALFKA_MAX_ACTIVE_FEATURES: usize = 32;
pub const FULL_THREATS_INPUTS: usize = 60_720;
pub const FULL_THREATS_MAX_ACTIVE_FEATURES: usize = 128;

const NUM_SQ: usize = 64;
const NUM_PT: usize = 12;
const NUM_PLANES: usize = NUM_SQ * NUM_PT;
const A_FILE: u64 = 0x0101_0101_0101_0101;
const H_FILE: u64 = 0x8080_8080_8080_8080;

const KING_BUCKETS: [i32; 64] = [
    -1, -1, -1, -1, 31, 30, 29, 28, -1, -1, -1, -1, 27, 26, 25, 24, -1, -1, -1, -1, 23, 22, 21, 20,
    -1, -1, -1, -1, 19, 18, 17, 16, -1, -1, -1, -1, 15, 14, 13, 12, -1, -1, -1, -1, 11, 10, 9, 8,
    -1, -1, -1, -1, 7, 6, 5, 4, -1, -1, -1, -1, 3, 2, 1, 0,
];

const NUM_VALID_TARGETS: [i32; 12] = [6, 6, 10, 10, 8, 8, 8, 8, 10, 10, 0, 0];
const THREAT_MAP: [[i32; 6]; 6] = [
    [0, 1, -1, 2, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [-1, -1, -1, -1, -1, -1],
];
const PIECE_TABLE: [Piece; 12] = [
    Piece::WHITE_PAWN,
    Piece::BLACK_PAWN,
    Piece::WHITE_KNIGHT,
    Piece::BLACK_KNIGHT,
    Piece::WHITE_BISHOP,
    Piece::BLACK_BISHOP,
    Piece::WHITE_ROOK,
    Piece::BLACK_ROOK,
    Piece::WHITE_QUEEN,
    Piece::BLACK_QUEEN,
    Piece::WHITE_KING,
    Piece::BLACK_KING,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureComponent {
    HalfKAv2Hm,
    FullThreats,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureSet {
    components: Vec<FeatureComponent>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureSetError {
    message: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseRow {
    pub num_inputs: usize,
    pub max_active_features: usize,
    pub is_white: f32,
    pub outcome: f32,
    pub score: f32,
    pub white_count: usize,
    pub black_count: usize,
    pub white: Vec<i32>,
    pub black: Vec<i32>,
    pub white_values: Vec<f32>,
    pub black_values: Vec<f32>,
    pub psqt_indices: i64,
    pub layer_stack_indices: i64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowMetadata {
    pub is_white: f32,
    pub outcome: f32,
    pub score: f32,
    pub white_count: usize,
    pub black_count: usize,
    pub psqt_indices: i64,
    pub layer_stack_indices: i64,
}

#[derive(Clone)]
struct ThreatFeatureCalculation {
    table: [[i32; 66]; 12],
    #[allow(dead_code)]
    total_features: i32,
}

impl FeatureComponent {
    pub const fn name(self) -> &'static str {
        match self {
            Self::HalfKAv2Hm => "HalfKAv2_hm",
            Self::FullThreats => "Full_Threats",
        }
    }

    pub const fn inputs(self) -> usize {
        match self {
            Self::HalfKAv2Hm => HALFKA_INPUTS,
            Self::FullThreats => FULL_THREATS_INPUTS,
        }
    }

    pub const fn max_active_features(self) -> usize {
        match self {
            Self::HalfKAv2Hm => HALFKA_MAX_ACTIVE_FEATURES,
            Self::FullThreats => FULL_THREATS_MAX_ACTIVE_FEATURES,
        }
    }

    fn fill_features_sparse(
        self,
        pos: &Position,
        color: Color,
        features: &mut [i32],
        values: &mut [f32],
    ) -> usize {
        match self {
            Self::HalfKAv2Hm => fill_halfka_features_sparse(pos, color, features, values),
            Self::FullThreats => fill_full_threats_sparse(pos, color, features, values),
        }
    }

    fn fill_feature_indices_sparse(
        self,
        pos: &Position,
        color: Color,
        features: &mut [i32],
    ) -> usize {
        match self {
            Self::HalfKAv2Hm => fill_halfka_feature_indices_sparse(pos, color, features),
            Self::FullThreats => fill_full_threat_feature_indices_sparse(pos, color, features),
        }
    }
}

impl FeatureSet {
    pub fn halfka() -> Self {
        Self {
            components: vec![FeatureComponent::HalfKAv2Hm],
        }
    }

    pub fn full_threats() -> Self {
        Self {
            components: vec![FeatureComponent::FullThreats],
        }
    }

    pub fn inputs(&self) -> usize {
        self.components
            .iter()
            .map(|component| component.inputs())
            .sum()
    }

    pub fn max_active_features(&self) -> usize {
        self.components
            .iter()
            .map(|component| component.max_active_features())
            .sum()
    }

    pub fn components(&self) -> &[FeatureComponent] {
        &self.components
    }

    pub fn fill_features_sparse(
        &self,
        pos: &Position,
        color: Color,
        features: &mut [i32],
        values: &mut [f32],
    ) -> usize {
        let mut total_written = 0usize;
        let mut input_offset = 0i32;

        for component in &self.components {
            let component_max = component.max_active_features();
            let written = component.fill_features_sparse(
                pos,
                color,
                &mut features[total_written..total_written + component_max],
                &mut values[total_written..total_written + component_max],
            );

            if input_offset != 0 {
                for feature in &mut features[total_written..total_written + written] {
                    *feature += input_offset;
                }
            }

            input_offset += component.inputs() as i32;
            total_written += written;
        }

        total_written
    }

    pub fn fill_feature_indices_sparse(
        &self,
        pos: &Position,
        color: Color,
        features: &mut [i32],
    ) -> usize {
        let mut total_written = 0usize;
        let mut input_offset = 0i32;

        for component in &self.components {
            let component_max = component.max_active_features();
            let written = component.fill_feature_indices_sparse(
                pos,
                color,
                &mut features[total_written..total_written + component_max],
            );

            if input_offset != 0 {
                for feature in &mut features[total_written..total_written + written] {
                    *feature += input_offset;
                }
            }

            input_offset += component.inputs() as i32;
            total_written += written;
        }

        total_written
    }
}

impl FromStr for FeatureSet {
    type Err = FeatureSetError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut components = Vec::new();

        for component in value.split('+') {
            let component = match component {
                "HalfKAv2_hm" => FeatureComponent::HalfKAv2Hm,
                "Full_Threats" => FeatureComponent::FullThreats,
                other => {
                    return Err(FeatureSetError {
                        message: format!("unknown feature component: {other}"),
                    });
                }
            };
            components.push(component);
        }

        if components.is_empty() {
            return Err(FeatureSetError {
                message: "feature set cannot be empty".to_string(),
            });
        }

        Ok(Self { components })
    }
}

impl std::fmt::Display for FeatureSetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FeatureSetError {}

pub fn build_sparse_row(pos: &Position, score: i32, result: i32) -> SparseRow {
    build_sparse_row_for_feature_set(pos, score, result, &FeatureSet::halfka())
}

pub fn encode_training_entry(
    entry: &TrainingDataEntry,
    feature_set: &FeatureSet,
    white: &mut [i32],
    white_values: &mut [f32],
    black: &mut [i32],
    black_values: &mut [f32],
) -> RowMetadata {
    fill_row_from_position(
        &entry.pos,
        entry.score as i32,
        entry.result as i32,
        feature_set,
        white,
        white_values,
        black,
        black_values,
    )
}

pub fn encode_training_entry_indices_only(
    entry: &TrainingDataEntry,
    feature_set: &FeatureSet,
    white: &mut [i32],
    black: &mut [i32],
) -> RowMetadata {
    fill_row_from_position_without_values(
        &entry.pos,
        entry.score as i32,
        entry.result as i32,
        feature_set,
        white,
        black,
    )
}

pub fn build_sparse_row_for_feature_set(
    pos: &Position,
    score: i32,
    result: i32,
    feature_set: &FeatureSet,
) -> SparseRow {
    let max_active_features = feature_set.max_active_features();
    let mut row = SparseRow {
        num_inputs: feature_set.inputs(),
        max_active_features,
        is_white: 0.0,
        outcome: 0.0,
        score: 0.0,
        white_count: 0,
        black_count: 0,
        white: vec![-1; max_active_features],
        black: vec![-1; max_active_features],
        white_values: vec![0.0; max_active_features],
        black_values: vec![0.0; max_active_features],
        psqt_indices: 0,
        layer_stack_indices: 0,
    };

    let metadata = fill_row_from_position(
        pos,
        score,
        result,
        feature_set,
        &mut row.white,
        &mut row.white_values,
        &mut row.black,
        &mut row.black_values,
    );

    row.is_white = metadata.is_white;
    row.outcome = metadata.outcome;
    row.score = metadata.score;
    row.white_count = metadata.white_count;
    row.black_count = metadata.black_count;
    row.psqt_indices = metadata.psqt_indices;
    row.layer_stack_indices = metadata.layer_stack_indices;

    row
}

pub fn orient_flip(color: Color, sq: Square, ksq: Square) -> Square {
    Square::new(sq.index() ^ orient_mask(color, ksq))
}

fn fill_halfka_features_sparse(
    pos: &Position,
    color: Color,
    features: &mut [i32],
    values: &mut [f32],
) -> usize {
    let mut count = 0usize;
    let mut occupied = pos.occupied();
    let ksq = pos.king_sq(color);

    while occupied.bits() != 0 {
        let sq = occupied.pop();
        let piece = pos.piece_at(sq);
        values[count] = 1.0;
        features[count] = halfka_feature_index(color, ksq, sq, piece);
        count += 1;
    }

    count
}

fn fill_halfka_feature_indices_sparse(pos: &Position, color: Color, features: &mut [i32]) -> usize {
    let mut count = 0usize;
    let mut occupied = pos.occupied();
    let ksq = pos.king_sq(color);

    while occupied.bits() != 0 {
        let sq = occupied.pop();
        let piece = pos.piece_at(sq);
        features[count] = halfka_feature_index(color, ksq, sq, piece);
        count += 1;
    }

    count
}

fn fill_row_from_position(
    pos: &Position,
    score: i32,
    result: i32,
    feature_set: &FeatureSet,
    white: &mut [i32],
    white_values: &mut [f32],
    black: &mut [i32],
    black_values: &mut [f32],
) -> RowMetadata {
    debug_assert_eq!(white.len(), feature_set.max_active_features());
    debug_assert_eq!(white_values.len(), feature_set.max_active_features());
    debug_assert_eq!(black.len(), feature_set.max_active_features());
    debug_assert_eq!(black_values.len(), feature_set.max_active_features());

    let white_count = feature_set.fill_features_sparse(pos, Color::White, white, white_values);
    let black_count = feature_set.fill_features_sparse(pos, Color::Black, black, black_values);
    white[white_count..].fill(-1);
    white_values[white_count..].fill(0.0);
    black[black_count..].fill(-1);
    black_values[black_count..].fill(0.0);
    debug_validate_sparse_features(white, white_count, feature_set.inputs());
    debug_validate_sparse_features(black, black_count, feature_set.inputs());
    let piece_count = pos.occupied().count() as i64;

    RowMetadata {
        is_white: if pos.side_to_move() == Color::White {
            1.0
        } else {
            0.0
        },
        outcome: (result as f32 + 1.0) / 2.0,
        score: score as f32,
        white_count,
        black_count,
        psqt_indices: (piece_count - 1) / 4,
        layer_stack_indices: (piece_count - 1) / 4,
    }
}

fn fill_row_from_position_without_values(
    pos: &Position,
    score: i32,
    result: i32,
    feature_set: &FeatureSet,
    white: &mut [i32],
    black: &mut [i32],
) -> RowMetadata {
    debug_assert_eq!(white.len(), feature_set.max_active_features());
    debug_assert_eq!(black.len(), feature_set.max_active_features());

    let white_count = feature_set.fill_feature_indices_sparse(pos, Color::White, white);
    let black_count = feature_set.fill_feature_indices_sparse(pos, Color::Black, black);
    white[white_count..].fill(-1);
    black[black_count..].fill(-1);
    debug_validate_sparse_features(white, white_count, feature_set.inputs());
    debug_validate_sparse_features(black, black_count, feature_set.inputs());
    let piece_count = pos.occupied().count() as i64;

    RowMetadata {
        is_white: if pos.side_to_move() == Color::White {
            1.0
        } else {
            0.0
        },
        outcome: (result as f32 + 1.0) / 2.0,
        score: score as f32,
        white_count,
        black_count,
        psqt_indices: (piece_count - 1) / 4,
        layer_stack_indices: (piece_count - 1) / 4,
    }
}

fn fill_full_threats_sparse(
    pos: &Position,
    perspective: Color,
    features: &mut [i32],
    values: &mut [f32],
) -> usize {
    let occupied = pos.occupied();
    let occupied_bits = occupied.bits();
    let occupied_pawns_bits = (pos.pieces_bb_color(Color::White, PieceType::Pawn)
        | pos.pieces_bb_color(Color::Black, PieceType::Pawn))
    .bits();
    let ksq = pos.king_sq(perspective);
    let color_order = match perspective {
        Color::White => [Color::White, Color::Black],
        Color::Black => [Color::Black, Color::White],
    };
    let mut count = 0usize;

    for color in color_order {
        for piece_type in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            let attacker = Piece::new(piece_type, color);
            let piece_bb = pos.pieces_bb_color(color, piece_type);

            if piece_type == PieceType::Pawn {
                let attacks_left =
                    Bitboard::new(pawn_attack_left_targets(piece_bb.bits(), color) & occupied_bits);
                let attacks_right = Bitboard::new(
                    pawn_attack_right_targets(piece_bb.bits(), color) & occupied_bits,
                );
                let attacks_forward =
                    Bitboard::new(pawn_push_targets(piece_bb.bits(), color) & occupied_pawns_bits);

                for to in attacks_left.iter() {
                    let from = match color {
                        Color::White => Square::new(to.index() - 9),
                        Color::Black => Square::new(to.index() + 9),
                    };
                    let attacked = pos.piece_at(to);
                    let index = full_threat_index(perspective, attacker, from, to, attacked, ksq);
                    if index >= 0 {
                        values[count] = 1.0;
                        features[count] = index;
                        count += 1;
                    }
                }

                for to in attacks_right.iter() {
                    let from = match color {
                        Color::White => Square::new(to.index() - 7),
                        Color::Black => Square::new(to.index() + 7),
                    };
                    let attacked = pos.piece_at(to);
                    let index = full_threat_index(perspective, attacker, from, to, attacked, ksq);
                    if index >= 0 {
                        values[count] = 1.0;
                        features[count] = index;
                        count += 1;
                    }
                }

                for to in attacks_forward.iter() {
                    let from = match color {
                        Color::White => Square::new(to.index() - 8),
                        Color::Black => Square::new(to.index() + 8),
                    };
                    let attacked = pos.piece_at(to);
                    let index = full_threat_index(perspective, attacker, from, to, attacked, ksq);
                    if index >= 0 {
                        values[count] = 1.0;
                        features[count] = index;
                        count += 1;
                    }
                }
            } else {
                for from in piece_bb.iter() {
                    let attacks = attacks::piece_attacks(piece_type, from, occupied) & occupied;
                    for to in attacks.iter() {
                        let attacked = pos.piece_at(to);
                        let index =
                            full_threat_index(perspective, attacker, from, to, attacked, ksq);
                        if index >= 0 {
                            values[count] = 1.0;
                            features[count] = index;
                            count += 1;
                        }
                    }
                }
            }
        }
    }

    count
}

fn fill_full_threat_feature_indices_sparse(
    pos: &Position,
    perspective: Color,
    features: &mut [i32],
) -> usize {
    let occupied = pos.occupied();
    let occupied_bits = occupied.bits();
    let occupied_pawns_bits = (pos.pieces_bb_color(Color::White, PieceType::Pawn)
        | pos.pieces_bb_color(Color::Black, PieceType::Pawn))
    .bits();
    let ksq = pos.king_sq(perspective);
    let color_order = match perspective {
        Color::White => [Color::White, Color::Black],
        Color::Black => [Color::Black, Color::White],
    };
    let mut count = 0usize;

    for color in color_order {
        for piece_type in [
            PieceType::Pawn,
            PieceType::Knight,
            PieceType::Bishop,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ] {
            let attacker = Piece::new(piece_type, color);
            let piece_bb = pos.pieces_bb_color(color, piece_type);

            if piece_type == PieceType::Pawn {
                let attacks_left =
                    Bitboard::new(pawn_attack_left_targets(piece_bb.bits(), color) & occupied_bits);
                let attacks_right = Bitboard::new(
                    pawn_attack_right_targets(piece_bb.bits(), color) & occupied_bits,
                );
                let attacks_forward =
                    Bitboard::new(pawn_push_targets(piece_bb.bits(), color) & occupied_pawns_bits);

                for to in attacks_left.iter() {
                    let from = match color {
                        Color::White => Square::new(to.index() - 9),
                        Color::Black => Square::new(to.index() + 9),
                    };
                    let attacked = pos.piece_at(to);
                    let index = full_threat_index(perspective, attacker, from, to, attacked, ksq);
                    if index >= 0 {
                        features[count] = index;
                        count += 1;
                    }
                }

                for to in attacks_right.iter() {
                    let from = match color {
                        Color::White => Square::new(to.index() - 7),
                        Color::Black => Square::new(to.index() + 7),
                    };
                    let attacked = pos.piece_at(to);
                    let index = full_threat_index(perspective, attacker, from, to, attacked, ksq);
                    if index >= 0 {
                        features[count] = index;
                        count += 1;
                    }
                }

                for to in attacks_forward.iter() {
                    let from = match color {
                        Color::White => Square::new(to.index() - 8),
                        Color::Black => Square::new(to.index() + 8),
                    };
                    let attacked = pos.piece_at(to);
                    let index = full_threat_index(perspective, attacker, from, to, attacked, ksq);
                    if index >= 0 {
                        features[count] = index;
                        count += 1;
                    }
                }
            } else {
                for from in piece_bb.iter() {
                    let attacks = attacks::piece_attacks(piece_type, from, occupied) & occupied;
                    for to in attacks.iter() {
                        let attacked = pos.piece_at(to);
                        let index =
                            full_threat_index(perspective, attacker, from, to, attacked, ksq);
                        if index >= 0 {
                            features[count] = index;
                            count += 1;
                        }
                    }
                }
            }
        }
    }

    count
}

fn halfka_feature_index(color: Color, ksq: Square, sq: Square, piece: Piece) -> i32 {
    let oriented_ksq = orient_flip(color, ksq, ksq);
    let piece_index = piece.piece_type().ordinal() as i32 * 2 + i32::from(piece.color() != color);
    let bucket = KING_BUCKETS[oriented_ksq.index() as usize];

    orient_flip(color, sq, ksq).index() as i32
        + piece_index * NUM_SQ as i32
        + bucket * NUM_PLANES as i32
}

fn full_threat_index(
    perspective: Color,
    attacker: Piece,
    from: Square,
    to: Square,
    attacked: Piece,
    ksq: Square,
) -> i32 {
    let enemy = attacker.color() != attacked.color();
    let mut attacker = attacker;
    let mut attacked = attacked;
    let from = full_threat_orient_flip(perspective, from, ksq);
    let to = full_threat_orient_flip(perspective, to, ksq);

    if perspective == Color::Black {
        attacker = Piece::from_id((attacker.id() ^ 1) as i32);
        attacked = Piece::from_id((attacked.id() ^ 1) as i32);
    }

    let attacker_type = attacker.piece_type();
    let attacked_type = attacked.piece_type();
    let mapped_target =
        THREAT_MAP[attacker_type.ordinal() as usize][attacked_type.ordinal() as usize];

    if mapped_target < 0
        || (attacker_type == attacked_type
            && (enemy || attacker_type != PieceType::Pawn)
            && from.index() < to.index())
    {
        return -1;
    }

    let attack_mask = if attacker_type == PieceType::Pawn {
        let mut attack_mask = attacks::pawn(attacker.color(), from);
        let push_to = match attacker.color() {
            Color::White => Square::new(from.index() + 8),
            Color::Black => Square::new(from.index() - 8),
        };
        attack_mask |= Bitboard::from_square(push_to);
        attack_mask
    } else {
        attacks::piece_attacks(attacker_type, from, Bitboard::new(0))
    };

    let calc = threat_feature_calc();
    let attacker_index = piece_index(attacker);
    calc.table[attacker_index][65]
        + (attacked.color().ordinal() as i32 * (NUM_VALID_TARGETS[attacker_index] / 2)
            + mapped_target)
            * calc.table[attacker_index][64]
        + calc.table[attacker_index][from.index() as usize]
        + ((Bitboard::from_before(to.index()) & attack_mask).count() as i32)
}

fn piece_index(piece: Piece) -> usize {
    piece.piece_type().ordinal() as usize * 2 + piece.color().ordinal() as usize
}

fn orient_mask(color: Color, ksq: Square) -> u32 {
    let horizontal_flip = if (ksq.index() & 7) < 4 { 7 } else { 0 };
    let vertical_flip = match color {
        Color::White => 0,
        Color::Black => 56,
    };
    horizontal_flip ^ vertical_flip
}

fn full_threat_orient_flip(color: Color, sq: Square, ksq: Square) -> Square {
    Square::new(sq.index() ^ full_threat_orient_mask(color, ksq))
}

fn full_threat_orient_mask(color: Color, ksq: Square) -> u32 {
    let horizontal_flip = if (ksq.index() & 7) >= 4 { 7 } else { 0 };
    let vertical_flip = match color {
        Color::White => 0,
        Color::Black => 56,
    };
    horizontal_flip ^ vertical_flip
}

fn pawn_attack_left_targets(pawns: u64, color: Color) -> u64 {
    match color {
        Color::White => (pawns & !H_FILE) << 9,
        Color::Black => (pawns & !A_FILE) >> 9,
    }
}

fn pawn_attack_right_targets(pawns: u64, color: Color) -> u64 {
    match color {
        Color::White => (pawns & !A_FILE) << 7,
        Color::Black => (pawns & !H_FILE) >> 7,
    }
}

fn pawn_push_targets(pawns: u64, color: Color) -> u64 {
    match color {
        Color::White => pawns << 8,
        Color::Black => pawns >> 8,
    }
}

fn threat_feature_calc() -> &'static ThreatFeatureCalculation {
    static THREAT_FEATURE_CALC: OnceLock<ThreatFeatureCalculation> = OnceLock::new();

    THREAT_FEATURE_CALC.get_or_init(|| {
        let mut table = [[0i32; 66]; 12];
        let mut piece_offset = 0i32;

        for color_index in 0..2 {
            for piece_type_index in 0..6 {
                let piece_index = 2 * piece_type_index + color_index;
                let piece = PIECE_TABLE[piece_index];
                table[piece_index][65] = piece_offset;

                let mut square_offset = 0i32;
                for from_index in 0..64u32 {
                    table[piece_index][from_index as usize] = square_offset;
                    let from = Square::new(from_index);

                    if piece.piece_type() != PieceType::Pawn {
                        square_offset +=
                            attacks::piece_attacks(piece.piece_type(), from, Bitboard::new(0))
                                .count() as i32;
                    } else if (8..=55).contains(&from_index) {
                        let mut attack_mask = attacks::pawn(piece.color(), from);
                        let push_to = match piece.color() {
                            Color::White => Square::new(from_index + 8),
                            Color::Black => Square::new(from_index - 8),
                        };
                        attack_mask |= Bitboard::from_square(push_to);
                        square_offset += attack_mask.count() as i32;
                    }
                }

                table[piece_index][64] = square_offset;
                piece_offset += NUM_VALID_TARGETS[piece_index] * square_offset;
            }
        }

        debug_assert_eq!(piece_offset, FULL_THREATS_INPUTS as i32);
        ThreatFeatureCalculation {
            table,
            total_features: piece_offset,
        }
    })
}

fn debug_validate_sparse_features(features: &[i32], active_count: usize, num_inputs: usize) {
    #[cfg(debug_assertions)]
    {
        let active = &features[..active_count];
        for &feature in active {
            debug_assert!((0..num_inputs as i32).contains(&feature));
        }
        for pair in active.windows(2) {
            let _ = pair;
        }
        let mut sorted = active.to_vec();
        sorted.sort_unstable();
        sorted
            .windows(2)
            .for_each(|pair| debug_assert_ne!(pair[0], pair[1]));
        debug_assert!(features[active_count..].iter().all(|&value| value == -1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_set_parser_supports_composed_features() {
        let feature_set = FeatureSet::from_str("Full_Threats+HalfKAv2_hm")
            .expect("composed feature set should parse");

        assert_eq!(feature_set.inputs(), FULL_THREATS_INPUTS + HALFKA_INPUTS);
        assert_eq!(
            feature_set.max_active_features(),
            FULL_THREATS_MAX_ACTIVE_FEATURES + HALFKA_MAX_ACTIVE_FEATURES
        );
        assert_eq!(feature_set.components().len(), 2);
    }

    #[test]
    fn halfka_startpos_extracts_32_features_per_side() {
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("valid startpos FEN");
        let row = build_sparse_row(&pos, 0, 0);

        assert_eq!(row.white_count, 32);
        assert_eq!(row.black_count, 32);
        assert_eq!(row.white.len(), HALFKA_MAX_ACTIVE_FEATURES);
        assert_eq!(row.black.len(), HALFKA_MAX_ACTIVE_FEATURES);
    }
}
