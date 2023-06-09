use shakmaty::{Color, Role, Square, Chess, Setup};


#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct Align<T>(pub T);

impl<T, const SIZE: usize> std::ops::Deref for Align<[T; SIZE]> {
    type Target = [T; SIZE];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T, const SIZE: usize> std::ops::DerefMut for Align<[T; SIZE]> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Activations of the hidden layer.
#[derive(Debug, Clone, Copy)]
pub struct Accumulator {
    pub white: Align<[i16; LAYER_1_SIZE]>,
    pub black: Align<[i16; LAYER_1_SIZE]>,
}

impl Accumulator {
    /// Evaluate the final layer on the partial activations.
    pub fn evaluate(&self, stm: Color) -> i32 {
        let (us, them) =
            if stm == Color::White { (&self.white, &self.black) } else { (&self.black, &self.white) };

        let output = screlu_flatten(us, them, &NNUE.output_weights);

        (output + i32::from(NNUE.output_bias)) * SCALE / QAB
    }
}

/// The size of the input layer of the network.
const INPUT: usize = 768;
/// The minimum value for the clipped relu activation.
const CR_MIN: i16 = 0;
/// The maximum value for the clipped relu activation.
const CR_MAX: i16 = 255;
/// The amount to scale the output of the network by.
/// This is to allow for the sigmoid activation to differentiate positions with
/// a small difference in evaluation.
const SCALE: i32 = 400;
/// The size of one-half of the hidden layer of the network.
pub const LAYER_1_SIZE: usize = 768;

const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

pub trait Activation {
    const ACTIVATE: bool;
    type Reverse: Activation;
}
pub struct Activate;
impl Activation for Activate {
    const ACTIVATE: bool = true;
    type Reverse = Deactivate;
}
pub struct Deactivate;
impl Activation for Deactivate {
    const ACTIVATE: bool = false;
    type Reverse = Activate;
}

// read in bytes from files and transmute them into u16s.
// SAFETY: alignment to u16 is guaranteed because transmute() is a copy operation.
pub static NNUE: NNUEParams = NNUEParams {
    feature_weights: unsafe { std::mem::transmute(*include_bytes!("../nnue/feature_weights.bin")) },
    feature_bias: unsafe { std::mem::transmute(*include_bytes!("../nnue/feature_bias.bin")) },
    output_weights: unsafe { std::mem::transmute(*include_bytes!("../nnue/output_weights.bin")) },
    output_bias: unsafe { std::mem::transmute(*include_bytes!("../nnue/output_bias.bin")) },
};

pub struct NNUEParams {
    pub feature_weights: Align<[i16; INPUT * LAYER_1_SIZE]>,
    pub feature_bias: Align<[i16; LAYER_1_SIZE]>,
    pub output_weights: Align<[i16; LAYER_1_SIZE * 2]>,
    pub output_bias: i16,
}

/// State of the partial activations of the NNUE network.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct NNUEState {
    pub accumulator: Accumulator
}

fn feature_indices(sq: Square, piece_type: Role, colour: Color) -> (usize, usize) {
    const COLOUR_STRIDE: usize = 64 * 6;
    const PIECE_STRIDE: usize = 64;

    let piece_type = piece_type as usize - 1; // hack for shakmaty having pawn = 1
    let colour = 1 ^ colour as usize; // hack for shakmaty having black = 0

    let white_idx = colour * COLOUR_STRIDE + piece_type * PIECE_STRIDE + sq as usize;
    let black_idx =
        (1 ^ colour) * COLOUR_STRIDE + piece_type * PIECE_STRIDE + sq.flip_vertical() as usize;

    (white_idx, black_idx)
}

impl NNUEState {
    /// Create a new `NNUEState`.
    pub fn new() -> Self {
        Self {
            accumulator: Accumulator {
                white: NNUE.feature_bias,
                black: NNUE.feature_bias,
            },
        }
    }

    /// Calculate the evaluation of the position.
    pub fn forward(&mut self, board: &Chess) -> i32 {
        for colour in [Color::White, Color::Black] {
            for piece_type in Role::ALL {
                let piece_bb = if board.turn() == colour {
                    board.our(piece_type)
                } else {
                    board.their(piece_type)
                };

                for sq in piece_bb {
                    self.update_feature::<Activate>(piece_type, colour, sq);
                }
            }
        }

        self.accumulator.evaluate(board.turn())
    }

    /// Update by activating or deactivating a piece.
    fn update_feature<A: Activation>(
        &mut self,
        piece_type: Role,
        colour: Color,
        sq: Square,
    ) {
        let (white_idx, black_idx) = feature_indices(sq, piece_type, colour);
        let acc = &mut self.accumulator;

        if A::ACTIVATE {
            add_to_all(&mut acc.white, &NNUE.feature_weights, white_idx * LAYER_1_SIZE);
            add_to_all(&mut acc.black, &NNUE.feature_weights, black_idx * LAYER_1_SIZE);
        } else {
            sub_from_all(&mut acc.white, &NNUE.feature_weights, white_idx * LAYER_1_SIZE);
            sub_from_all(&mut acc.black, &NNUE.feature_weights, black_idx * LAYER_1_SIZE);
        }
    }
}

/// Add a feature to a square.
fn add_to_all<const SIZE: usize, const WEIGHTS: usize>(
    input: &mut Align<[i16; SIZE]>,
    delta: &Align<[i16; WEIGHTS]>,
    offset_add: usize,
) {
    let a_block = &delta[offset_add..offset_add + SIZE];
    for (i, d) in input.iter_mut().zip(a_block) {
        *i += *d;
    }
}

/// Subtract a feature from a square.
fn sub_from_all<const SIZE: usize, const WEIGHTS: usize>(
    input: &mut Align<[i16; SIZE]>,
    delta: &Align<[i16; WEIGHTS]>,
    offset_sub: usize,
) {
    let s_block = &delta[offset_sub..offset_sub + SIZE];
    for (i, d) in input.iter_mut().zip(s_block) {
        *i -= *d;
    }
}

fn screlu(x: i16) -> i32 {
    let x = x.clamp(CR_MIN, CR_MAX);
    let x = i32::from(x);
    x * x
}

/// Execute squared + clipped relu on the partial activations,
/// and accumulate the result into a sum.
pub fn screlu_flatten(
    us: &Align<[i16; LAYER_1_SIZE]>,
    them: &Align<[i16; LAYER_1_SIZE]>,
    weights: &Align<[i16; LAYER_1_SIZE * 2]>,
) -> i32 {
    let mut sum: i32 = 0;
    for (&i, &w) in us.iter().zip(&weights[..LAYER_1_SIZE]) {
        sum += screlu(i) * i32::from(w);
    }
    for (&i, &w) in them.iter().zip(&weights[LAYER_1_SIZE..]) {
        sum += screlu(i) * i32::from(w);
    }
    sum / QA
}