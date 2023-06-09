use shakmaty::{MoveList, Position};
use shakmaty_syzygy::Wdl;

use crate::{math, nnue_eval::NNUEState};
use crate::search::SCALE;
use crate::state::{self, State};
use crate::tablebase::probe_tablebase_wdl;

const MATE: Evaluation = Evaluation::Terminal(SCALE as i64);
const DRAW: Evaluation = Evaluation::Terminal(0);

const TB_WIN: Evaluation = Evaluation::Tablebase(SCALE as i64);
const TB_LOSS: Evaluation = Evaluation::Tablebase(-SCALE as i64);
const TB_DRAW: Evaluation = Evaluation::Tablebase(0);

#[derive(Debug, Copy, Clone)]
pub enum Evaluation {
    Scaled(i64),
    Tablebase(i64),
    Terminal(i64),
}

impl Evaluation {
    pub const fn draw() -> Self {
        DRAW
    }

    pub fn flip(&self) -> Self {
        match self {
            Evaluation::Scaled(s) => Evaluation::Scaled(-s),
            Evaluation::Tablebase(s) => Evaluation::Tablebase(-s),
            Evaluation::Terminal(s) => Evaluation::Terminal(-s),
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Evaluation::Terminal(_))
    }

    pub fn is_tablebase(&self) -> bool {
        matches!(self, Evaluation::Tablebase(_))
    }
}

impl From<f32> for Evaluation {
    fn from(f: f32) -> Self {
        Evaluation::Scaled((f * SCALE) as i64)
    }
}

impl From<Evaluation> for i64 {
    fn from(e: Evaluation) -> Self {
        match e {
            Evaluation::Scaled(s) | Evaluation::Tablebase(s) | Evaluation::Terminal(s) => s,
        }
    }
}

pub fn evaluate_new_state(state: &State, moves: &MoveList) -> (Vec<f32>, Evaluation) {
    let (state_evaluation, move_evaluations) = run_nets(state, moves);

    let state_evaluation = if moves.is_empty() {
        if state.board().is_check() {
            MATE.flip()
        } else {
            DRAW
        }
    } else if let Some(wdl) = probe_tablebase_wdl(state.board()) {
        match wdl {
            Wdl::Win => TB_WIN,
            Wdl::Loss => TB_LOSS,
            _ => TB_DRAW,
        }
    } else {
        Evaluation::from(state_evaluation)
    };

    (
        move_evaluations,
        state
            .side_to_move()
            .fold_wb(state_evaluation, state_evaluation.flip()),
    )
}

const POLICY_NUMBER_INPUTS: usize = state::NUMBER_FEATURES;

#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
static POLICY_WEIGHTS: [[f32; POLICY_NUMBER_INPUTS]; 384] = include!("policy/output_weights");

fn run_nets(state: &State, moves: &MoveList) -> (f32, Vec<f32>) {
    let mut evalns = Vec::with_capacity(moves.len());

    if moves.is_empty() {
        // Returning 0 is ok here, as we'll immediately check for why there's no moves
        return (0., evalns);
    }

    let mut move_idxs = Vec::with_capacity(moves.len());

    for m in 0..moves.len() {
        move_idxs.push(state.move_to_index(&moves[m]));
        evalns.push(0.);
    }

    state.features_map(|idx| {
        for m in 0..moves.len() {
            evalns[m] += POLICY_WEIGHTS[move_idxs[m]][idx];
        }
    });

    math::softmax(&mut evalns);

    let mut nnue_buffer = NNUEState::new();
    let value = nnue_buffer.forward(state.board()) as f32 / 400.0;
    let activated = 1. / (1. + (-value).exp());
    let value = activated * 2. - 1.;

    (value, evalns)
}
