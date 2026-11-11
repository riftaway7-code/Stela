//! Half arithmetic lanes (`half::emit`) against the reference on every
//! target the host runs: every inline completion matches the reference in
//! its lanes and FPSR, every exit leaves the state unchanged, and normal
//! inputs with a normal result at least twice the minimum, and vector lanes
//! with a signed zero input beside a normal or zero one, must complete
//! inline.
use crate::{
    GuestAddress, GuestState, Options, TargetCpu,
    aarch64::floating::{
        BinaryOperation, FPCR_AH, FPCR_DN, FPCR_EXCEPTION_ENABLE_MASK, FPCR_FZ16,
        FPCR_ROUNDING_SHIFT, Format, INEXACT, evaluate,
    },
    block::{BlockProfile, CompilationPlan},
    x86_64::native::run,
};
use alloc::vec::Vec;

const PC: GuestAddress = GuestAddress(0x1000_0000);
const RETURN: u64 = 0x4000;
const D: u32 = 3;
const N: u32 = 4;
const M: u32 = 5;

fn targets() -> Vec<TargetCpu> {
    let host = TargetCpu::detect();
    let mut targets: Vec<TargetCpu> = [
        TargetCpu::Baseline,
        TargetCpu::X86_64_V2,
        TargetCpu::X86_64_V2.with_f16c(),
        TargetCpu::X86_64_V3,
    ]
    .into_iter()
    .filter(|target| host.supports(*target))
    .collect();
    targets.push(host);
    targets
}

fn compile(word: u32, target: TargetCpu) -> Vec<u8> {
    let source = [word, 0xd65f_03c0].map(u32::to_le_bytes).concat();
    CompilationPlan::new(&source, PC, &Options::new(BlockProfile::ORDINARY, target))
        .unwrap()
        .compile()
        .unwrap()
        .native
        .x86_64_code
}

/// (word without registers, operation, lanes; 1 is scalar).
const FORMS: &[(u32, BinaryOperation, usize)] = &[
    (0x6e40_3c00, BinaryOperation::Divide, 8),
    (0x2e40_3c00, BinaryOperation::Divide, 4),
    (0x1ee0_1800, BinaryOperation::Divide, 1),
    (0x4e40_1400, BinaryOperation::Add, 8),
    (0x6e40_1c00, BinaryOperation::Multiply, 8),
    (0x0e40_1400, BinaryOperation::Add, 4),
    (0x4ec0_1400, BinaryOperation::Subtract, 8),
    (0x2e40_1c00, BinaryOperation::Multiply, 4),
];

fn controls() -> Vec<u32> {
    let mut controls: Vec<u32> = (0..4).map(|mode| mode << FPCR_ROUNDING_SHIFT).collect();
    controls.extend([
        FPCR_FZ16,
        FPCR_AH,
        FPCR_DN,
        1 << 12, // IXE
        FPCR_EXCEPTION_ENABLE_MASK,
    ]);
    controls
}

fn half(random: &mut u64) -> u64 {
    *random ^= *random << 13;
    *random ^= *random >> 7;
    *random ^= *random << 17;
    match *random % 16 {
        0 | 5 => (*random >> 40 & 1) << 15,
        1 => 0x7c00,
        2 => *random >> 20 & 0x3ff,
        // Exact quotients and products: powers of two and small integers.
        3 => 0x3c00 | (*random >> 24 & 0x7) << 10,
        4 => 0x4000 + (*random >> 30 & 0x1f) * 0x80,
        _ => ((*random >> 16 & 0x7fff) % 0x7800 + 0x400) | ((*random >> 40 & 1) << 15),
    }
}

fn normal(bits: u64) -> bool {
    let exponent = bits >> 10 & 31;
    exponent != 0 && exponent != 31
}

#[test]
fn half_arithmetic_lanes_match_the_reference() {
    let mut random = 0x1357_9bdf_2468_ace0_u64;
    let mut inline = 0usize;
    for target in targets() {
        for &(base, operation, lanes) in FORMS {
            let word = base | M << 16 | N << 5 | D;
            let code = compile(word, target);
            for case in 0..6000 {
                let inputs: Vec<(u64, u64)> = (0..lanes)
                    .map(|_| (half(&mut random), half(&mut random)))
                    .collect();
                let pack = |pick: &dyn Fn(&(u64, u64)) -> u64| {
                    let mut limbs = [0u64; 2];
                    for (lane, pair) in inputs.iter().enumerate() {
                        limbs[lane / 4] |= pick(pair) << (16 * (lane % 4));
                    }
                    limbs
                };
                let controls: Vec<u32> = if case % 8 == 0 {
                    controls()
                } else {
                    alloc::vec![0]
                };
                for fpcr in controls {
                    for fpsr in [0, INEXACT] {
                        let mut initial = GuestState {
                            fpcr,
                            fpsr,
                            ..GuestState::default()
                        };
                        initial.x[30] = RETURN;
                        initial.v[N as usize] = pack(&|pair| pair.0);
                        initial.v[M as usize] = pack(&|pair| pair.1);
                        initial.v[D as usize] = [u64::MAX, u64::MAX];
                        let mut actual = initial.clone();
                        let exit = run(&code, &mut actual);
                        let outcomes: Vec<_> = inputs
                            .iter()
                            .map(|&(left, right)| {
                                evaluate(Format::Half, operation, fpcr, left, right)
                            })
                            .collect();
                        let covered = fpcr & FPCR_EXCEPTION_ENABLE_MASK == 0
                            && inputs
                                .iter()
                                .zip(&outcomes)
                                .all(|(&(left, right), outcome)| {
                                    let outcome = outcome.as_ref().unwrap();
                                    let bits = outcome.bits;
                                    let zero = |bits: u64| bits & 0x7fff == 0;
                                    let finite = |bits| normal(bits) || zero(bits);
                                    // Vector lanes with a zero input and no zero divisor.
                                    let zero_lane = lanes > 1
                                        && (zero(left) || zero(right))
                                        && finite(left)
                                        && finite(right)
                                        && !(operation == BinaryOperation::Divide && zero(right));
                                    // Overflow and underflow take the helper.
                                    zero_lane
                                        || normal(left)
                                            && normal(right)
                                            && normal(bits)
                                            && bits >> 10 & 31 >= 2
                                            && outcome.exceptions & 0b1100 == 0
                                });
                        if exit == PC.0 {
                            assert_eq!(actual, initial, "{word:08x} {inputs:x?} {fpcr:#x}");
                            assert!(
                                !covered,
                                "covered lanes exited: {word:08x} {inputs:x?} {fpcr:#x} {target:?}"
                            );
                            continue;
                        }
                        assert_eq!(exit, RETURN, "{word:08x} {inputs:x?}");
                        let mut expected = initial.clone();
                        let mut limbs = [0u64; 2];
                        for (lane, outcome) in outcomes.iter().enumerate() {
                            let outcome = outcome.as_ref().expect("no enabled traps");
                            expected.fpsr |= outcome.exceptions;
                            limbs[lane / 4] |= outcome.bits << (16 * (lane % 4));
                        }
                        expected.v[D as usize] = limbs;
                        assert_eq!(
                            actual, expected,
                            "{word:08x} {inputs:x?} {fpcr:#x} {fpsr:#x} {target:?}"
                        );
                        inline += 1;
                    }
                }
            }
        }
    }
    assert!(inline > 20_000, "{inline}");
}
