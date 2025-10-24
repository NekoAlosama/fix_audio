use core::f64::consts::TAU;
use itertools::izip;
use realfft::{RealFftPlanner, num_complex::Complex};

/// List of cosine coefficients of window function
// https://holometer.fnal.gov/GH_FFT.pdf
// Processing time can be reduced using an HFTxx(D) window with less coefficients
// Currently using D.3.4 HFT116D, good enough to capture all 19.4-bit-or-less audio
// Might be worth experimenting with the HFT70, HFT95, or HFT90D since spectral leakage might not be noticed here
// (Internal multiplier, external multiplier)
const WINDOW_COSINES: [(f64, f64); 5] = [
    (TAU, -1.957_537_5_f64),
    (2.0 * TAU, 1.478_070_5_f64),
    (3.0 * TAU, -0.636_743_1_f64),
    (4.0 * TAU, 0.122_838_9_f64),
    (5.0 * TAU, -0.006_628_8_f64),
];

/// Faster `.norm_sqr()` that uses `mul_add`
fn norm_squared(complex: Complex<f64>) -> f64 {
    f64::mul_add(complex.re, complex.re, complex.im * complex.im)
}

/// Aligns the phase angle of the left and right channels
/// Generally speaking, it seems that I want the maximum phase shift to be 90 degrees per channel.
/// This algorithm made each channel a scaled verison of `sum` so the phase angles were correct. If `sum` was
///   near zero, then it could lead to near-180-degree flips on one channel
/// Currently being researched on how to smoothly saturate to 90 degrees per channel
///   Previous experimental algorithm modified the amplitude of the channels, approximating this algorithm if
///   the channels were at least somewhat in-phase, but attenuated one channel to silence and increased the other if out-of-phase
#[expect(
    clippy::arithmetic_side_effects,
    reason = "clippy thinks the operations done on Complex<f64> are for integers"
)]
fn align(original_left: &mut Complex<f64>, original_right: &mut Complex<f64>) {
    // norm_squared().sqrt() is used over .hypot() for efficiency
    // TODO: maybe combine norm_squared().sqrt()? We never need norm_squared() by itself
    // Clicks still present unfortunately
    let left_norm = norm_squared(*original_left).sqrt();
    let right_norm = norm_squared(*original_right).sqrt();
    let sum = *original_left + *original_right;
    let normalized_sum = sum / norm_squared(sum).sqrt();

    let true_modifed_left = left_norm * normalized_sum;
    let true_modifed_right = right_norm * normalized_sum;

    if true_modifed_left.is_finite() && true_modifed_right.is_finite() {
        // Ideally, any NaNs and Inf's should be caught by the above
        *original_left = true_modifed_left;
        *original_right = true_modifed_right;
    } else {
        // This case occurs when the `sum` is near zero, i.e. `left` approximately equals `-right`
        // The best solutions seem to be just replicating the near-zero behavior, as in:
        //   1. Copy left channel to right channel (if using simple algorithm)
        //      Just to be consistent: copying the right channel to the left should be equivalent sonically
        //      Should be mostly equivalent to inverting the right channel, but should be faster that inverting?
        //   2. Randomly silence one channel and double the magnitude of the other (if using experimental algorithm)
        //      It'd suck to just silence the left channel and double the right all the time
        *original_right = *original_left;
    }
}

/// Windowing is used to make the signal chunk fade in and out
///   to prevent discontinuities, which causes spectral leakage (noise tuned to the music).
fn window(time_frame: usize) -> Box<[f64]> {
    let f64_rate_recip = 1.0_f64 / (time_frame as f64);
    // The actual level of the window doesn't really matter
    // Window selection: minimize side-lobe level, ignore bandwidth of main lobe?
    (0..time_frame)
        .map(|n| {
            WINDOW_COSINES
                .iter()
                .copied()
                .fold(1.0_f64, |acc, (internal, external)| {
                    external.mul_add(f64::cos(internal * n as f64 * f64_rate_recip), acc)
                })
        })
        .collect()
}

/// Specific overlapping
pub fn overlapping_fft(
    planner: &mut RealFftPlanner<f64>,
    time_frame: f64,
    left_channel: Vec<f64>,
    right_channel: Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let original_length = left_channel.len();

    // Idea is that time_frame gives us the amount of samples (possibly fractional) that we need to FFT
    let rounded_time_frame = time_frame.round_ties_even() as usize;
    // We should pad with half-a-second of silence to allow for half-windows at the beginning and end
    let half_time_frame = (time_frame * 0.5_f64).round_ties_even() as usize;

    // Lots of Vecs are used here to reuse memory space instead of reallocating
    let fft_size = rounded_time_frame.next_power_of_two();
    let fft_norm = 1.0_f64 / fft_size as f64;
    let r2c = planner.plan_fft_forward(fft_size);
    let c2r = planner.plan_fft_inverse(fft_size);
    let mut left_chunk = Vec::with_capacity(fft_size);
    let mut right_chunk = Vec::with_capacity(fft_size);
    let mut left_complex = r2c.make_output_vec();
    let mut right_complex = r2c.make_output_vec();
    let mut scratch_complex = r2c.make_scratch_vec();
    let mut scratch_real = c2r.make_scratch_vec();

    // This consumes left_channel and right_channel
    let mut extended_left = vec![0.0_f64; half_time_frame]; // Allow half-window at start
    let mut extended_right = vec![0.0_f64; half_time_frame];
    extended_left.extend(left_channel);
    extended_right.extend(right_channel);
    extended_left.extend(vec![0.0_f64; fft_size]); // Allow for last FFT chunk to be added
    extended_right.extend(vec![0.0_f64; fft_size]);
    let extended_length = extended_left.len();

    let mut holding_left = vec![0.0_f64; extended_length];
    let mut holding_right = vec![0.0_f64; extended_length];
    let mut holding_position = 0_usize;

    let window = window(rounded_time_frame);

    // Windows need a bunch of hops
    // Doing more chunks will help with clicking/zipper noise, but will increase runtime
    let hop_time_frame = 1.0_f64 / (WINDOW_COSINES.len() as f64 + 1.0_f64);
    let hop_size = (time_frame * hop_time_frame).round_ties_even() as usize;

    // Up until the end, which should be basically a half-window
    while holding_position <= usize::strict_sub(extended_length, rounded_time_frame) {
        left_chunk.extend(
            extended_left
                .iter()
                .skip(holding_position)
                .take(rounded_time_frame),
        );
        right_chunk.extend(
            extended_right
                .iter()
                .skip(holding_position)
                .take(rounded_time_frame),
        );

        izip!(left_chunk.iter_mut(), right_chunk.iter_mut(), window.iter()).for_each(
            |(left_samp, right_samp, window_mult)| {
                *left_samp *= *window_mult;
                *right_samp *= *window_mult;
            },
        );

        left_chunk.resize(fft_size, 0.0_f64);
        right_chunk.resize(fft_size, 0.0_f64);

        _ = r2c.process_with_scratch(&mut left_chunk, &mut left_complex, &mut scratch_complex);
        _ = r2c.process_with_scratch(&mut right_chunk, &mut right_complex, &mut scratch_complex);

        // Since DC bias is removed, the DC/first/zeroth bin should be near zero and
        //   it should be fine to manipulate it
        // Even if it isn't near zero, the value probably spread out to the other bins anyway
        izip!(left_complex.iter_mut(), right_complex.iter_mut()).for_each(
            |(left_point, right_point)| {
                align(left_point, right_point);
            },
        );

        // left_chunk and right_chunk will be overwritten
        _ = c2r.process_with_scratch(&mut left_complex, &mut left_chunk, &mut scratch_real);
        _ = c2r.process_with_scratch(&mut right_complex, &mut right_chunk, &mut scratch_real);

        // RustFFT, and in turn RealFFT, do not perform post-FFT normalization
        izip!(left_chunk.iter_mut(), right_chunk.iter_mut()).for_each(|(left_samp, right_samp)| {
            *left_samp *= fft_norm;
            *right_samp *= fft_norm;
        });

        #[expect(
            clippy::iter_with_drain,
            reason = "nursery lint, .drain(..) is needed to clear the chunk vecs"
        )]
        izip!(
            holding_left.iter_mut().skip(holding_position),
            holding_right.iter_mut().skip(holding_position),
            left_chunk.drain(..),
            right_chunk.drain(..)
        )
        .for_each(|(hold_left, hold_right, left_samp, right_samp)| {
            *hold_left += left_samp;
            *hold_right += right_samp;
        });

        holding_position = usize::strict_add(holding_position, hop_size);
    }

    let f64_hop_time_frame = hop_time_frame;
    izip!(holding_left.iter_mut(), holding_right.iter_mut()).for_each(|(hold_left, hold_right)| {
        *hold_left *= f64_hop_time_frame;
        *hold_right *= f64_hop_time_frame;
    });

    (
        holding_left
            .iter()
            .skip(half_time_frame)
            .take(original_length)
            .copied()
            .collect(),
        holding_right
            .iter()
            .skip(half_time_frame)
            .take(original_length)
            .copied()
            .collect(),
    )
}
