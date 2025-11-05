use core::f64::consts::TAU;
use itertools::izip;
use realfft::{RealFftPlanner, num_complex::Complex};

/// List of cosine coefficients of window function
// https://holometer.fnal.gov/GH_FFT.pdf
// Currently using the HFT248D window, which needs 11 overlaps but has
//   a maximum leakage level of -248.4dB.
// While this seems like overkill, the levels are amplified due to FFT and overlap-adding,
//   so there is a chance that a lesser window would introduce clicks
const WINDOW_COSINES: [(f64, f64); 10] = [
    (1. * TAU, -1.985_844_164_102),
    (2. * TAU, 1.791_176_438_506),
    (3. * TAU, -1.282_075_284_005),
    (4. * TAU, 0.667_777_530_266),
    (5. * TAU, -0.240_160_796_57),
    (6. * TAU, 0.056_656_381_764),
    (7. * TAU, -0.008_134_974_479),
    (8. * TAU, 0.000_624_544_650),
    (9. * TAU, -0.000_019_808_998),
    (10. * TAU, 0.000_000_132_974),
];

/// Windowing is used to make the signal chunk fade in and out
///   to prevent discontinuities, which causes spectral leakage (noise tuned to the music).
fn window(time_frame: usize) -> Box<[f64]> {
    let f64_rate_recip = 1_f64 / (time_frame as f64);
    // The actual level of the window doesn't really matter
    // Window selection: minimize side-lobe level, ignore bandwidth of main lobe?
    (0..time_frame)
        .map(|n| {
            WINDOW_COSINES
                .iter()
                .copied()
                .fold(1_f64, |acc, (internal, external)| {
                    external.mul_add(f64::cos(internal * n as f64 * f64_rate_recip), acc)
                })
        })
        .collect()
}

/// Faster `.norm_sqr()` that uses `mul_add`
fn norm_squared(complex: Complex<f64>) -> f64 {
    f64::mul_add(complex.re, complex.re, complex.im * complex.im)
}

/// Faster `.is_finite()` that skips NaN check in `num_traits::FloatCore`
fn is_finite(complex: Complex<f64>) -> bool {
    complex.re.abs() < f64::INFINITY && complex.im.abs() < f64::INFINITY
}

/// Aligns the phase angle of the left and right channels
#[expect(
    clippy::arithmetic_side_effects,
    reason = "clippy thinks the operations done on Complex<f64> are for integers"
)]
fn align(original_left: &mut Complex<f64>, original_right: &mut Complex<f64>) {
    // norm_squared().sqrt() is used over .hypot() for efficiency
    // TODO: maybe combine norm_squared().sqrt()? We never need norm_squared() by itself
    let left_norm = norm_squared(*original_left).sqrt();
    let right_norm = norm_squared(*original_right).sqrt();

    let target_sum = *original_left + *original_right;
    let normalized_sum = target_sum / norm_squared(target_sum).sqrt();
    if is_finite(normalized_sum) {
        *original_left = left_norm * normalized_sum;
        *original_right = right_norm * normalized_sum;
    } else {
        // If the above doesn't work, the channels are out-of-phase with each other. To be consistent,
        //   we'll just invert the right channel or copy the left channel to make them in-phase.
        *original_right = *original_left;
    }
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
    // For fft_size specifically, there's different opinions online on how much zero-padding is needed
    let fft_size = rounded_time_frame
        .next_power_of_two() // Round to next power of two for more zero-padding and for fast FFT algorithm
        .saturating_mul(2_usize); // Ensure at least 50% of FFT is silence
    let r2c = planner.plan_fft_forward(fft_size);
    let c2r = planner.plan_fft_inverse(fft_size);
    let mut left_chunk = Vec::with_capacity(fft_size);
    let mut right_chunk = Vec::with_capacity(fft_size);
    let mut left_complex = r2c.make_output_vec();
    let mut right_complex = r2c.make_output_vec();
    let mut scratch_complex = r2c.make_scratch_vec();
    let mut scratch_real = c2r.make_scratch_vec();

    // This consumes left_channel and right_channel
    let mut extended_left = vec![0_f64; half_time_frame]; // Allow half-window at start
    let mut extended_right = vec![0_f64; half_time_frame];
    extended_left.extend(left_channel);
    extended_right.extend(right_channel);
    extended_left.extend(vec![0_f64; fft_size]); // Allow for last FFT chunk to be added
    extended_right.extend(vec![0_f64; fft_size]);
    let extended_length = extended_left.len();

    let mut holding_left = vec![0_f64; extended_length];
    let mut holding_right = vec![0_f64; extended_length];
    let mut holding_position = 0_usize;

    let window = window(rounded_time_frame);

    // Windows need a bunch of hops
    // Doing more chunks will help with clicking/zipper noise, but will increase runtime
    let hop_time_frame = 1_f64 / (WINDOW_COSINES.len() as f64 + 1_f64);
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

        left_chunk.resize(fft_size, 0_f64);
        right_chunk.resize(fft_size, 0_f64);

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

        // RealFFT, which uses RustFFT, amplifies the signal by fft_size
        // Normalization happens later in processing.rs

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

    // Overlap-adding amplifies the signal by (WINDOW_COSINES.len() as f64 + 1_f64) or 1/hop_time_frame
    // Normalization happens later in processing.rs

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
