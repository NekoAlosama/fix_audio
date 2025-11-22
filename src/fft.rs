use core::f64::consts::TAU;

use itertools::izip;
use realfft::{RealFftPlanner, num_complex::Complex};

/// List of cosine coefficients of window function
///
/// Taken from <https://holometer.fnal.gov/GH_FFT.pdf>
// Ideal candidate is HFT144D, a flat top window which needs 7 overlaps and has a noise floor of -144.1dB, enough for the humman auditory system
// The bandwidth of the main lobe doesn't seem to matter for some reason
const WINDOW_COSINES: [(f64, f64); 6] = [
    (1. * TAU, -1.967_600_33),
    (2. * TAU, 1.579_836_07),
    (3. * TAU, -0.811_236_44),
    (4. * TAU, 0.225_835_58),
    (5. * TAU, -0.027_738_48),
    (6. * TAU, 0.000_903_60),
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
                .into_iter()
                .fold(1_f64, |acc, (internal, external)| {
                    external.mul_add(f64::cos(internal * n as f64 * f64_rate_recip), acc)
                })
        })
        .collect()
}

/// Faster `.norm_sqr()` that uses `mul_add`
fn norm_squared(complex: &Complex<f64>) -> f64 {
    f64::mul_add(complex.re, complex.re, complex.im * complex.im)
}

/// Faster `.is_finite()` that skips NaN check in `num_traits::FloatCore`
fn is_finite(complex: &Complex<f64>) -> bool {
    complex.re.abs() < f64::INFINITY && complex.im.abs() < f64::INFINITY
}

/// Aligns the phase angle of the left and right channels
#[expect(
    clippy::arithmetic_side_effects,
    reason = "clippy thinks the operations done on Complex<f64> are for integers"
)]
fn align(original_left: &mut Complex<f64>, original_right: &mut Complex<f64>) {
    // custom function is used over .hypot() for efficiency
    let left_norm_sqr = norm_squared(original_left);
    let right_norm_sqr = norm_squared(original_right);

    // This method aligns the quieter channel with the louder channel.
    // Mathematically, this seems to minimize the rotation distance needed.
    // Of course, this still creates clicks, but is simpler than the `sum` method
    if left_norm_sqr < right_norm_sqr {
        let new_left = *original_right * (left_norm_sqr / right_norm_sqr).sqrt();
        if is_finite(&new_left) {
            *original_left = new_left;
        }
    }
    // Implicitly left_norm_sqr >= right_norm_sqr
    else {
        let new_right = *original_left * (right_norm_sqr / left_norm_sqr).sqrt();
        if is_finite(&new_right) {
            *original_right = new_right;
        }
    }

    // If no channel was changed, then left_norm_sqr and right_norm_sqr had to be pretty small
}

/// Specific overlapping
pub fn overlapping_fft(
    planner: &mut RealFftPlanner<f64>,
    time_frame: f64,
    left_channel: Box<[f64]>,
    right_channel: Box<[f64]>,
) -> (Box<[f64]>, Box<[f64]>) {
    let original_length = left_channel.len();

    // Idea is that time_frame gives us the amount of samples (possibly fractional) that we need to FFT
    let rounded_time_frame = time_frame.round_ties_even() as usize;
    // We should pad with half-a-second of silence to allow for half-windows at the beginning and end
    let half_time_frame = (time_frame * 0.5_f64).round_ties_even() as usize;

    // This consumes left_channel and right_channel
    let mut pre_extended_left = vec![0_f64; half_time_frame]; // Allow half-window at start
    let mut pre_extended_right = vec![0_f64; half_time_frame];
    pre_extended_left.extend(left_channel);
    pre_extended_right.extend(right_channel);

    // For fft_size, there's different opinions online on how much zero-padding is needed
    let mut fft_size = rounded_time_frame.next_power_of_two(); // Round to next power of 2 for some zero-padding and for a fast FFT
    if (fft_size as f64) < 1.5_f64 * (rounded_time_frame as f64) {
        // Ensure fft_size is at least 150% of rounded_time_frame
        fft_size = fft_size.saturating_mul(2); // Move to the next power of two
    }
    pre_extended_left.extend(vec![0_f64; fft_size]); // Allow for last FFT chunk to be added
    pre_extended_right.extend(vec![0_f64; fft_size]);
    let extended_left = pre_extended_left.into_boxed_slice();
    let extended_right = pre_extended_right.into_boxed_slice();
    let extended_length = extended_left.len();

    // Lots of Vecs are used here to reuse memory space instead of reallocating
    // `.into_boxed_slice()` is here to prevent overallocation if it stayed as a Vec
    let r2c = planner.plan_fft_forward(fft_size);
    let c2r = planner.plan_fft_inverse(fft_size);
    let mut left_chunk = Vec::with_capacity(fft_size);
    let mut right_chunk = Vec::with_capacity(fft_size);
    let mut left_complex = r2c.make_output_vec().into_boxed_slice();
    let mut right_complex = r2c.make_output_vec().into_boxed_slice();
    let mut scratch_complex = r2c.make_scratch_vec().into_boxed_slice();
    let mut scratch_real = c2r.make_scratch_vec().into_boxed_slice();
    let mut holding_left = vec![0_f64; extended_length].into_boxed_slice();
    let mut holding_right = vec![0_f64; extended_length].into_boxed_slice();
    let mut holding_position = 0_usize;

    let window = window(rounded_time_frame);

    // Windows need a bunch of hops
    // Doing more chunks will help with clicking/zipper noise, but will increase runtime
    // Sorta acts like anti-aliasing in a way
    // Numerator should be 1_f64 for the mininum amount of overlaps, currently doing 16x
    let hop_time_frame = 0.0625_f64 / (WINDOW_COSINES.len() as f64 + 1_f64);
    let hop_size = f64::max((time_frame * hop_time_frame).round_ties_even(), 1.0) as usize;

    // Up until the end, which should be basically a half-window
    while holding_position <= usize::strict_sub(extended_length, rounded_time_frame) {
        // .copy_from_*() cannot be used here
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

        izip!(
            left_chunk.iter_mut(),
            right_chunk.iter_mut(),
            window.iter().copied()
        )
        .for_each(|(left_samp, right_samp, window_mult)| {
            *left_samp *= window_mult;
            *right_samp *= window_mult;
        });

        left_chunk.resize(fft_size, 0_f64);
        right_chunk.resize(fft_size, 0_f64);

        _ = r2c.process_with_scratch(&mut left_chunk, &mut left_complex, &mut scratch_complex);
        _ = r2c.process_with_scratch(&mut right_chunk, &mut right_complex, &mut scratch_complex);

        // Remove DC offset for a better overlap
        if let Some(dc) = left_complex.first_mut() {
            *dc = Complex::ZERO;
        }
        if let Some(dc) = right_complex.first_mut() {
            *dc = Complex::ZERO;
        }
        // Skip first/DC bin
        izip!(left_complex.iter_mut(), right_complex.iter_mut())
            .skip(1)
            .for_each(|(left_point, right_point)| {
                align(left_point, right_point);
            });

        // left_chunk and right_chunk will be overwritten
        _ = c2r.process_with_scratch(&mut left_complex, &mut left_chunk, &mut scratch_real);
        _ = c2r.process_with_scratch(&mut right_complex, &mut right_chunk, &mut scratch_real);

        // RealFFT, which uses RustFFT, amplifies the signal by fft_size
        // Normalization happens later in processing.rs

        // left_chunk and right_chunk need to be emptied to allow .extend() on the next iteration
        // Doing .iter() then .clear() is faster than .drain(..)
        izip!(
            izip!(holding_left.iter_mut(), holding_right.iter_mut()).skip(holding_position),
            left_chunk.iter(),
            right_chunk.iter()
        )
        .for_each(|((hold_left, hold_right), left_samp, right_samp)| {
            *hold_left += left_samp;
            *hold_right += right_samp;
        });
        left_chunk.clear();
        right_chunk.clear();

        holding_position = usize::strict_add(holding_position, hop_size);
    }

    // Overlap-adding amplifies the signal by (WINDOW_COSINES.len() as f64 + 1_f64) or 1/hop_time_frame
    // Normalization happens later in processing.rs

    (
        holding_left
            .into_iter()
            .skip(half_time_frame)
            .take(original_length)
            .collect(),
        holding_right
            .into_iter()
            .skip(half_time_frame)
            .take(original_length)
            .collect(),
    )
}
