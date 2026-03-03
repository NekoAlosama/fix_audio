use core::{f64::consts::TAU, iter::once};
use std::sync::Mutex;

use rayon::iter::{IntoParallelIterator as _, ParallelIterator as _};
use realfft::{RealFftPlanner, num_complex::Complex};

#[cfg(feature = "final_rotation")]
use crate::processing;
#[cfg(feature = "final_rotation")]
use core::f64::consts::PI;

/// List of cosine coefficients of window function.
///
/// Taken from <https://holometer.fnal.gov/GH_FFT.pdf>.
// Ideal candidate is probably HFT144D, a flat top window which needs 7 overlaps and has a noise floor of -144.1dB,
//   almost enough for 24-bit integer audio, definitely good enough for the humman auditory system
// Note that the large (normalized) effective noise bandwidth just indicates that the resulting FFT output is scaled up by some number
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
    let f64_rate_recip = (time_frame as f64 - 1_f64).recip(); // N - 1 for symmetric window
    // The actual level of the window doesn't really matter
    (0..time_frame)
        .into_par_iter()
        .map(|n| {
            WINDOW_COSINES
                .into_iter()
                .fold(1_f64, |acc, (internal, external)| {
                    external.mul_add(f64::cos(internal * n as f64 * f64_rate_recip), acc)
                })
        })
        .collect()
}

/// Returns an FFT size in the form of `2^n * 3^m` since `rustfft` claims to work the fastest on these types.
fn get_fft_size(length: usize) -> usize {
    // LN_2
    let ln_3 = 3_f64.ln();
    let ln_6 = 6_f64.ln();

    let float_length = length as f64;
    let pow_of_2 = float_length.log2().ceil().exp2().round_ties_even();
    let pow_of_3 = 3_f64.powi(f64::ceil(float_length.ln() / ln_3) as i32);
    let pow_of_6 = 6_f64.powi(f64::ceil(float_length.ln() / ln_6) as i32);

    pow_of_2.min(pow_of_3).min(pow_of_6) as usize
}

/// Faster `.is_finite()` check compared to `num_traits`.
fn is_finite(point: Complex<f64>) -> bool {
    point.re.abs() < f64::INFINITY && point.im.abs() < f64::INFINITY
}

/// Faster `.norm_sqr()` calculation compared to `num_complex`.
fn norm_sqr(point: Complex<f64>) -> f64 {
    point.re.mul_add(point.re, point.im * point.im)
}

/// Aligns the phase angle of the left and right channels.
// According to Intel VTune Profiler, this is the hottest function since it's in a hot loop.
// I've tried a branchless and an SIMD version, but they pretty much compile to the same peformance.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "clippy thinks the operations done on Complex<f64> are for integers"
)]
fn align(original_left: &mut Complex<f64>, original_right: &mut Complex<f64>) {
    let left_norm_sqr = norm_sqr(*original_left);
    let right_norm_sqr = norm_sqr(*original_right);

    // Make the quieter channel a scaled-down copy of the louder channel
    if left_norm_sqr >= right_norm_sqr {
        // If the left channel is louder, the right channel should have the same angle as the left
        let new_right = *original_left * f64::sqrt(right_norm_sqr / left_norm_sqr); // This division is probably taking up the most time. Unsure how to fix that

        if is_finite(new_right) {
            *original_right = new_right;
        }
    } else {
        let new_left = *original_right * f64::sqrt(left_norm_sqr / right_norm_sqr);

        if is_finite(new_left) {
            *original_left = new_left;
        }
    }
}

/// An STFT.
// Memory usage: more than four times the size of the result (f64 import -> slightly longer f64 import + f64 holding for longer import -> f32 export later on)
pub fn overlapping_fft(
    realfft_planner: &mut RealFftPlanner<f64>,
    time_frame: f64,
    left_channel: Box<[f64]>,
    right_channel: Box<[f64]>,
) -> (Box<[f64]>, Box<[f64]>) {
    // Idea is that time_frame gives us the amount of samples (possibly fractional) that we need to FFT
    let rounded_time_frame = time_frame.round_ties_even() as usize;
    // We should pad with half-a-second of silence to allow for half-windows at the beginning and end
    let half_time_frame = (time_frame * 0.5_f64).round_ties_even() as usize;

    // Since we're using a flat-top window and doing overlaps, we can probably assume that there should be no zero-padding aside from what's needed for a fast FFT.
    // Plus, adding more padding will increase the runtime non-linearly. In this case, it's probably better to add more overlaps instead, since that is linear.
    let fft_size = get_fft_size(rounded_time_frame);

    // We need a bit of silence at the beginning
    // This consumes left_channel and right_channel
    let extended_left = vec![0_f64; half_time_frame]
        .into_iter()
        .chain(left_channel)
        .collect::<Box<[f64]>>();
    let extended_right = vec![0_f64; half_time_frame]
        .into_iter()
        .chain(right_channel)
        .collect::<Box<[f64]>>();
    let extended_length = extended_left.len();

    // `.into_boxed_slice()` is here to prevent overallocation if it stayed as a Vec
    let r2c = realfft_planner.plan_fft_forward(fft_size);
    let c2r = realfft_planner.plan_fft_inverse(fft_size);
    let holding_left = Mutex::new(vec![0_f64; extended_length].into_boxed_slice());
    let holding_right = Mutex::new(vec![0_f64; extended_length].into_boxed_slice());
    let window = window(rounded_time_frame);

    // Windows need a bunch of hops.
    // A zipper noise is heard below ~16x the required overlaps, max should be 256x since we should have at least 1.0 sample per hop at 44.1khz and MIN_FREQ=20.0
    // Noise likely comes from the fact that phase isn't smoothed with the previous frame, but that would require this run in serial instead of parallel
    // Thus, we'll just overlap a lot so inconsistencies are smoothed out.
    let hop_indexes = {
        let mut pre_hop_indexes = vec![0_usize];
        let hop_size = (time_frame / (WINDOW_COSINES.len() as f64 + 1_f64)) / 32_f64; // 32x seems good enough, at least this increases runtime proportionally
        let mut pre_candidate = hop_size;
        while let hop_candidate = pre_candidate.round_ties_even() as usize
            && hop_candidate < extended_length
        {
            pre_hop_indexes.push(hop_candidate);
            pre_candidate += hop_size; // pre_candidate should probably increase by at least 1 at 44.1khz with MIN_FREQ=20
        }
        pre_hop_indexes.into_par_iter()
    };

    // Function moved due to Clippy lint
    let multiply_and_pad = |channel: &[f64], hold_pos: usize| {
        channel
            .iter()
            .skip(hold_pos)
            .take(rounded_time_frame)
            .zip(window.iter())
            .map(|(&samp, &mult)| samp * mult)
            .chain(once(0_f64).cycle()) // Extend iterator by cycling 0
            .take(fft_size)
            .collect::<Box<[f64]>>()
    };
    // Up until the end, which should be basically a half-window
    hop_indexes.for_each(|holding_position| {
        // Surprisingly, these don't take much memory per thread
        let mut left_chunk = multiply_and_pad(&extended_left, holding_position);
        let mut right_chunk = multiply_and_pad(&extended_right, holding_position);

        let mut scratch = c2r.make_scratch_vec().into_boxed_slice();

        let mut left_complex = {
            let mut pre_left_complex = r2c.make_output_vec().into_boxed_slice();
            _ = r2c.process_with_scratch(&mut left_chunk, &mut pre_left_complex, &mut scratch);
            pre_left_complex
        };
        let mut right_complex = {
            let mut pre_right_complex = r2c.make_output_vec().into_boxed_slice();
            _ = r2c.process_with_scratch(&mut right_chunk, &mut pre_right_complex, &mut scratch);
            pre_right_complex
        };

        left_complex
            .iter_mut()
            .zip(right_complex.iter_mut())
            .for_each(|(left_point, right_point)| {
                align(left_point, right_point);
            });

        // left_chunk and right_chunk will be overwritten
        _ = c2r.process_with_scratch(&mut left_complex, &mut left_chunk, &mut scratch);
        drop(left_complex);
        _ = c2r.process_with_scratch(&mut right_complex, &mut right_chunk, &mut scratch);
        drop(right_complex);
        drop(scratch);

        // RealFFT, which uses RustFFT, amplifies the signal by fft_size
        // Normalization happens later in processing.rs

        // left_chunk is done first so less time is used locking the mutexes
        left_chunk
            .into_iter()
            .zip(
                holding_left
                    .lock()
                    .expect("Critical thread was killed")
                    .iter_mut()
                    .skip(holding_position),
            )
            .for_each(|(left_samp, hold_left)| *hold_left += left_samp);

        right_chunk
            .into_iter()
            .zip(
                holding_right
                    .lock()
                    .expect("Critical thread was killed")
                    .iter_mut()
                    .skip(holding_position),
            )
            .for_each(|(right_samp, hold_right)| *hold_right += right_samp);
    });

    // Overlap-adding amplifies the signal by (WINDOW_COSINES.len() as f64 + 1_f64) or 1/hop_time_frame
    // Normalization happens later in processing.rs

    (
        holding_left
            .into_inner()
            .expect("Critical thread was killed")
            .into_iter() // Don't think doing .into_par_iter() does anything
            .skip(half_time_frame)
            .collect(),
        holding_right
            .into_inner()
            .expect("Critical thread was killed")
            .into_iter()
            .skip(half_time_frame)
            .collect(),
    )
}

#[cfg(feature = "final_rotation")]
/// Minimize peaks by interpreting the analytic signal as a polygon and finding the rotating angle that will minimize the width on the real axis.
// Memory usage: a bit more than STFT's usage
// Luckily faster than STFT above
pub fn minimize_peak(left_channel: Box<[f64]>, right_channel: Box<[f64]>) -> processing::AudioPeak {
    let f32_left_channel;
    let f32_right_channel;

    // Get a 90-degree rotated version of the audio signal to make it easier to rotate
    // We have to renormalize everything since we already did that in the previous processing.rs step (unless I want to do it again?)
    let (f32_rotated_left, f32_rotated_right) = {
        let original_length = left_channel.len();
        let fft_size = get_fft_size(original_length);
        // The planner has an internal cache to store different FFT sizes. This makes sense for reusing songs with the same sample rate as in the STFT,
        //   but each song probably have different lengths to each other, so it's more efficient to make a specific FFT planner for each song
        let mut long_realfft_planner = RealFftPlanner::new();
        let r2c = long_realfft_planner.plan_fft_forward(fft_size);
        let c2r = long_realfft_planner.plan_fft_inverse(fft_size);
        let mut scratch = c2r.make_scratch_vec().into_boxed_slice();

        // Also convert to f32 to reduce memory usage
        // Also pre-normalize since this will usually cause problems in the f32 area
        let fft_norm = (fft_size as f64).recip();

        // Function reduced
        let mut compute_rotated_channel = |channel: &[f64]| {
            let mut complex = r2c.make_output_vec().into_boxed_slice();
            _ = r2c.process_with_scratch(
                &mut channel
                    .iter()
                    .cycle() // Pad signal by cycling it
                    .take(fft_size)
                    .map(|samp| (samp * fft_norm) as f32)
                    .collect::<Box<[f32]>>(),
                &mut complex,
                &mut scratch,
            );

            complex
                .iter_mut()
                .for_each(|point| *point = Complex::new(-point.im, point.re)); // Equivalent to multiplying by i
            let mut finished = c2r.make_output_vec().into_boxed_slice();
            _ = c2r.process_with_scratch(&mut complex, &mut finished, &mut scratch);
            finished.into_iter()
        };
        let rotated_left = compute_rotated_channel(&left_channel);
        f32_left_channel = left_channel.into_iter().map(|samp| samp as f32); // Reduce memory by mapping f64's to f32's now instead of later
        let rotated_right = compute_rotated_channel(&right_channel);
        f32_right_channel = right_channel.into_iter().map(|samp| samp as f32);
        (rotated_left, rotated_right)
    };

    let analytic_left = f32_left_channel
        .zip(f32_rotated_left)
        .map(|(left, rot_left)| Complex::new(left, rot_left))
        .collect::<Box<[Complex<f32>]>>();
    let analytic_right = f32_right_channel
        .zip(f32_rotated_right)
        .map(|(right, rot_right)| Complex::new(right, rot_right))
        .collect::<Box<[Complex<f32>]>>();

    // Since the multiplications take a long time to compute, the best way for me to get a good estimate would be by sampling in fixed intervals.
    // 1 to 31 since 0 or 32 will be the original level.
    // Seems good enough, actual best peak value is probably lower by 0.5dB or less
    // Unfortunately also makes the track `ReplayGain` slightly inaccurate (usually no change, but sometimes 0.15dB change or less).
    // No idea why this happens, but it could just be that the EBU R 128 loudness estimate does change with phase rotations.
    let mut saved_angle = Complex::new(1_f32, 0_f32);
    let mut saved_peak = analytic_left
        .iter()
        .chain(analytic_right.iter())
        .fold(f32::NEG_INFINITY, |acc, point| {
            f32::max(acc, point.re.abs())
        });
    let candidate_angle = (1_i32..=31_i32).map(|numerator| {
        let (sine, cosine) = f64::sin_cos(f64::from(numerator) * PI / 32_f64);
        Complex::new(cosine as f32, -sine as f32)
    });
    // Short-circuting loop
    for test_angle in candidate_angle {
        let mut good_angle = true;
        let mut local_max_peak = f32::NEG_INFINITY;
        for point in analytic_left.iter().chain(analytic_right.iter()) {
            let point_peak = point
                .re
                .mul_add(test_angle.re, point.im * test_angle.im)
                .abs();
            if point_peak == f32::INFINITY {
                // Ideally, the playback system is smart enough not to play an INFINITY or NEG_INFINITY sample
                continue;
            }

            if point_peak > saved_peak {
                good_angle = false;
                break;
            } else if point_peak > local_max_peak {
                local_max_peak = point_peak;
            } else {
                // point_peak wasn't that high, no reason to save it
            }
        }
        if good_angle {
            saved_angle = test_angle;
            saved_peak = local_max_peak;
        }
    }

    (
        (
            analytic_left
                .into_par_iter()
                .map(|point| point.re.mul_add(saved_angle.re, point.im * saved_angle.im))
                .collect(),
            analytic_right
                .into_par_iter()
                .map(|point| point.re.mul_add(saved_angle.re, point.im * saved_angle.im))
                .collect(),
        ),
        saved_peak,
    )
}
