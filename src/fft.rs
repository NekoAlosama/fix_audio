use core::{
    f64::consts::{PI, TAU},
    iter::once,
};
use std::sync::Mutex;

use rayon::iter::{IntoParallelIterator as _, ParallelIterator as _};
use realfft::{RealFftPlanner, num_complex::Complex};

use crate::processing;

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
    let f64_rate_recip = 1_f64 / (time_frame as f64);
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

/// Aligns the phase angle of the left and right channels.
// According to Intel VTune Profiler, this is the hottest function since it's in a hot loop.
// I've tried a branchless and an SIMD version, but they pretty much compile to the same peformance.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "clippy thinks the operations done on Complex<f64> are for integers"
)]
fn align(original_left: &mut Complex<f64>, original_right: &mut Complex<f64>) {
    //
    let left_norm_sqr = original_left
        .re
        .mul_add(original_left.re, original_left.im * original_left.im);
    let right_norm_sqr = original_right
        .re
        .mul_add(original_right.re, original_right.im * original_right.im);

    // Make the quieter channel a scaled-down copy of the louder channel
    if left_norm_sqr >= right_norm_sqr {
        // If the left channel is louder, the right channel should have the same angle as the left
        let new_right = *original_left * f64::sqrt(right_norm_sqr / left_norm_sqr); // This division is probably taking up the most time. Unsure how to fix that

        if new_right.re.abs() < f64::INFINITY && new_right.im.abs() < f64::INFINITY {
            *original_right = new_right;
        }
    } else {
        let new_left = *original_right * f64::sqrt(left_norm_sqr / right_norm_sqr);

        if new_left.re.abs() < f64::INFINITY && new_left.im.abs() < f64::INFINITY {
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

    // For fft_size, there's different opinions online on how much zero-padding is needed
    // Circular artifacts? Only useful in spectrograms?
    let fft_size = {
        let pre_fft_size = get_fft_size(rounded_time_frame); // Round to next power of 2 for some zero-padding and for a fast FFT
        if (pre_fft_size as f64) >= 1.5_f64 * (rounded_time_frame as f64) {
            pre_fft_size
        } else {
            // Ensure fft_size is at least 150% of rounded_time_frame
            pre_fft_size.saturating_mul(2) // Move to the next power of two
        }
    };

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

    // Windows need a bunch of hops
    // Doing more chunks will help with clicking/zipper noise, but will increase runtime
    // Sorta acts like anti-aliasing in a way
    // Numerator should be 1_f64 for the mininum amount of overlaps, currently doing at least 16x (low difference between 16x and 32x) and at most 256x (at least 1 sample with 44.1khz and MIN_FREQ=20_f64)
    // NOTE: possible reason for zipper noise is that the window isn't overlapping correctly/exactly.
    //   alternatively, it's just a result of having discrete samples, so the only solution is this anti-aliasing-type solution
    // TODO: check if upsampling first then downsampling the result is better
    let hop_indexes = {
        let mut pre_hop_indexes = vec![0_usize];
        let hop_size = (time_frame / (WINDOW_COSINES.len() as f64 + 1_f64)) / 32_f64; // Minor spectrogram difference with 16x
        let mut pre_candidate = hop_size;
        while let hop_candidate = pre_candidate.round_ties_even() as usize
            && hop_candidate < extended_length
        {
            pre_hop_indexes.push(hop_candidate);
            pre_candidate += hop_size; // pre_candidate should probably increase by at least 1 at 44.1khz with MIN_FREQ=20
        }
        pre_hop_indexes.into_par_iter()
    };

    // Up until the end, which should be basically a half-window
    hop_indexes.for_each(|holding_position| {
        // Surprisingly, these don't take much memory per thread
        let mut left_chunk = extended_left
            .iter()
            .skip(holding_position)
            .take(rounded_time_frame)
            .zip(window.iter())
            .map(|(&samp, &mult)| samp * mult)
            .chain(once(0_f64).cycle()) // Extend iterator by cycling 0
            .take(fft_size)
            .collect::<Box<_>>();
        let mut right_chunk = extended_right
            .iter()
            .skip(holding_position)
            .take(rounded_time_frame)
            .zip(window.iter())
            .map(|(&samp, &mult)| samp * mult)
            .chain(once(0_f64).cycle())
            .take(fft_size)
            .collect::<Box<_>>();

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

        // Remove DC offset for a better overlap
        left_complex[0] = Complex::ZERO;
        right_complex[0] = Complex::ZERO;

        // Skip first/DC bin
        left_complex
            .iter_mut()
            .zip(right_complex.iter_mut())
            .skip(1)
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

        // TODO: check if we need to remove the DC from the IFFT
        //   setting the DC bin to zero removes most DC, but not all, and I don't know if all of it needs to be removed since it could just be truly low-frequency noise
        processing::plain_remove_dc(&mut left_chunk);
        processing::plain_remove_dc(&mut right_chunk);

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

        // Might be able to reduce these to a function
        let rotated_left = {
            let mut left_complex = r2c.make_output_vec().into_boxed_slice();
            _ = r2c.process_with_scratch(
                &mut left_channel
                    .iter()
                    .cycle() // Pad signal by cycling it
                    .take(fft_size)
                    .map(|samp| (samp * fft_norm) as f32)
                    .collect::<Box<_>>(),
                &mut left_complex,
                &mut scratch,
            );
            f32_left_channel = left_channel
                .into_iter()
                .map(|samp| samp as f32)
                .collect::<Box<_>>();

            left_complex
                .iter_mut()
                .for_each(|point| *point = Complex::new(-point.im, point.re)); // Equivalent to multiplying by i
            let mut finished_left = c2r.make_output_vec().into_boxed_slice();
            _ = c2r.process_with_scratch(&mut left_complex, &mut finished_left, &mut scratch);
            finished_left
                .into_iter()
                .take(original_length)
                .collect::<Box<_>>()
        };
        let rotated_right = {
            let mut right_complex = r2c.make_output_vec().into_boxed_slice();
            _ = r2c.process_with_scratch(
                &mut right_channel
                    .iter()
                    .cycle() // Pad signal by cycling it
                    .take(fft_size)
                    .map(|samp| (samp * fft_norm) as f32)
                    .collect::<Box<_>>(),
                &mut right_complex,
                &mut scratch,
            );
            f32_right_channel = right_channel
                .into_iter()
                .map(|samp| samp as f32)
                .collect::<Box<_>>();

            right_complex
                .iter_mut()
                .for_each(|point| *point = Complex::new(-point.im, point.re));
            let mut finished_right = c2r.make_output_vec().into_boxed_slice();
            _ = c2r.process_with_scratch(&mut right_complex, &mut finished_right, &mut scratch);
            finished_right
                .into_iter()
                .take(original_length)
                .collect::<Box<_>>()
        };
        (rotated_left, rotated_right)
    };

    let analytic_left = f32_left_channel
        .into_iter()
        .zip(f32_rotated_left)
        .map(|(left, rot_left)| Complex::new(left, rot_left))
        .collect::<Box<_>>();
    let analytic_right = f32_right_channel
        .into_iter()
        .zip(f32_rotated_right)
        .map(|(right, rot_right)| Complex::new(right, rot_right))
        .collect::<Box<_>>();

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
