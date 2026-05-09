use core::{f64::consts::TAU, hint, iter::once};
#[cfg(feature = "final_rotation")]
use std::sync::Mutex;

use rayon::iter::{IntoParallelIterator as _, ParallelIterator as _};
use realfft::{RealFftPlanner, num_complex::Complex};

#[cfg(feature = "final_rotation")]
use crate::processing;
#[cfg(feature = "final_rotation")]
use core::f32::consts::PI;

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

/// Base-2 logarithm of 3 for `get_full_fft_size()` for an arbitrary FFT size.
const LB_3: f32 = 1.584_962_5;

/// Windowing is used to make the signal chunk fade in and out
///   to prevent discontinuities, which causes spectral leakage (noise tuned to the music).
// This is a periodic window, where the first point is zero and the ending point is non-zero, with the idea that the next window has the next zero point.
// Choosing a symmetric window with both or neither endpoints being zero doesn't seem to matter much here.
fn window(time_frame: usize) -> Box<[f32]> {
    let f64_rate_recip = (time_frame as f64).recip();
    // The actual level of the window doesn't really matter
    (0..time_frame)
        .into_par_iter()
        .map(|n| {
            WINDOW_COSINES
                .into_iter()
                .fold(1_f64, |acc, (internal, external)| {
                    external.mul_add(f64::cos(internal * n as f64 * f64_rate_recip), acc)
                }) as f32
        })
        .collect()
}

/// Returns an STFT frame size in the form of `2^n * 3^m` since `rustfft` claims to work the fastest on these types.
fn get_stft_frame_size(length: usize) -> usize {
    match length {
        2_205 => 2_304,                 // 44100 / 20
        2_400 => 2_592,                 // 48000 / 20
        4_410 => 4_608,                 // 88200 / 20
        4_800 => 5_184,                 // 96000 / 20
        8_820 => 9_216,                 // 176400 / 20
        9_600 => 10_368, // 192000 / 20, probably the highest sample rate still in distribution (e.g. Frank Zappa's 'Hot Rats' on Qobuz)
        17_640 => 18_432, // 352800 / 20, VVV
        19_200 => 19_683, // 384000 / 20, These rates are here if the length is doubled for more zero-padding
        35_280 => 36_864, // 705600 / 20, ^^^
        38_400 => 39_366, // 768000 / 20, ^^^
        _ => get_full_fft_size(length), // Better if your code called `get_full_fft_size` directly instead of using `get_stft_frame_size`
    }
}

/// Returns an FFT size in the form of `2^n * 3^m` since `rustfft` claims to work the fastest on these types.
fn get_full_fft_size(length: usize) -> usize {
    let lb_length = f32::log2(length as f32);
    let mut saved_pow_of_2 = 0;
    let mut saved_pow_of_3 = 0;
    let mut lowest_remainder = 1.0_f32;

    let max_pow_of_3 = f32::ceil(lb_length / LB_3) as u32;

    for test_pow_of_3 in 0..=max_pow_of_3 {
        let criteria = (-LB_3).mul_add(test_pow_of_3 as f32, lb_length); // lb_length - saved_pow_of_3 * LB_3 
        let test_remainder = f32::ceil(criteria) - criteria;

        if test_remainder < lowest_remainder {
            lowest_remainder = test_remainder;
            saved_pow_of_2 = f32::ceil(criteria) as u32;
            saved_pow_of_3 = test_pow_of_3;
        }
    }

    2_usize
        .saturating_pow(saved_pow_of_2)
        .saturating_mul(3_usize.saturating_pow(saved_pow_of_3))
}

/// Faster `.norm_sqr()` calculation compared to `num_complex`.
fn norm_sqr(point: Complex<f32>) -> f32 {
    point.re.mul_add(point.re, point.im * point.im)
}

/// Aligns the phase angle of the left and right channels.
// According to Intel VTune Profiler, this is the hottest function since it's in a hot loop.
// I've tried a branchless and an SIMD version, but they pretty much compile to the same peformance.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "clippy thinks the operations done on Complex<f32> are for integers"
)]
fn align(original_left: &mut Complex<f32>, original_right: &mut Complex<f32>) {
    let align = *original_left + *original_right;

    // The only time that a problem would occur is in this division by an f32 that could be near or at 0_f32.
    let inv_align_norm_sqr = 1_f32 / norm_sqr(align);
    if inv_align_norm_sqr.is_finite() {
        let left_norm_sqr = norm_sqr(*original_left);
        let right_norm_sqr = norm_sqr(*original_right);

        *original_left = align * f32::sqrt(left_norm_sqr * inv_align_norm_sqr);
        *original_right = align * f32::sqrt(right_norm_sqr * inv_align_norm_sqr);
    } else {
        // If inv_align_norm_sqr is infinite or NaN, then *original_left is approximately equal to -*original_right,
        //     so we just invert the right channel since the actual solution is undefined.
        // This path becomes hot if the signal contains silence or simple waves (i.e. sine, triangle, etc., since many frequencies could be 0)
        *original_right = -*original_right;
    }
}

/// STFT that, in each frame, aligns each frequency to the louder channel's phase angle.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "clippy thinks the operations done on Complex<f32> are for integers"
)]
pub fn overlapping_fft(
    realfft_planner: &mut RealFftPlanner<f32>,
    time_frame: f32,
    left_channel: Box<[f32]>,
    right_channel: Box<[f32]>,
) -> (Box<[f32]>, Box<[f32]>) {
    // Idea is that time_frame gives us the amount of samples (possibly fractional) that we need to FFT
    let rounded_time_frame = time_frame.round_ties_even() as usize;
    // We should pad with half-a-second of silence to allow for half-windows at the beginning and end
    let half_time_frame = (time_frame * 0.5_f32).round_ties_even() as usize;

    // Since we're using a flat-top window and doing overlaps, we can probably assume that there should be no zero-padding aside from what's needed for a fast FFT.
    // Plus, adding more padding will increase the runtime non-linearly. In this case, it's probably better to add more overlaps instead, since that is linear.
    let fft_size = get_stft_frame_size(rounded_time_frame);
    let fft_sqrt_norm = f32::sqrt(1_f32 / fft_size as f32);

    // We need a bit of silence at the beginning
    // This consumes left_channel and right_channel
    let build = |channel: Box<[f32]>| {
        vec![0_f32; half_time_frame]
            .into_iter()
            .chain(channel)
            .collect::<Box<[f32]>>()
    };
    let extended_left = build(left_channel);
    let extended_right = build(right_channel);
    let extended_length = extended_left.len();

    // `.into_boxed_slice()` is here to prevent overallocation if it stayed as a Vec
    let r2c = realfft_planner.plan_fft_forward(fft_size);
    let c2r = realfft_planner.plan_fft_inverse(fft_size);
    let holding_left = Mutex::new(vec![0_f32; extended_length].into_boxed_slice());
    let holding_right = Mutex::new(vec![0_f32; extended_length].into_boxed_slice());
    let window = window(rounded_time_frame);

    // Windows need a bunch of hops.
    // A zipper noise is heard without more overlaps, likely because of phase discontinuities between frames. Fixing this would require dependence on the previous frame.
    // This dependence would then require the function to run in serial instead of parallel, so we'll just do a lot of overlaps to smooth these out.
    // More overlaps means more discontinuities, but at a reduced amplitude, so it's like a high-frequency noise that's increasing in pitch but decreasing in volume.v
    let hop_size = time_frame * 0.001_f32; // equal to sample_rate/20_000, so ideally the generated noise is at 20khz and thus inaudible
    let hop_indexes = {
        let max_index = (extended_length as f32 / hop_size).round_ties_even() as usize;

        (0..max_index)
            .into_par_iter()
            .map(|index| (index as f32 * hop_size).round_ties_even() as usize)
    };

    // Function moved due to Clippy lint
    let multiply_and_pad = |channel: &[f32], hold_pos: usize| {
        channel
            .iter()
            .skip(hold_pos)
            .take(rounded_time_frame)
            .zip(window.iter())
            .map(|(&samp, &mult)| samp * mult)
            .chain(once(0_f32).cycle()) // Extend iterator by cycling 0
            .take(fft_size)
            .collect::<Box<[f32]>>()
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

        #[cfg(feature = "subsonic_removal")]
        {
            // The first two elements correspond to the 0Hz and 20Hz frequencies, where inbetween frequencies are rounded to the nearest one.
            // As a crude low-cut/high-pass filter, the 0Hz frequency is removed, thus removing frequencies from 0Hz to 10Hz, randomly inclusive or exclusive.
            left_complex[0] = Complex::ZERO;
            right_complex[0] = Complex::ZERO;
        }

        left_complex
            .iter_mut()
            .zip(right_complex.iter_mut())
            .for_each(|(left_point, right_point)| {
                align(left_point, right_point);
                *left_point *= fft_sqrt_norm; // Do normalization after so the above points are using larger numbers
                *right_point *= fft_sqrt_norm;
            });

        // left_chunk and right_chunk will be overwritten
        _ = c2r.process_with_scratch(&mut left_complex, &mut left_chunk, &mut scratch);
        drop(left_complex);
        _ = c2r.process_with_scratch(&mut right_complex, &mut right_chunk, &mut scratch);
        drop(right_complex);
        drop(scratch);

        // RealFFT, which uses RustFFT, amplifies the signal by fft_size
        // Normalization happens later in processing.rs

        let add_to_hold = |chunk: Box<[f32]>, hold_channel: &Mutex<Box<[f32]>>| {
            chunk
                .into_iter()
                .zip(
                    hold_channel
                        .lock()
                        .expect("Critical thread was killed")
                        .iter_mut()
                        .skip(holding_position),
                )
                .for_each(|(new_samp, hold_samp)| {
                    *hold_samp = new_samp.mul_add(fft_sqrt_norm, *hold_samp);
                });
        };

        // left_chunk is done first so less time is used locking the mutexes
        add_to_hold(left_chunk, &holding_left);
        add_to_hold(right_chunk, &holding_right);
    });

    // Overlap-adding amplifies the signal by (WINDOW_COSINES.len() as f32 + 1_f32) or 1/hop_time_frame
    // Normalization happens later in processing.rs

    let collect = |channel: Mutex<Box<[f32]>>| {
        channel
            .into_inner()
            .expect("Critical thread was killed")
            .into_iter() // Don't think doing .into_par_iter() does anything
            .skip(half_time_frame)
            .collect()
    };
    (collect(holding_left), collect(holding_right))
}

#[cfg(feature = "final_rotation")]
/// Minimize peaks by interpreting the analytic signal as a polygon and finding the rotating angle that will minimize the width on the real axis.
// Memory usage: a bit more than STFT's usage
// Luckily faster than STFT above
#[expect(
    clippy::arithmetic_side_effects,
    reason = "clippy thinks the operations done on Complex<f32> are for integers"
)]
pub fn minimize_peak(left_channel: Box<[f32]>, right_channel: Box<[f32]>) -> processing::AudioPeak {
    let fft_sqrt_norm;

    // Get a 90-degree rotated version of the audio signal to make it easier to rotate
    // We have to renormalize everything since we already did that in the previous processing.rs step (unless I want to do it again?)
    let (rotated_left, rotated_right) = {
        let original_length = left_channel.len();
        let fft_size = get_full_fft_size(original_length);
        // The planner has an internal cache to store different FFT sizes. This makes sense for reusing songs with the same sample rate as in the STFT,
        //   but each song probably have different lengths to each other, so it's more efficient to make a specific FFT planner for each song
        let mut long_realfft_planner = RealFftPlanner::new();
        let r2c = long_realfft_planner.plan_fft_forward(fft_size);
        let c2r = long_realfft_planner.plan_fft_inverse(fft_size);
        let mut scratch = c2r.make_scratch_vec().into_boxed_slice();

        fft_sqrt_norm = f32::sqrt(1_f32 / fft_size as f32);

        // Function reduced
        let mut compute_rotated_channel = |channel: &[f32]| {
            let mut complex = r2c.make_output_vec().into_boxed_slice();
            _ = r2c.process_with_scratch(
                &mut channel
                    .iter()
                    .cycle() // Pad signal by cycling it
                    .take(fft_size)
                    .copied()
                    .collect::<Box<[f32]>>(),
                &mut complex,
                &mut scratch,
            );

            complex
                .iter_mut()
                .for_each(|point| *point = Complex::new(-point.im, point.re) * fft_sqrt_norm); // Equivalent to multiplying by i
            let mut finished = c2r.make_output_vec().into_boxed_slice();
            _ = c2r.process_with_scratch(&mut complex, &mut finished, &mut scratch);
            finished.into_iter()
        };
        (
            compute_rotated_channel(&left_channel).map(|samp| samp * fft_sqrt_norm),
            compute_rotated_channel(&right_channel).map(|samp| samp * fft_sqrt_norm),
        )
    };

    let analytic_left = left_channel
        .into_iter()
        .zip(rotated_left)
        .map(|(left, rot_left)| Complex::new(left, rot_left))
        .collect::<Box<[Complex<f32>]>>();
    let analytic_right = right_channel
        .into_iter()
        .zip(rotated_right)
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
        let (sine, cosine) = f32::sin_cos(numerator as f32 * PI / 32_f32);
        Complex::new(cosine, -sine)
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
                hint::cold_path();
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
