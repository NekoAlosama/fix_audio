use core::f64::consts::TAU;
use std::sync::Mutex;

use itertools::izip;
use rayon::iter::{IndexedParallelIterator as _, IntoParallelIterator as _, ParallelIterator as _};
use realfft::{RealFftPlanner, num_complex::Complex};

/// List of cosine coefficients of window function.
///
/// Taken from <https://holometer.fnal.gov/GH_FFT.pdf>.
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

/// Aligns the phase angle of the left and right channels.
/// This attempts to use branchless programming, but benchmarks can't be consistent due to I/O calls.
#[expect(
    clippy::arithmetic_side_effects,
    reason = "clippy thinks the operations done on Complex<f64> are for integers"
)]
fn align(original_left: &mut Complex<f64>, original_right: &mut Complex<f64>) {
    // custom function is used over .hypot() for efficiency
    let left_norm_sqr = f64::mul_add(
        original_left.re,
        original_left.re,
        original_left.im * original_left.im,
    );
    let right_norm_sqr = f64::mul_add(
        original_right.re,
        original_right.re,
        original_right.im * original_right.im,
    );

    // false, then left; true, then right
    let louder_mask = left_norm_sqr < right_norm_sqr;
    let louder = usize::from(louder_mask);
    let quieter = usize::from(!louder_mask);
    let norm_sqr_branches = [left_norm_sqr, right_norm_sqr];
    let channel_branches = &mut [original_left, original_right];

    // This method aligns the quieter channel using the louder channel.
    // Mathematically, this seems to minimize the rotation distance needed.
    // Of course, this still creates clicks, but is simpler than the `sum` method
    // Research note: doing the opposite (aligning the louder using the quieter) maximizes the rotation distance,
    //   generally increases peak levels and significantly softens/changes the shape of the spectrum

    // Unsafe unwraps are used since this is part of a hot loop

    // SAFETY: already defined
    let new_quieter_channel = **(unsafe { channel_branches.get_unchecked(louder) })
    // SAFETY: already defined
        * (unsafe { norm_sqr_branches.get_unchecked(quieter) }
    // SAFETY: already defined
            / unsafe { norm_sqr_branches.get_unchecked(louder) })
        .sqrt();
    let is_finite_mask = usize::from(
        new_quieter_channel.re.abs() < f64::INFINITY
            && new_quieter_channel.im.abs() < f64::INFINITY,
    );
    let new_branches = [
        // SAFETY: already defined
        **unsafe { channel_branches.get_unchecked(quieter) },
        new_quieter_channel,
    ];
    // SAFETY: already defined
    **unsafe { channel_branches.get_unchecked_mut(quieter) } =
    // SAFETY: already defined
        *unsafe { new_branches.get_unchecked(is_finite_mask) };

    // If no channel was changed, then left_norm_sqr and right_norm_sqr had to be pretty small
}

/// Specific overlapping.
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
    let scratch_complex = Mutex::new(r2c.make_scratch_vec().into_boxed_slice());
    let scratch_real = Mutex::new(c2r.make_scratch_vec().into_boxed_slice());
    let holding_left = Mutex::new(vec![0_f64; extended_length].into_boxed_slice());
    let holding_right = Mutex::new(vec![0_f64; extended_length].into_boxed_slice());

    let window = window(rounded_time_frame);

    // Windows need a bunch of hops
    // Doing more chunks will help with clicking/zipper noise, but will increase runtime
    // Sorta acts like anti-aliasing in a way
    // Numerator should be 1_f64 for the mininum amount of overlaps, currently doing 16x
    let hop_time_frame = 0.0625_f64 / (WINDOW_COSINES.len() as f64 + 1_f64);
    let hop_size = f64::max((time_frame * hop_time_frame).round_ties_even(), 1.0) as usize;

    // Up until the end, which should be basically a half-window
    // Can't use RangeInclusive unfortunately
    (0..usize::strict_sub(extended_length + 1, rounded_time_frame))
        .into_par_iter()
        .step_by(hop_size)
        .for_each(|holding_position| {
            // Surprisingly, these don't take much memory per thread
            let mut left_chunk = extended_left
                .iter()
                .skip(holding_position)
                .take(rounded_time_frame)
                .copied()
                .collect::<Vec<f64>>();
            let mut right_chunk = extended_right
                .iter()
                .skip(holding_position)
                .take(rounded_time_frame)
                .copied()
                .collect::<Vec<f64>>();
            let mut left_complex = r2c.make_output_vec().into_boxed_slice();
            let mut right_complex = r2c.make_output_vec().into_boxed_slice();

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

            _ = r2c.process_with_scratch(
                &mut left_chunk,
                &mut left_complex,
                &mut scratch_complex.lock().unwrap(),
            );
            _ = r2c.process_with_scratch(
                &mut right_chunk,
                &mut right_complex,
                &mut scratch_complex.lock().unwrap(),
            );

            // Remove DC offset for a better overlap
            // Unsafe being used since this is in a hot loop
            // SAFETY: at least one bin will exist
            *(unsafe { left_complex.first_mut().unwrap_unchecked() }) = Complex::ZERO;
            // SAFETY: at least one bin will exist
            *(unsafe { right_complex.first_mut().unwrap_unchecked() }) = Complex::ZERO;

            // Skip first/DC bin
            left_complex
                .iter_mut()
                .zip(right_complex.iter_mut())
                .skip(1)
                .for_each(|(left_point, right_point)| {
                    align(left_point, right_point);
                });

            // left_chunk and right_chunk will be overwritten
            _ = c2r.process_with_scratch(
                &mut left_complex,
                &mut left_chunk,
                &mut scratch_real.lock().unwrap(),
            );
            _ = c2r.process_with_scratch(
                &mut right_complex,
                &mut right_chunk,
                &mut scratch_real.lock().unwrap(),
            );

            // RealFFT, which uses RustFFT, amplifies the signal by fft_size
            // Normalization happens later in processing.rs

            holding_left
                .lock()
                .unwrap()
                .iter_mut()
                .skip(holding_position)
                .zip(left_chunk)
                .for_each(|(hold_left, left_samp)| *hold_left += left_samp);

            holding_right
                .lock()
                .unwrap()
                .iter_mut()
                .skip(holding_position)
                .zip(right_chunk)
                .for_each(|(hold_right, right_samp)| *hold_right += right_samp);
        });

    // Overlap-adding amplifies the signal by (WINDOW_COSINES.len() as f64 + 1_f64) or 1/hop_time_frame
    // Normalization happens later in processing.rs

    (
        holding_left
            .into_inner()
            .unwrap()
            .into_iter() // Don't think doing .into_par_iter() does anything
            .skip(half_time_frame)
            .take(original_length)
            .collect(),
        holding_right
            .into_inner()
            .unwrap()
            .into_iter()
            .skip(half_time_frame)
            .take(original_length)
            .collect(),
    )
}
