use core::f64::consts::TAU;
use std::sync::Mutex;

use itertools::izip;
use rayon::iter::{IntoParallelIterator as _, ParallelIterator as _};
use realfft::{RealFftPlanner, num_complex::Complex};

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

    // To attempt a branchless version of this function, we just use a boolean to decide on which channels to use and have each new line be evaluated based on it
    // louder_mask = 0 means to use the left to overwrite the right, 1 means to use the right to overwrite the left
    let louder_mask = right_norm_sqr >= left_norm_sqr;
    let louder = usize::from(louder_mask); // Booleans need to be converted to usize for indexing, so we'll just do this early
    let quieter = usize::from(!louder_mask); // ^
    let norm_sqr_branches = [left_norm_sqr, right_norm_sqr];
    let channel_branches = &mut [original_left, original_right];

    // This method aligns the quieter channel using the louder channel.
    // Mathematically, this seems to minimize the rotation distance needed.
    // Of course, this still creates clicks, but is simpler than the `sum` method
    // Research note: doing the opposite (aligning the louder using the quieter) maximizes the rotation distance,
    //   generally increases peak levels and significantly softens/changes the shape of the spectrum

    // Unsafe unwraps are used since this is part of a hot loop

    // We will take the louder channel's bin and rescale it to match the magnitude of the quieter channel
    // i.e. c_2_hat = c_1 * |c_2|/|c_1|
    // SAFETY: already defined
    let new_quieter_channel = **(unsafe { channel_branches.get_unchecked(louder) })
    // SAFETY: already defined
        * (unsafe { norm_sqr_branches.get_unchecked(quieter) }
    // SAFETY: already defined
            / unsafe { norm_sqr_branches.get_unchecked(louder) })
        .sqrt();

    // Even though in the above we are dividing by |c_1|, it could stil be low enough to create NaN if equal to +0.0 or +Inf if subnormal
    let is_finite_mask = usize::from(
        new_quieter_channel.re.abs() < f64::INFINITY
            && new_quieter_channel.im.abs() < f64::INFINITY,
    );

    // Over here, we choose whether to set the quieter channel to the new version, or set it to itself (which is hopefully optimized out either by the compiler or CPU)
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
    // Idea is that time_frame gives us the amount of samples (possibly fractional) that we need to FFT
    let rounded_time_frame = time_frame.round_ties_even() as usize;
    // We should pad with half-a-second of silence to allow for half-windows at the beginning and end
    let half_time_frame = (time_frame * 0.5_f64).round_ties_even() as usize;

    // For fft_size, there's different opinions online on how much zero-padding is needed
    let fft_size = {
        let pre_fft_size = rounded_time_frame.next_power_of_two(); // Round to next power of 2 for some zero-padding and for a fast FFT
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
    let r2c = planner.plan_fft_forward(fft_size);
    let c2r = planner.plan_fft_inverse(fft_size);
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
        let mut scratch = c2r.make_scratch_vec().into_boxed_slice();

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

        _ = r2c.process_with_scratch(&mut left_chunk, &mut left_complex, &mut scratch);
        _ = r2c.process_with_scratch(&mut right_chunk, &mut right_complex, &mut scratch);

        // Remove DC offset for a better overlap
        // Unsafe being used since this is in a hot loop
        // SAFETY: at least one bin will exist
        *(unsafe { left_complex.get_unchecked_mut(0) }) = Complex::ZERO;
        // SAFETY: at least one bin will exist
        *(unsafe { right_complex.get_unchecked_mut(0) }) = Complex::ZERO;

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

        // left_chunk is done first so less time is used locking the mutexes
        left_chunk
            .into_iter()
            .zip(
                holding_left
                    .lock()
                    .unwrap()
                    .iter_mut()
                    .skip(holding_position),
            )
            .for_each(|(left_samp, hold_left)| *hold_left += left_samp);

        right_chunk
            .into_iter()
            .zip(
                holding_right
                    .lock()
                    .unwrap()
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
            .unwrap()
            .into_iter() // Don't think doing .into_par_iter() does anything
            .skip(half_time_frame)
            .collect(),
        holding_right
            .into_inner()
            .unwrap()
            .into_iter()
            .skip(half_time_frame)
            .collect(),
    )
}
