use core::f64::consts::TAU;
use itertools::izip;
use realfft::{RealFftPlanner, num_complex::Complex};

// TODO: add algorithm for arbitrary-length FFT?
/// Given a sample rate, get the best FFT size for `RustFFT` to process
/// `RustFFT` likes FFT lengths which are powers of 2 multiplied with powers of 3
/// We'll zero-pad the seconds anyway
const fn next_fast_fft(rate: usize) -> usize {
    match rate {
        4410 => 4608,   // 44_100
        4800 => 5184,   // 48_000
        8820 => 9216,   // 88_200
        9600 => 10368,  // 96_000
        17640 => 18432, // 176_400
        19200 => 19683, // 192_000
        _ => rate.strict_add(2).next_power_of_two(),
    }
}

/// Windowing is used to make the signal chunk fade in and out
///   to prevent discontinuities, which causes spectral leakage (noise tuned to the music).
fn window(rate: usize) -> Box<[f32]> {
    let new_rate = rate.strict_add(1) as f64;
    // The actual level of the window doesn't really matter
    // Window selection: minimize side-lobe level, ignore bandwidth of main lobe?
    (0..=rate)
        .map(|n| {
            (
                // Matlab flat top window
                // Needs 5 FFTs
                0.21557895 - 0.41663158 * f64::cos(TAU * n as f64 / new_rate)
                    + 0.277263158 * f64::cos(2.0_f64 * TAU * n as f64 / new_rate)
                    - 0.083578947 * f64::cos(3.0_f64 * TAU * n as f64 / new_rate)
                    + 0.006947368 * f64::cos(4.0_f64 * TAU * n as f64 / new_rate)
            ) as f32
        })
        .collect()
}

/// Align the phase of the left and right channels using the circular mean / true midpoint
/// Using this method makes the resulting phase match the downmixed signal phase (left + right / 2),
///   i.e. zero-crossings should match with mid channel
/// Higher weight towards higher magnitude, so the channel with the higher magnitude doesn't rotate much,
///   while the smaller magnitude channel may rotate a lot
fn align(original_left: &mut Complex<f32>, original_right: &mut Complex<f32>) {
    // TODO: find better algorithm
    // For some reason, this causes a noticable amount of clicks when processing songs
    //   with loud bass. In the meantime, we could add error checking
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "clippy thinks this is an integer add"
    )]
    let sum = *original_left + *original_right;
    let sum_norm_recip = sum.norm().recip();
    if sum_norm_recip.is_finite() {
        *original_left = sum.scale(original_left.norm() * sum_norm_recip);
        *original_right = sum.scale(original_right.norm() * sum_norm_recip);
    } else {
        // Just copying the left channel in case of conflicts
        // Seems better than choosing the louder channel in order to preserve the phase between FFTs
        // Implicitly, *original_left = *original_left
        *original_right = *original_left;
    }
}

/// Specific overlapping
pub fn overlap(
    planner: &mut RealFftPlanner<f32>,
    rate: usize,
    left_channel: &[f32],
    right_channel: &[f32],
    holding_left_channel: &mut [f32],
    holding_right_channel: &mut [f32],
    offset: usize,
) {
    let mut offset_left = vec![0.0_f32; offset];
    let mut offset_right = vec![0.0_f32; offset];
    offset_left.extend(left_channel.iter());
    offset_right.extend(right_channel.iter());
    offset_left.shrink_to_fit();
    offset_right.shrink_to_fit();
    let (processed_left, processed_right) = fft_process(planner, offset_left, offset_right, rate);
    izip!(
        holding_left_channel.iter_mut(),
        holding_right_channel.iter_mut(),
        processed_left.iter().skip(offset),
        processed_right.iter().skip(offset),
    )
    .for_each(|(held_left, held_right, left, right)| {
        *held_left += left;
        *held_right += right;
    });
}

/// Two-channel FFT processing
pub fn fft_process(
    planner: &mut RealFftPlanner<f32>,
    mut left_channel: Vec<f32>,
    mut right_channel: Vec<f32>,
    rate: usize,
) -> (Vec<f32>, Vec<f32>) {
    // Best to do a small FFT to prevent transient smearing
    let fft_size = next_fast_fft(rate);
    let recip_fft = (fft_size as f64).recip() as f32;
    let r2c = planner.plan_fft_forward(fft_size);
    let c2r = planner.plan_fft_inverse(fft_size);

    // Pre-calculate window function
    let window = window(rate);

    // The algorithm I want to use will chunk each signal by sample_rate, so it's better to round up
    //   to the next multiple so we can use ChunksExact and have no remainder
    let fft_total = left_channel.len().next_multiple_of(rate);
    left_channel.resize(fft_total, 0.0);
    right_channel.resize(fft_total, 0.0);
    left_channel.shrink_to_fit();
    right_channel.shrink_to_fit();

    // Chunking for later
    let left_channel_chunks = left_channel.chunks_exact(rate);
    let right_channel_chunks = right_channel.chunks_exact(rate);

    // Saving samples for later
    // .reserve_exact() reduces memory by preventing over-allocation
    let mut processed_left: Vec<f32> = vec![];
    let mut processed_right: Vec<f32> = vec![];
    processed_left.reserve_exact(fft_total);
    processed_right.reserve_exact(fft_total);

    // Create scratch FFT for `RealFFT`
    // `RealFFT` uses `RustFFT`'s .process_with_scratch() for its .process() function
    let mut left_fft = r2c.make_output_vec();
    let mut right_fft = r2c.make_output_vec();

    // Scratch vec for chunks
    let mut left = vec![];
    let mut right = vec![];
    left.reserve_exact(fft_size);
    right.reserve_exact(fft_size);

    izip!(left_channel_chunks, right_channel_chunks).for_each(|(left_chunk, right_chunk)| {
        left.push(0.0_f32);
        right.push(0.0_f32);
        left.extend(left_chunk);
        right.extend(right_chunk);

        // length is now sample_rate + 1
        // Skip the first element, which should be zero for all of these iterators
        izip!(left.iter_mut(), right.iter_mut(), window.iter())
            .skip(1)
            .for_each(|(left_point, right_point, window_multiplier)| {
                *left_point *= window_multiplier;
                *right_point *= window_multiplier;
            });

        left.resize(fft_size, 0.0_f32);
        right.resize(fft_size, 0.0_f32);

        // Ignore errors by `RealFFT`
        // `RustFFT` does not return a Result after processing,
        //   but `RealFFT` does return Results due to some zero-check
        //   `RealFFT` author says to just ignore these in the meantime.
        // https://github.com/HEnquist/realfft/issues/41#issuecomment-2050347470
        _ = r2c.process(&mut left, &mut left_fft);
        _ = r2c.process(&mut right, &mut right_fft);

        // The first bin is the DC bin, which is the average unheard noise
        // It's better to handle DC in the time domain, not the frequency domain,
        //   so we skip it
        izip!(left_fft.iter_mut(), right_fft.iter_mut())
            .skip(1)
            .for_each(|(left_fft_point, right_fft_point)| {
                align(left_fft_point, right_fft_point);
            });

        _ = c2r.process(&mut left_fft, &mut left);
        _ = c2r.process(&mut right_fft, &mut right);

        // Remove remaining FFT silence
        left.truncate(rate.strict_add(1));
        right.truncate(rate.strict_add(1));

        // Remove first sample, as it should be silence
        // This should be faster than using a VecDeque since we're just removing one time
        left.remove(0);
        right.remove(0);

        izip!(left.iter_mut(), right.iter_mut()).for_each(|(left_samp, right_samp)| {
            *left_samp *= recip_fft;
            *right_samp *= recip_fft;
        });

        // Scratch Vec's are cleared by these lines
        processed_left.append(&mut left);
        processed_right.append(&mut right);
    });

    (processed_left, processed_right)
}
