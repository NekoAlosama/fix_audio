extern crate alloc;
use alloc::sync::Arc;
use core::f64::consts::TAU;
use itertools::izip;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex, num_complex::Complex};

/// Align the phase of the left and right channels using the circular mean / true midpoint
/// Using this method makes the resulting phase match the downmixed signal phase (left + right / 2),
///   i.e. zero-crossings should match with mid channel
/// Higher weight towards higher magnitude, so the channel with the higher magnitude doesn't rotate much,
///   while the smaller magnitude channel may rotate a lot
#[expect(
    clippy::arithmetic_side_effects,
    reason = "clippy thinks the operations done on Complex<f32> are for integers"
)]
fn align(original_left: &mut Complex<f32>, original_right: &mut Complex<f32>) {
    // TODO: find better algorithm
    // For some reason, this causes a noticable amount of clicks when processing songs
    //   with loud bass. In the meantime, we could add error checking
    let sum = *original_left + *original_right;
    let sum_sqr_recip = sum.norm_sqr().recip(); // Should never be subnormal
    if sum_sqr_recip.is_finite() {
        // This catches most cases
        *original_left = sum.scale((original_left.norm_sqr() * sum_sqr_recip).sqrt());
        *original_right = sum.scale((original_right.norm_sqr() * sum_sqr_recip).sqrt());
    } else {
        let sum_norm_recip = sum.norm().recip(); // Should never be subnormal
        let left_norm = original_left.norm();
        let right_norm = original_right.norm();

        if sum_norm_recip.is_finite() {
            *original_left = sum.scale(left_norm * sum_norm_recip);
            *original_right = sum.scale(right_norm * sum_norm_recip);
        } else {
            // Just flip the phase of the quieter channel in case of conflicts
            if left_norm < right_norm {
                *original_left = -*original_left;
            } else {
                *original_right = -*original_right;
            }
        }
    }
}

/// Two-channel FFT processing
fn fft_process(
    r2c: &Arc<dyn RealToComplex<f32>>,
    c2r: &Arc<dyn ComplexToReal<f32>>,
    window: &[f32],
    fft_size: usize,
    mut left_channel: Vec<f32>,
    mut right_channel: Vec<f32>,
) -> (Vec<f32>, Vec<f32>) {
    let time_frame = window.len();

    let recip_fft = (fft_size as f64).recip() as f32;

    // The algorithm I want to use will chunk each signal by sample_rate, so it's better to round up
    //   to the next multiple so we can use ChunksExact and have no remainder
    let fft_total = left_channel.len().next_multiple_of(time_frame);
    left_channel.resize(fft_total, 0.0);
    right_channel.resize(fft_total, 0.0);
    left_channel.shrink_to_fit();
    right_channel.shrink_to_fit();

    // Chunking for later
    let left_channel_chunks = left_channel.chunks_exact(time_frame);
    let right_channel_chunks = right_channel.chunks_exact(time_frame);

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
        left.extend(left_chunk);
        right.extend(right_chunk);

        // length is now sample_rate + 1
        // Skip the first element, which should be zero for all of these iterators
        izip!(left.iter_mut(), right.iter_mut(), window.iter()).for_each(
            |(left_point, right_point, window_multiplier)| {
                *left_point *= window_multiplier;
                *right_point *= window_multiplier;
            },
        );

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

        // Remove FFT silence
        left.truncate(time_frame);
        right.truncate(time_frame);

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

/// Windowing is used to make the signal chunk fade in and out
///   to prevent discontinuities, which causes spectral leakage (noise tuned to the music).
fn window(time_frame: usize) -> Box<[f32]> {
    let f64_rate = time_frame as f64;
    // The actual level of the window doesn't really matter
    // Window selection: minimize side-lobe level, ignore bandwidth of main lobe?
    (0..time_frame)
        .map(|n| {
            (
                // 6-term (5 cosines and 1 constant) HFT116D window,
                //   flat top window with -116.8dB max leakage, but is pretty wide
                // In backwards order due to mul_add
                // https://holometer.fnal.gov/GH_FFT.pdf
                f64::mul_add(
                    -0.006_628_8_f64,
                    f64::cos(5.0_f64 * TAU * n as f64 / f64_rate),
                    f64::mul_add(
                        0.122_838_9_f64,
                        f64::cos(4.0_f64 * TAU * n as f64 / f64_rate),
                        f64::mul_add(
                            -0.636_743_1_f64,
                            f64::cos(3.0_f64 * TAU * n as f64 / f64_rate),
                            f64::mul_add(
                                1.478_070_5_f64,
                                f64::cos(2.0_f64 * TAU * n as f64 / f64_rate),
                                f64::mul_add(
                                    -1.957_537_5_f64,
                                    f64::cos(TAU * n as f64 / f64_rate),
                                    1.0_f64,
                                ),
                            ),
                        ),
                    ),
                )
            ) as f32
        })
        .collect()
}

/// Specific overlapping
pub fn overlapping_fft(
    time_frame: f64,
    left_channel: &[f32],
    right_channel: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let original_length = left_channel.len();
    let mut holding_left = vec![0.0_f32; original_length];
    let mut holding_right = vec![0.0_f32; original_length];
    holding_left.shrink_to_fit();
    holding_right.shrink_to_fit();

    let rounded_time_frame = time_frame.round_ties_even() as usize;

    let window = window(rounded_time_frame);
    let fft_size = rounded_time_frame.next_power_of_two();
    let mut planner = RealFftPlanner::new();
    let r2c = planner.plan_fft_forward(fft_size);
    let c2r = planner.plan_fft_inverse(fft_size);

    let number_of_overlaps = 6_i32; // at least one overlap per term in window
    let f64_noo_recip = f64::from(number_of_overlaps).recip();
    (0_i32..number_of_overlaps).for_each(|notch| {
        let offset = (time_frame * f64::from(notch) * f64_noo_recip).round_ties_even() as usize;
        let mut offset_left = vec![0.0_f32; offset];
        let mut offset_right = vec![0.0_f32; offset];
        offset_left.extend(left_channel.iter());
        offset_right.extend(right_channel.iter());

        let (processed_left, processed_right) =
            fft_process(&r2c, &c2r, &window, fft_size, offset_left, offset_right);
        izip!(
            holding_left.iter_mut(),
            holding_right.iter_mut(),
            processed_left.iter().skip(offset),
            processed_right.iter().skip(offset)
        )
        .for_each(|(hold_left, hold_right, proc_left, proc_right)| {
            *hold_left += *proc_left;
            *hold_right += *proc_right;
        });
    });

    // Correct for overlapping
    let f32_noo_recip = f64_noo_recip as f32;
    izip!(holding_left.iter_mut(), holding_right.iter_mut()).for_each(|(hold_left, hold_right)| {
        *hold_left *= f32_noo_recip;
        *hold_right *= f32_noo_recip;
    });

    (holding_left, holding_right)
}
