use core::f64::consts::TAU;
use itertools::izip;
use realfft::{RealFftPlanner, num_complex::Complex};

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
    let sum = *original_left + *original_right;
    let sum_sqr_recip = sum.norm_sqr().recip(); // Division-by-near-zero check
    if sum_sqr_recip.is_finite() {
        // This catches almost all cases
        *original_left = sum.scale(f32::sqrt(original_left.norm_sqr() * sum_sqr_recip));
        *original_right = sum.scale(f32::sqrt(original_right.norm_sqr() * sum_sqr_recip));
    } else {
        // This case should occur if sum.norm_sqr() is subnormal or 0.0
        // In such a case, we should try doing .norm() which uses .hypot(), which might get us out of subnormality
        // NOTE: .norm_sqr().sqrt() can give different results from .hypot(), and I don't know whether both can give subnormal numbers
        let sum_norm_recip = sum.norm().recip(); // Second division-by-near-zero check
        let left_norm = original_left.norm();
        let right_norm = original_right.norm();

        if sum_norm_recip.is_finite() {
            *original_left = sum.scale(left_norm * sum_norm_recip);
            *original_right = sum.scale(right_norm * sum_norm_recip);
        } else {
            // In the very rare case that sum.norm() is still subnormal or 0.0,
            //   assume that there is zero phase, i.e. 0.0i
            *original_left = Complex::new(left_norm, 0.0_f32);
            *original_right = Complex::new(right_norm, 0.0_f32);
        }
    }
}

/// Windowing is used to make the signal chunk fade in and out
///   to prevent discontinuities, which causes spectral leakage (noise tuned to the music).
fn window(time_frame: usize) -> Box<[f32]> {
    let f64_rate_recip = (time_frame as f64).recip();
    // The actual level of the window doesn't really matter
    // Window selection: minimize side-lobe level, ignore bandwidth of main lobe?
    (0..time_frame)
        .map(|n| {
            // 6-term (5 cosines and 1 constant) HFT116D window,
            //   flat top window with -116.8dB max leakage, but is pretty wide
            // https://holometer.fnal.gov/GH_FFT.pdf

            // List of coefficients, where the first term is the constant (multiplier * cos(0) == multiplier)
            [
                1.0_f64,
                -1.957_537_5_f64,
                1.478_070_5_f64,
                -0.636_743_1_f64,
                0.122_838_9_f64,
                -0.006_628_8_f64,
            ]
            .iter()
            .enumerate()
            .map(|(index, multiplier)| {
                multiplier * f64::cos(index as f64 * TAU * n as f64 * f64_rate_recip)
            })
            .sum::<f64>() as f32
        })
        .collect()
}

/// Specific overlapping
pub fn overlapping_fft(
    planner: &mut RealFftPlanner<f32>,
    time_frame: f64,
    left_channel: Vec<f32>,
    right_channel: Vec<f32>,
) -> (Vec<f32>, Vec<f32>) {
    let original_length = left_channel.len();

    // Idea is that time_frame gives us the amount of samples (possibly fractional) that we need to FFT
    let rounded_time_frame = time_frame.round_ties_even() as usize;
    // We should pad with half-a-second of silence to allow for half-windows at the beginning and end
    let half_time_frame = (time_frame * 0.5_f64).round_ties_even() as usize;

    // Lots of Vecs are used here to reuse memory space instead of reallocating
    let fft_size = rounded_time_frame.next_power_of_two();
    let fft_norm = (fft_size as f64).recip() as f32;
    let r2c = planner.plan_fft_forward(fft_size);
    let c2r = planner.plan_fft_inverse(fft_size);
    let mut left_chunk = Vec::with_capacity(fft_size);
    let mut right_chunk = Vec::with_capacity(fft_size);
    let mut left_complex = r2c.make_output_vec();
    let mut right_complex = r2c.make_output_vec();
    let mut scratch_complex = r2c.make_scratch_vec();
    let mut scratch_real = c2r.make_scratch_vec();

    // This consumes left_channel and right_channel
    let mut extended_left = vec![0.0_f32; half_time_frame]; // Allow half-window at start
    let mut extended_right = vec![0.0_f32; half_time_frame];
    extended_left.extend(left_channel);
    extended_right.extend(right_channel);
    extended_left.extend(vec![0.0_f32; fft_size]); // Allow for last FFT chunk to be added
    extended_right.extend(vec![0.0_f32; fft_size]);
    let extended_length = extended_left.len();

    let mut holding_left = vec![0.0_f32; extended_length];
    let mut holding_right = vec![0.0_f32; extended_length];
    let mut holding_position = 0_usize;

    let window = window(rounded_time_frame);

    // Window has 6 terms (5 cosine and 1 constant), so we need to add at least 6 shifted chunks to get a constant output level
    // doing more chunks will help with clicking/zipper noise, but will of course increase runtime
    let hop_time_frame = 1.0_f64 / 6.0_f64;
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

        left_chunk.resize(fft_size, 0.0_f32);
        right_chunk.resize(fft_size, 0.0_f32);

        _ = r2c.process_with_scratch(&mut left_chunk, &mut left_complex, &mut scratch_complex);
        _ = r2c.process_with_scratch(&mut right_chunk, &mut right_complex, &mut scratch_complex);

        // .skip(1) is needed to ignore the DC bin, which is the average vertical offset
        // It shouldn't be changed in case the vertical offset is actually just a very low frequency
        // TODO: check for songs where .skip(1) is absolutely necessary
        izip!(left_complex.iter_mut(), right_complex.iter_mut())
            .skip(1)
            .for_each(|(left_point, right_point)| {
                align(left_point, right_point);
            });

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
            reason = "nursery lint, .drain(..) is needed to clear the vec"
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

    let f32_hop_time_frame = hop_time_frame as f32;
    izip!(holding_left.iter_mut(), holding_right.iter_mut()).for_each(|(hold_left, hold_right)| {
        *hold_left *= f32_hop_time_frame;
        *hold_right *= f32_hop_time_frame;
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
