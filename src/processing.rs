use crate::fft;
use core::f64::consts::PI;
use ebur128::{EbuR128, Mode};
use itertools::{Itertools as _, izip};
use realfft::RealFftPlanner;

/// Force minimum reconstructed frequency to `MIN_FREQ` hertz
// 16hz is used since it's short enough to prevent smearing
// Recommended range is 10hz to 20hz, inclusive
const MIN_FREQ: f64 = 16.0_f64;

/// EBU R 128 Integrated Loudness calculation
/// Basically a two-pass windowed RMS.
///   First pass is used to detect and ignore silence at -70dB
///   Second pass is used to detect and ignore audio that's 10dB below the first-pass result
fn gated_rms(samples: &[f64], sample_rate: u32) -> f64 {
    let mut ebur128 = EbuR128::new(1_u32, sample_rate, Mode::I).expect("Shouldn't happen");
    // .add_frames_f64_planar() sucks since it requires an array of channel arrays, so for one channel, it needs an array around it
    // f64 samples are not needed
    _ = ebur128.add_frames_f64(samples);
    let loudness = ebur128.loudness_global().expect("Shouldn't happen");

    10.0_f64.powf(loudness * 0.05_f64)
}

/// Active DC removal
fn remove_dc(sample_rate: u32, channel: &mut [f64]) {
    let f64_sample_rate = f64::from(sample_rate);

    // Discrete-time high pass filter with `MIN_FREQ/2`hz as the corner frequency
    // i.e. at `MIN_FREQ/2`hz, the level should be lowered by 3dB
    let alpha = f64_sample_rate / f64::mul_add(MIN_FREQ, PI, f64_sample_rate);
    let mut previous_output = 0.0_f64;
    let mut previous_input = 0.0_f64;
    for input in channel.iter_mut() {
        let passed_output = alpha * (previous_output + *input - previous_input);
        previous_input = *input;
        previous_output = passed_output;
        *input = passed_output;
    }
}

/// All three processing steps into one function
pub fn process_samples(
    planner: &mut RealFftPlanner<f64>,
    data: (Vec<f64>, Vec<f64>),
    sample_rate: u32,
) -> (Vec<f64>, Vec<f64>) {
    let mut left_channel = data.0;
    let mut right_channel = data.1;

    // Remove DC before processing
    // DC might affect magnitude of `MIN_FREQ` Hz and interpolated values close to it
    remove_dc(sample_rate, &mut left_channel);
    remove_dc(sample_rate, &mut right_channel);

    // Integrated Loudness shouldn't be affected by DC noise, but this is placed after DC removal just in case
    let true_left_rms = gated_rms(&left_channel, sample_rate);
    let true_right_rms = gated_rms(&right_channel, sample_rate);
    let true_mean_rms = f64::sqrt(true_left_rms * true_right_rms);

    // Average out plain RMS of left and right channels before processing
    // Might help in phase conflicts
    // Human hearing doesn't matter here
    let length_recip = 1.0_f64 / left_channel.len() as f64;
    let plain_left_rms = f64::sqrt(
        left_channel
            .iter()
            .fold(0.0_f64, |acc, samp| samp.mul_add(*samp, acc))
            * length_recip,
    );
    let plain_right_rms = f64::sqrt(
        right_channel
            .iter()
            .fold(0.0_f64, |acc, samp| samp.mul_add(*samp, acc))
            * length_recip,
    );
    let plain_mean_rms = f64::sqrt(plain_left_rms * plain_right_rms);
    let left_mult = plain_mean_rms / plain_left_rms;
    let right_mult = plain_mean_rms / plain_right_rms;
    izip!(left_channel.iter_mut(), right_channel.iter_mut()).for_each(|(left_samp, right_samp)| {
        *left_samp *= left_mult;
        *right_samp *= right_mult;
    });

    let time_frame = f64::from(sample_rate) / MIN_FREQ; // actually in number of samples
    let (mut processed_left, mut processed_right) =
        fft::overlapping_fft(planner, time_frame, left_channel, right_channel);

    // Remove DC after processing
    remove_dc(sample_rate, &mut processed_left);
    remove_dc(sample_rate, &mut processed_right);

    // Average out the loudness of the left and right channels
    let processed_left_rms = gated_rms(&processed_left, sample_rate);
    let processed_right_rms = gated_rms(&processed_right, sample_rate);
    let processed_left_mult = true_mean_rms / processed_left_rms;
    let processed_right_mult = true_mean_rms / processed_right_rms;
    izip!(processed_left.iter_mut(), processed_right.iter_mut()).for_each(
        |(left_samp, right_samp)| {
            *left_samp *= processed_left_mult;
            *right_samp *= processed_right_mult;
        },
    );

    // Add DC noise to reduce peak levels
    let (left_min, left_max) = processed_left.iter().minmax().into_option().unwrap();
    let (right_min, right_max) = processed_right.iter().minmax().into_option().unwrap();
    let new_left_dc = f64::midpoint(*left_min, *left_max);
    let new_right_dc = f64::midpoint(*right_min, *right_max);
    izip!(processed_left.iter_mut(), processed_right.iter_mut()).for_each(
        |(left_samp, right_samp)| {
            *left_samp -= new_left_dc;
            *right_samp -= new_right_dc;
        },
    );

    // Overall processing is done
    // pack it up
    (processed_left, processed_right)
}
