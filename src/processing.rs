use crate::fft;
use ebur128::{EbuR128, Mode};
use itertools::{Itertools as _, izip};

/// EBU R 128 Integrated Loudness calculation
/// Basically a two-pass windowed RMS.
///   First pass is used to detect and ignore silence at -70dB
///   Second pass is used to detect and ignore quieter points at the first-pass RMS minus 10dB
fn gated_rms(samples: &Vec<f32>, sample_rate: u32) -> f64 {
    let mut ebur128 = EbuR128::new(1_u32, sample_rate, Mode::I).expect("Shouldn't happen");
    // Planar sucks since it requires an array of channel arrays, so for one channel, it needs an array around it
    // f64 samples are not needed
    _ = ebur128.add_frames_f32(&samples.to_owned());
    let loudness = ebur128.loudness_global().expect("Shouldn't happen");

    10.0_f64.powf(loudness / 20.0_f64)
}

/// Static DC removal
fn remove_dc(channel_1: &mut [f32], channel_2: &mut [f32]) {
    let length = channel_1.len() as f32;
    let first_dc = channel_1.iter().sum::<f32>() / length;
    let second_dc = channel_2.iter().sum::<f32>() / length;
    izip!(channel_1.iter_mut(), channel_2.iter_mut()).for_each(|(first, second)| {
        *first -= first_dc;
        *second -= second_dc;
    });
}

/// All three processing steps into one function
pub fn process_samples(data: (Vec<f32>, Vec<f32>), sample_rate: u32) -> (Vec<f32>, Vec<f32>) {
    let mut left_channel = data.0;
    let mut right_channel = data.1;
    let left_rms = gated_rms(&left_channel, sample_rate);
    let right_rms = gated_rms(&right_channel, sample_rate);
    let mean_rms = f64::sqrt(left_rms * right_rms);

    // Remove DC before processing
    // DC might affect magnitude of N Hz and interpolated values close to it
    remove_dc(&mut left_channel, &mut right_channel);

    // Average out RMS of left and right channels before processing
    // Might help in phase conflicts
    let left_mult = (mean_rms / left_rms) as f32;
    let right_mult = (mean_rms / right_rms) as f32;
    izip!(left_channel.iter_mut(), right_channel.iter_mut()).for_each(
        |(left_samp, right_samp)| {
            *left_samp *= left_mult;
            *right_samp *= right_mult;
        },
    );

    // Force minimum reconstructed frequency to N hertz
    // 16hz is used since it's short enough to prevent smearing
    let time_frame = f64::from(sample_rate) / 16.0_f64;
    let (mut processed_left, mut processed_right) =
        fft::overlapping_fft(time_frame, &left_channel, &right_channel);
    drop(left_channel);
    drop(right_channel);

    // Remove DC after processing
    remove_dc(&mut processed_left, &mut processed_right);

    // Average out the loudness of the left and right channels
    // Need to .sqrt() the RMS to get the per-sample multiplier instead of the per-RMS multiplier
    let processed_left_rms = gated_rms(&processed_left, sample_rate);
    let processed_right_rms = gated_rms(&processed_right, sample_rate);
    let processed_left_mult = (mean_rms / processed_left_rms) as f32;
    let processed_right_mult = (mean_rms / processed_right_rms) as f32;
    izip!(processed_left.iter_mut(), processed_right.iter_mut()).for_each(
        |(left_samp, right_samp)| {
            *left_samp *= processed_left_mult;
            *right_samp *= processed_right_mult;
        },
    );

    // Add DC noise to reduce peak levels
    let (left_min, left_max) = processed_left.iter().minmax().into_option().unwrap();
    let (right_min, right_max) = processed_right.iter().minmax().into_option().unwrap();
    let new_left_dc = f32::midpoint(*left_min, *left_max);
    let new_right_dc = f32::midpoint(*right_min, *right_max);
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
