use crate::fft::{fft_process, overlap};
use ebur128::{EbuR128, Mode};
use itertools::{Itertools as _, izip};
use realfft::RealFftPlanner;

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
    let original_length = left_channel.len();
    // Force minimum reconstructed frequency to N hertz
    let f64_rate = f64::from(sample_rate) / 20.0_f64; // Testing with 20 Hz
    let rate = f64_rate.round_ties_even() as usize;

    // Remove DC before processing
    // DC might affect magnitude of N Hz and interpolated values close to it
    remove_dc(&mut left_channel, &mut right_channel);

    let mut real_planner = RealFftPlanner::<f32>::new();
    let (mut processed_left, mut processed_right) = fft_process(
        &mut real_planner,
        left_channel.clone(),
        right_channel.clone(),
        rate,
    );

    // Make other FFTs to overlap with the original
    // Overlaps have to have a 50% difference, not necessarily sum to 100%
    // Also for some reason reduces peak levels???
    // TODO: offset can technically fail original_length is near isize::MAX
    overlap(
        &mut real_planner,
        rate,
        &left_channel,
        &right_channel,
        &mut processed_left,
        &mut processed_right,
        (f64_rate * 0.5).round_ties_even() as usize,
    );
    overlap(
        &mut real_planner,
        rate,
        &left_channel,
        &right_channel,
        &mut processed_left,
        &mut processed_right,
        (f64_rate * 0.25).round_ties_even() as usize,
    );
    overlap(
        &mut real_planner,
        rate,
        &left_channel,
        &right_channel,
        &mut processed_left,
        &mut processed_right,
        (f64_rate * 0.75).round_ties_even() as usize,
    );
    drop(left_channel);
    drop(right_channel);

    // Divide by 2 because of two full overlaps being used
    //   first full: original (i.e. 0%) + 50%
    //   second full: 25% + 75%
    izip!(processed_left.iter_mut(), processed_right.iter_mut()).for_each(|(left, right)| {
        *left *= 0.5;
        *right *= 0.5;
    });

    // Remove chunking zero-padding
    processed_left.truncate(original_length);
    processed_right.truncate(original_length);
    processed_left.shrink_to_fit();
    processed_right.shrink_to_fit();

    // Remove DC after processing
    remove_dc(&mut processed_left, &mut processed_right);

    // Average out the loudness of the left and right channels
    // Need to .sqrt() the RMS to get the per-sample multiplier instead of the per-RMS multiplier
    let left_rms_sqrt = gated_rms(&processed_left, sample_rate).sqrt();
    let right_rms_sqrt = gated_rms(&processed_right, sample_rate).sqrt();
    let left_equalizer = (right_rms_sqrt / left_rms_sqrt) as f32;
    let right_equalizer = (left_rms_sqrt / right_rms_sqrt) as f32;
    izip!(processed_left.iter_mut(), processed_right.iter_mut()).for_each(
        |(left_samp, right_samp)| {
            *left_samp *= left_equalizer;
            *right_samp *= right_equalizer;
        },
    );

    // Add DC noise to reduce peak levels
    let (left_min, left_max) = processed_left.iter().minmax().into_option().unwrap();
    let (right_min, right_max) = processed_right.iter().minmax().into_option().unwrap();
    let new_left_dc = (*left_min + *left_max) * 0.5_f32;
    let new_right_dc = (*right_min + *right_max) * 0.5_f32;
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
