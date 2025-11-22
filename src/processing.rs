use ebur128::{EbuR128, Mode};
use itertools::izip;
use realfft::RealFftPlanner;

use crate::fft;

/// Force minimum reconstructed frequency to `MIN_FREQ` hertz
/// Thimeo Stereo Tool suggests that it uses 4096 samples, which is about 11hz between 44.1khz and 48khz audio
// For some reason, the 11hz creates smearing here, but not in Stereo Tool?
const MIN_FREQ: f64 = 20.;

/// 10^(1/20), for `gated_rms()`
/// `10_f64.powf(loudness * 0.05_f64)` == `LOUDNESS_BASE.powf(loudness)`
const LOUDNESS_BASE: f64 = 1.122_018_454_301_963_3_f64;

/// EBU R 128 Integrated Loudness calculation
/// Basically a two-pass windowed RMS.
///   First pass is used to detect and ignore silence at -70dB
///   Second pass is used to detect and ignore audio that's 10dB below the first-pass result
fn gated_rms(samples: &[f64], sample_rate: u32) -> f64 {
    let mut ebur128 = EbuR128::new(1_u32, sample_rate, Mode::I).expect("Shouldn't happen");
    // .add_frames_f64_planar() sucks since it requires an array of channel arrays, so for one channel, it needs an array around it
    _ = ebur128.add_frames_f64(samples);
    let loudness = ebur128.loudness_global().expect("Shouldn't happen");

    LOUDNESS_BASE.powf(loudness)
}

/// Plain DC removal
/// A high-pass filter isn't being used here in order to preserve the shape of the waveform
fn remove_dc(channel: &mut [f64]) {
    let length = channel.len() as f64;
    let dc = channel.iter().sum::<f64>() / length;
    for samp in channel.iter_mut() {
        *samp -= dc;
    }
}

/// All processing steps into one function
pub fn process_samples(
    planner: &mut RealFftPlanner<f64>,
    data: (Box<[f64]>, Box<[f64]>),
    sample_rate: u32,
) -> (Box<[f64]>, Box<[f64]>) {
    let mut left_channel = data.0;
    let mut right_channel = data.1;

    // Remove DC before processing
    // DC might affect magnitude of `MIN_FREQ` Hz and interpolated values close to it
    remove_dc(&mut left_channel);
    remove_dc(&mut right_channel);

    // Integrated Loudness shouldn't be affected by DC noise, but this is placed after DC removal just in case
    let true_left_rms = gated_rms(&left_channel, sample_rate);
    let true_right_rms = gated_rms(&right_channel, sample_rate);
    let true_mean_rms = f64::sqrt(true_left_rms * true_right_rms);

    // Average out plain RMS of left and right channels before processing
    // Might help in phase conflicts
    // Human hearing doesn't matter here?
    let plain_left_rms = f64::sqrt(
        left_channel
            .iter()
            .fold(0_f64, |acc, samp| samp.mul_add(*samp, acc)),
    );
    let plain_right_rms = f64::sqrt(
        right_channel
            .iter()
            .fold(0_f64, |acc, samp| samp.mul_add(*samp, acc)),
    );
    let plain_mean_rms = f64::sqrt(plain_left_rms * plain_right_rms);
    let left_mult = plain_mean_rms / plain_left_rms;
    let right_mult = plain_mean_rms / plain_right_rms;
    izip!(left_channel.iter_mut(), right_channel.iter_mut()).for_each(|(left_samp, right_samp)| {
        *left_samp *= left_mult;
        *right_samp *= right_mult;
    });

    // Out-of-phase checker, flips the right channel if a majority of samples are out-of-phase
    // Used to reduce a lot of near-zero-sum cases
    let mut oop_counter: usize = 0;
    izip!(left_channel.iter(), right_channel.iter()).for_each(|(left_samp, right_samp)| {
        if left_samp.signum() != right_samp.signum() {
            oop_counter = oop_counter.saturating_add(1);
        }
    });
    if oop_counter as f64 > 0.5_f64 * left_channel.len() as f64 {
        right_channel.iter_mut().for_each(|samp| *samp = -*samp);
    }

    let time_frame = f64::from(sample_rate) / MIN_FREQ; // actually in number of samples
    let (mut processed_left, mut processed_right) =
        fft::overlapping_fft(planner, time_frame, left_channel, right_channel);

    // STFT will generate sub-MIN_FREQ noise
    // As such, DC noise is likely added and should be removed since we'll multiply the signals later
    remove_dc(&mut processed_left);
    remove_dc(&mut processed_right);

    // Average out the loudness of the left and right channels
    // This handles amplification by RustFFT and overlap-adding, assuming there isn't much precision loss
    // TODO: This might make the first -70dB pass redundant
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

    // Overall processing is done
    (processed_left, processed_right)
}
