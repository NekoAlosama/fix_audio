// General Clippy warnings
#![warn(clippy::complexity)]
#![warn(clippy::correctness)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![warn(clippy::perf)]
#![warn(clippy::suspicious)]
#![allow(clippy::cast_precision_loss)] // Allow u64 as f64, since we shouldn't really have u64 exceeding the 52-bit mantissa
#![allow(clippy::cast_possible_truncation)] // Allow f64 as f32, since f64 takes double the memory of f32 and we can only write f32

// Imports
// TODO: consider Clippy's style lints for imports
use realfft::{RealFftPlanner, num_complex, num_traits::Zero};
use std::{
    fs,
    io::{self, Write},
    path, time,
};
use symphonia::{
    core::{
        self, codecs::DecoderOptions, formats::FormatOptions, io::MediaSourceStreamOptions,
        meta::MetadataOptions,
    },
    default,
};

// Hard-coded directories
const INPUT_DIR: &str = "./inputs/";
const OUTPUT_DIR: &str = "./outputs/";

type AudioMatrix = ((Vec<f32>, Vec<f32>), u32); // Seperated here due to Clippy lint
fn get_samples_and_metadata(path: &path::PathBuf) -> Result<AudioMatrix, core::errors::Error> {
    // Based on Symphonia's docs.rs page and example code (mix of 0.5.4 and dev-0.6)
    // Numbers are from the Symphonia basic proceedures in its docs.rs

    // 1
    let code_registry = default::get_codecs();
    // 2
    let probe = default::get_probe();

    // 3
    // 4
    let mss = core::io::MediaSourceStream::new(
        Box::new(fs::File::open(path)?),
        MediaSourceStreamOptions::default(),
    );

    // 5
    // 6
    let probe_result = probe.format(
        core::probe::Hint::new().with_extension("flac"),
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probe_result.format;

    // 7
    let track = format.default_track().unwrap();

    // 8
    let mut decoder = code_registry.make(&track.codec_params, &DecoderOptions::default())?;

    let track_id = track.id;

    let mut left_samples: Vec<f32> = vec![];
    let mut right_samples: Vec<f32> = vec![];

    let mut sample_buf = None;
    let mut meta_spec = 0;

    // 9
    // 10
    // 11
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() == track_id {
            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    if sample_buf.is_none() {
                        let spec = *audio_buf.spec();
                        meta_spec = spec.rate;

                        let duration = audio_buf.capacity() as u64;

                        sample_buf = Some(core::audio::SampleBuffer::<f32>::new(duration, spec));
                    }

                    if let Some(buf) = &mut sample_buf {
                        // Doesn't seem like plannar (first half is left samples, second half is right) is working
                        buf.copy_interleaved_ref(audio_buf);
                        let reservation = buf.samples().len() / 2;
                        left_samples.reserve(reservation);
                        right_samples.reserve(reservation);

                        for sample in buf.samples().chunks_exact(2) {
                            assert!(sample.len() >= 2);
                            left_samples.push(sample[0]);
                            right_samples.push(sample[1]);
                        }
                    }
                }
                // For some reason, Symphonia is fine if the decode doesn't work?
                // like with malformed data or something
                Err(core::errors::Error::DecodeError(_)) => (),
                Err(_) => break,
            }
        }
    }
    Ok(((left_samples, right_samples), meta_spec))
}

// TODO: add algorithm for arbitrary-length FFT?
fn next_fast_fft(rate: usize) -> usize {
    // RustFFT likes FFT lengths which are powers of 2 multiplied with powers of 3
    // We'll zero-pad the seconds anyway
    match rate {
        // Sample rate is divided by 20
        // Commented values are currently unsupported by Symphonia 0.5.4
        //   but are expected to be supported in dev-0.6
        2205 => 2304, // 44100
        2400 => 2592, // 48000
        // 4410 => 4608,  // 88200
        4800 => 5184, // 96000
        // 8820 => 9216,  // 176400
        // 9600 => 10368, // 192000
        _ => unimplemented!(),
    }
}

fn window(rate: usize) -> Box<[f32]> {
    // Windowing is used to make the signal chunk fade in and out
    //   to prevent discontinuities, which causes spectral leakage (noise tuned to the music).
    // This function uses the Hann window, which is considered to be a jack-of-all-trades.
    // One useful property is that 50% overlapping (i.e. window(n) + window(n + rate/2)) == 1.0, so
    //   we get back to the original level of the input if we overlap two FFTs with this offset between them.
    // Note that window(0, rate) == 0.0 and window(rate - 1, rate) == window(1, rate) != 0.0, so the only zero point is at n == 0.
    //   This is to satisfy some periodicity property/implication, rather than both ends being 0.0
    (0..rate)
        .map(|n| {
            (std::f64::consts::PI * n as f64 / rate as f64)
                .sin()
                .powi(2) as f32
        })
        .collect()
}

// TODO: break up into smaller functions
fn process_samples(
    planner: &mut RealFftPlanner<f32>,
    data: (Vec<f32>, Vec<f32>),
    rate: u32,
) -> (Box<[f32]>, Box<[f32]>) {
    let mut left_channel = data.0;
    let mut right_channel = data.1;
    // Force minimum reconstructed frequency to 20 hz
    let sample_rate = rate as usize / 20;

    // Best to do a small FFT to prevent transient smearing
    let fft_size = next_fast_fft(sample_rate);
    let recip_fft = (fft_size as f64).recip() as f32;
    let r2c = planner.plan_fft_forward(fft_size);
    let c2r = planner.plan_fft_inverse(fft_size);
    let fft_complex_size = r2c.complex_len();

    // The first bin is the DC bin, which is the average unheard noise
    // From what we've seen, this should be a real number (C + 0.0i), but it's better to be safe
    //   by zeroing it out in both axes
    let new_dc = num_complex::Complex::zero();

    // Pre-calculate window function
    let window = window(sample_rate);

    // It's better to add the offset between channels as silence here
    // For the original signal, silence is added to the end, while the
    //   offset signal's silence is added to the beginning
    let original_length = left_channel.len();
    let original_length_inv = (original_length as f64).recip() as f32;
    let offset = sample_rate >> 1; // If using 20 hz, 44.1khz gives 1102.5 samples. Probably insignificant?
    let offset_vec = vec![0.0_f32; offset];
    let mut not_left_channel = offset_vec.clone();
    let mut not_right_channel = offset_vec;
    not_left_channel.append(&mut left_channel.clone());
    not_right_channel.append(&mut right_channel.clone());
    left_channel.resize(original_length + offset, 0.0);
    right_channel.resize(original_length + offset, 0.0);

    // The algorithm I want to use will chunk each signal by sample_rate, so it's better to round up
    //   to the next multiple so we can use ChunksExact and have no remainder
    let fft_total = left_channel.len().next_multiple_of(sample_rate);
    left_channel.resize(fft_total, 0.0);
    right_channel.resize(fft_total, 0.0);
    not_left_channel.resize(fft_total, 0.0);
    not_right_channel.resize(fft_total, 0.0);

    // Turning the Vec's into Box'es will shrink them and prevent further allocations and mutations
    let left_channel = left_channel.into_boxed_slice();
    let right_channel = right_channel.into_boxed_slice();
    let not_left_channel = not_left_channel.into_boxed_slice();
    let not_right_channel = not_right_channel.into_boxed_slice();
    // Chunking for later
    let left_channel = left_channel.chunks_exact(sample_rate);
    let right_channel = right_channel.chunks_exact(sample_rate);
    let not_left_channel = not_left_channel.chunks_exact(sample_rate);
    let not_right_channel = not_right_channel.chunks_exact(sample_rate);

    // Saving samples for later
    // .reserve_exact() reduces memory by preventing over-allocation
    let mut processed_left: Vec<f32> = vec![];
    let mut processed_right: Vec<f32> = vec![];
    let mut processed_not_left: Vec<f32> = vec![];
    let mut processed_not_right: Vec<f32> = vec![];
    processed_left.reserve_exact(fft_total);
    processed_right.reserve_exact(fft_total);
    processed_not_left.reserve_exact(fft_total);
    processed_not_right.reserve_exact(fft_total);

    // Create scratch FFT for RealFFT
    // RealFFT uses RustFFT's .process_with_scratch() for its .process() function
    let mut left_fft = r2c.make_output_vec();
    let mut right_fft = r2c.make_output_vec();
    let mut not_left_fft = r2c.make_output_vec();
    let mut not_right_fft = r2c.make_output_vec();

    for chunk in left_channel
        .zip(right_channel)
        .zip(not_left_channel.zip(not_right_channel))
    {
        let mut left = chunk.0.0.to_vec();
        let mut right = chunk.0.1.to_vec();
        let mut not_left = chunk.1.0.to_vec();
        let mut not_right = chunk.1.1.to_vec();

        assert!(left.len() >= sample_rate);
        assert!(right.len() >= sample_rate);
        assert!(not_left.len() >= sample_rate);
        assert!(not_right.len() >= sample_rate);
        for index in 0..sample_rate {
            let window_multiplier = window[index];
            left[index] *= window_multiplier;
            right[index] *= window_multiplier;
            not_left[index] *= window_multiplier;
            not_right[index] *= window_multiplier;
        }

        // Zero-pad signal for FFT
        // It's probably fine not to shrink these, since they're rather small and would reallocate
        //   in this hot loop
        left.resize(fft_size, 0.0);
        right.resize(fft_size, 0.0);
        not_left.resize(fft_size, 0.0);
        not_right.resize(fft_size, 0.0);

        // Ignore errors by RealFFT
        // RustFFT does not return a Result after processing,
        //   but RealFFT does return Results due to some zero-check
        //   RealFFT author says to just ignore these in the meantime.
        // https://github.com/HEnquist/realfft/issues/41#issuecomment-2050347470
        let _ = r2c.process(&mut left, &mut left_fft);
        let _ = r2c.process(&mut right, &mut right_fft);
        let _ = r2c.process(&mut not_left, &mut not_left_fft);
        let _ = r2c.process(&mut not_right, &mut not_right_fft);

        assert!(left_fft.len() >= fft_complex_size);
        assert!(right_fft.len() >= fft_complex_size);
        assert!(not_left_fft.len() >= fft_complex_size);
        assert!(not_right_fft.len() >= fft_complex_size);

        // Remove local DC offset
        left_fft[0] = new_dc;
        right_fft[0] = new_dc;
        not_left_fft[0] = new_dc;
        not_right_fft[0] = new_dc;

        for index in 1..fft_complex_size {
            // Align the phase of the left and right channels using the circular mean / true midpoint
            // Higher weight towards higher magnitude, so the channel with the higher magnitude doesn't rotate much,
            //   while the smaller magnitude channel may rotate a lot

            // Circular mean / true midpoint
            let sum = left_fft[index] + right_fft[index];
            let not_sum = not_left_fft[index] + not_right_fft[index];

            // Squares without .sqrt until later
            let sum_sqr_inv = sum.norm_sqr().recip();
            let left_sqr = left_fft[index].norm_sqr();
            let right_sqr = right_fft[index].norm_sqr();
            let not_sum_sqr_inv = not_sum.norm_sqr().recip();
            let not_left_sqr = not_left_fft[index].norm_sqr();
            let not_right_sqr = not_right_fft[index].norm_sqr();

            // Ensure no division by zero or similar
            if sum_sqr_inv.is_normal() {
                // Equivalent to using cos-sin of atan2(sum)
                left_fft[index] = sum.scale((left_sqr * sum_sqr_inv).sqrt());
                right_fft[index] = sum.scale((right_sqr * sum_sqr_inv).sqrt());
            } else {
                let sum_norm_inv = sum_sqr_inv.sqrt();
                let left_norm = left_sqr.sqrt();
                let right_norm = right_sqr.sqrt();
                // Check if the division by close-to-zero is recoverable by taking the square root
                if sum_norm_inv.is_normal() {
                    left_fft[index] = sum.scale(left_norm * sum_norm_inv);
                    right_fft[index] = sum.scale(right_norm * sum_norm_inv);
                } else if left_fft[index].re.abs() < left_fft[index].im.abs() {
                    // If there would be a division by zero, the sum coordinate is near the origin of 0.0 + 0.0i, so
                    //   the left and right channels are either silence or are completely out-of-phase
                    // This block makes the channels land on the Im axis if the Re coordinate is small
                    left_fft[index].re = 0.0;
                    left_fft[index].im = left_norm;
                    right_fft[index].re = 0.0;
                    right_fft[index].im = right_norm;
                } else {
                    // aka left_fft[index].im.abs() <= left_fft[index].re.abs(),
                    //   so we should land on the Re axis. This emulates the behavior of .atan2() where the angle is 0 radians
                    left_fft[index].re = left_norm;
                    left_fft[index].im = 0.0;
                    right_fft[index].re = right_norm;
                    right_fft[index].im = 0.0;
                }
            }
            // ^
            if not_sum_sqr_inv.is_normal() {
                not_left_fft[index] = not_sum.scale((not_left_sqr * not_sum_sqr_inv).sqrt());
                not_right_fft[index] = not_sum.scale((not_right_sqr * not_sum_sqr_inv).sqrt());
            } else {
                let not_sum_norm_inv = not_sum_sqr_inv.sqrt();
                let not_left_norm = not_left_sqr.sqrt();
                let not_right_norm = not_right_sqr.sqrt();
                if not_sum_norm_inv.is_normal() {
                    not_left_fft[index] = not_sum.scale(not_left_norm * not_sum_norm_inv);
                    not_right_fft[index] = not_sum.scale(not_right_norm * not_sum_norm_inv);
                } else if not_left_fft[index].re.abs() < not_left_fft[index].im.abs() {
                    not_left_fft[index].re = 0.0;
                    not_left_fft[index].im = not_left_norm;
                    not_right_fft[index].re = 0.0;
                    not_right_fft[index].im = not_right_norm;
                } else {
                    not_left_fft[index].re = not_left_norm;
                    not_left_fft[index].im = 0.0;
                    not_right_fft[index].re = not_right_norm;
                    not_right_fft[index].im = 0.0;
                }
            }
        }

        let _ = c2r.process(&mut left_fft, &mut left);
        let _ = c2r.process(&mut right_fft, &mut right);
        let _ = c2r.process(&mut not_left_fft, &mut not_left);
        let _ = c2r.process(&mut not_right_fft, &mut not_right);

        // FFT zero-padding is ignored with this `for`` loop
        for index in 0..sample_rate {
            // Normalize FFT values to finish them off
            processed_left.push(left[index] * recip_fft);
            processed_right.push(right[index] * recip_fft);
            processed_not_left.push(not_left[index] * recip_fft);
            processed_not_right.push(not_right[index] * recip_fft);
        }
    }

    drop(left_fft);
    drop(right_fft);
    drop(not_left_fft);
    drop(not_right_fft);
    drop(r2c);
    drop(c2r);

    assert!(processed_left.len() >= original_length);
    assert!(processed_right.len() >= original_length);
    assert!(processed_not_left.len() >= fft_total);
    assert!(processed_not_right.len() >= fft_total);

    // Add the original and offset signals together to get the unwindowed level
    for index in 0..original_length {
        processed_left[index] += processed_not_left[index + offset];
        processed_right[index] += processed_not_right[index + offset];
    }
    drop(processed_not_left);
    drop(processed_not_right);

    // Remove chunking zero-padding
    processed_left.truncate(original_length);
    processed_right.truncate(original_length);
    processed_left.shrink_to(original_length);
    processed_right.shrink_to(original_length);

    assert!(processed_left.len() >= original_length);
    assert!(processed_right.len() >= original_length);
    // Remove overall DC after all local DC was removed
    // DC is just the average of the whole signal
    let left_dc = processed_left.clone().iter().sum::<f32>() * original_length_inv;
    let right_dc = processed_right.clone().iter().sum::<f32>() * original_length_inv;
    for index in 0..original_length {
        processed_left[index] -= left_dc;
        processed_right[index] -= right_dc;
    }

    // Average out the RMS of the left and right channels
    // No need to divide by original_length to get the mean, nor take the square root,
    //   since the divisions cancel out later and the square root is made later once
    let left_s = processed_left
        .clone()
        .iter()
        .fold(0.0, |acc, s| s.mul_add(*s, acc));
    let right_s = processed_right
        .clone()
        .iter()
        .fold(0.0, |acc, s| s.mul_add(*s, acc));
    // First square root is to get the multiplier when applied to s^2,
    //   second square root is to get the multiplier when applied to just s.
    let left_equalizer = (right_s / left_s).powf(0.25);
    let right_equalizer = (left_s / right_s).powf(0.25);

    for index in 0..original_length {
        processed_left[index] *= left_equalizer;
        processed_right[index] *= right_equalizer;
    }

    // Overall processing is done
    // pack it up
    (
        processed_left.into_boxed_slice(),
        processed_right.into_boxed_slice(),
    )
}

fn save_audio(file_path: &std::path::Path, audio: &(Box<[f32]>, Box<[f32]>), rate: u32) {
    // TODO: add simple functionality for mono signals?
    // Might be a lot of work for something you can re-render to stereo in foobar2000
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: rate,
        bits_per_sample: 32, // hound only supports 32-bit float
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(file_path, spec).expect("Could not create writer");
    let length = audio.0.len();

    assert!(audio.0.len() >= length);
    assert!(audio.1.len() >= length);
    for index in 0..length {
        writer
            .write_sample(audio.0[index])
            .expect("Could not write sample");
        writer
            .write_sample(audio.1[index])
            .expect("Could not write sample");
    }
    writer.finalize().expect("Could not finalize WAV file");
}

fn get_paths(directory: path::PathBuf) -> io::Result<Vec<path::PathBuf>> {
    // Recursive file path retriever from StackOverflow
    // TODO: check if there's a newer std function or crate to do this
    let mut entries: Vec<path::PathBuf> = vec![];
    let folder_read = fs::read_dir(directory)?;

    for entry in folder_read {
        let entry = entry?;
        let meta = entry.metadata()?;

        if meta.is_dir() {
            let mut subdir = get_paths(entry.path())?;
            entries.append(&mut subdir);
        }

        if meta.is_file() {
            entries.push(entry.path());
        }
    }
    Ok(entries)
}

fn main() -> Result<(), core::errors::Error> {
    // Keeping the time for benchmarking
    let time = time::Instant::now();

    // Check if INPUT_DIR exists, or create it if it doesn't
    match fs::exists(INPUT_DIR) {
        Ok(true) => {}
        Ok(false) => {
            let _ = fs::create_dir(INPUT_DIR);
            println!("Notice: Inputs folder created. Copy audio files here to process them.");
            return Ok(());
        }
        // Symphonia has a wrapper for IoErrors
        Err(err) => return Err(core::errors::Error::IoError(err)),
    }

    // Get list of files in INPUT_DIR
    let entries: Vec<path::PathBuf> = get_paths(INPUT_DIR.into())?;

    let mut real_planner = RealFftPlanner::<f32>::new();

    println!("Setup and file-exploring time: {:#?}", time.elapsed());
    for entry in entries {
        println!("Found file: {}", entry.display());
        print!("    Decoding...");
        io::stdout().flush()?; // Show print instantly
        let mut output_path = path::PathBuf::new();
        let unprefix_output = entry.strip_prefix(INPUT_DIR).unwrap();
        output_path.push(OUTPUT_DIR);
        output_path.push(unprefix_output);

        let channels: (Vec<f32>, Vec<f32>);
        let sample_rate: u32;
        // If we can't properly decode the data, it's likely not audio data
        // In this case, we just send it to OUTPUT_DIR so it's not spitting the warning every run
        match get_samples_and_metadata(&entry) {
            Ok(data) => {
                channels = data.0;
                sample_rate = data.1;
            }
            // The following errors usually happen if Symphonia attempts to open a .jpg or .png
            Err(core::errors::Error::IoError(err)) => {
                if err.kind() == io::ErrorKind::UnexpectedEof {
                    println!("  Invalid or unsupported audio, sent to output.");
                    fs::create_dir_all(output_path.parent().unwrap())?;
                    fs::rename(entry, output_path)?;
                    continue;
                }
                return Err(core::errors::Error::IoError(err)); // Except here, where an actual audio file failed decoding
            }
            Err(core::errors::Error::Unsupported(_) | core::errors::Error::DecodeError(_)) => {
                println!("    Invalid or unsupported audio, sent to output");
                fs::create_dir_all(output_path.parent().unwrap())?;
                fs::rename(entry, output_path)?;
                continue;
            }
            Err(other) => return Err(other), // some other unknown error
        }

        print!("    Processing... ");
        io::stdout().flush()?;
        let modified_audio = process_samples(&mut real_planner, channels, sample_rate);

        print!("    Saving...");
        io::stdout().flush()?;
        output_path.set_extension("wav");
        fs::create_dir_all(output_path.parent().unwrap())?;
        save_audio(&output_path, &modified_audio, sample_rate);

        println!("    T+{:#?} ", time.elapsed());
    }
    Ok(())
}
