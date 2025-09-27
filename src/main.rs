// Allow-list for non-restriction lints
#![allow(
    clippy::cast_precision_loss,
    reason = "allow u64 as f64, likey no precision loss"
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "allow using f64 for precision, then truncating to f32"
)]
// Modifications to restriction lints
#![allow(
    clippy::as_conversions,
    reason = "some conversions are required, other cast_* lints are still used"
)]
#![allow(
    clippy::blanket_clippy_restriction_lints,
    reason = "whitelist seens better than blacklist"
)]
#![allow(clippy::cast_sign_loss, reason = "only positive/unsigned values used")]
#![allow(clippy::float_arithmetic, reason = "float arithmetic is required")]
#![allow(clippy::implicit_return, reason = "make things rusty")]
#![allow(clippy::print_stdout, reason = "printing to console is required")]
#![allow(
    clippy::question_mark_used,
    reason = "no need to add additional error info"
)]
#![allow(clippy::semicolon_outside_block, reason = "personal preference")]
#![allow(clippy::separated_literal_suffix, reason = "personal preference")]
#![allow(
    clippy::single_call_fn,
    reason = "allow breaking long functions into smaller functions"
)]
// #![allow(clippy::unwrap_used)]
#![allow(clippy::use_debug, reason = "pretty print is good enough")]

// Imports
use core::f64::consts::PI;
use ebur128::{EbuR128, Mode};
use itertools::izip;
use realfft::{
    RealFftPlanner,
    num_complex::{self, Complex},
    num_traits::Zero,
};
use std::{
    fs,
    io::{self, Write as _},
    path::{self, Path},
    time,
};
use symphonia::{
    core::{
        audio::SampleBuffer,
        codecs::DecoderOptions,
        errors::Error,
        formats::FormatOptions,
        io::{MediaSourceStream, MediaSourceStreamOptions},
        meta::MetadataOptions,
        probe::Hint,
    },
    default,
};

// Hard-coded directories
/// Input directory, created on first run
const INPUT_DIR: &str = "./inputs/";
/// Output directory
const OUTPUT_DIR: &str = "./outputs/";

/// Seperated here due to Clippy lint
type AudioMatrix = ((Vec<f32>, Vec<f32>), u32);
/// Get samples and metadata for a given file using `Symphonia`
fn get_samples_and_metadata(path: &path::PathBuf) -> Result<AudioMatrix, Error> {
    // Based on `Symphonia`'s docs.rs page and example code (mix of 0.5.4 and dev-0.6)
    // Numbers are from the `Symphonia` basic proceedures in its docs.rs

    // 1
    let code_registry = default::get_codecs();
    // 2
    let probe = default::get_probe();

    // 3
    // 4
    let mss = MediaSourceStream::new(
        Box::new(fs::File::open(path)?),
        MediaSourceStreamOptions::default(),
    );

    // 5
    // 6
    let probe_result = probe.format(
        Hint::new().with_extension("flac"),
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

                        // SAFETY: usize to u64 should never fail
                        let duration: u64 =
                            unsafe { audio_buf.capacity().try_into().unwrap_unchecked() };

                        sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                    }

                    if let Some(ref mut buf) = sample_buf {
                        buf.copy_planar_ref(audio_buf);
                        let length = buf.samples().len();
                        // Unsure if reserving space is necessary due to extend_from_slice later
                        let reservation = length >> 1_usize; // div by 2
                        left_samples.reserve(reservation);
                        right_samples.reserve(reservation);

                        // SAFETY: reservation < length
                        let (left, right) =
                            unsafe { buf.samples().split_at_unchecked(reservation) };
                        left_samples.extend_from_slice(left);
                        right_samples.extend_from_slice(right);
                    }
                }
                // For some reason, `Symphonia` is fine if the decode doesn't work?
                // like with malformed data or something
                Err(Error::DecodeError(_)) => (),
                Err(_) => break,
            }
        }
    }
    // No need to call .shrink_to_fit or care about over-allocation
    // TODO: return error if fft_total would be larger than usize::MAX
    Ok(((left_samples, right_samples), meta_spec))
}

// TODO: add algorithm for arbitrary-length FFT?
/// Given a sample rate, get the best FFT size for `RustFFT` to process
/// `RustFFT` likes FFT lengths which are powers of 2 multiplied with powers of 3
/// We'll zero-pad the seconds anyway
fn next_fast_fft(rate: usize) -> usize {
    #[expect(
        clippy::unimplemented,
        reason = "`symphonia` 0.5.4 only supports 3 sample rates"
    )]
    match rate {
        // Sample rate is divided by 20
        // Commented values are currently unsupported by `Symphonia` 0.5.4
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

/// Windowing is used to make the signal chunk fade in and out
///   to prevent discontinuities, which causes spectral leakage (noise tuned to the music).
/// This function uses the Hann window, which is considered to be a jack-of-all-trades.
/// One useful property is that 50% overlapping (i.e. window(n) + window(n + rate/2)) == 1.0, so
///   we get back to the original level of the input if we overlap two FFTs with this offset between them.
/// Note that window(0, rate) == 0.0 and window(rate - 1, rate) == window(1, rate) != 0.0, so the only zero point is at n == 0.
///   This is to satisfy some periodicity property/implication, rather than both ends being 0.0
fn window(rate: usize) -> Box<[f32]> {
    (0..rate)
        .map(|n| (PI * n as f64 / rate as f64).sin().powi(2) as f32)
        .collect()
}

/// Align the phase of the left and right channels using the circular mean / true midpoint
/// Using this method makes the resulting phase match the downmixed signal phase (left + right / 2),
///   i.e. zero-crossings should match with mid channel
/// Higher weight towards higher magnitude, so the channel with the higher magnitude doesn't rotate much,
///   while the smaller magnitude channel may rotate a lot
fn align(left: &mut Complex<f32>, right: &mut Complex<f32>) {
    // Circular mean / true midpoint
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "clippy thinks this is an integer add, whereas f32 will saturate to the infinities"
    )]
    let sum = *left + *right;

    // Squares without .sqrt until later
    let sum_sqr_inv = sum.norm_sqr().recip();
    let left_sqr = left.norm_sqr();
    let right_sqr = right.norm_sqr();

    // Prevent propagation of NaN or infinity
    // Used to be .is_normal(), but subnormal values aren't possible since
    //   we call .recip() beforehand (i.e. 1.0/f32::MIN_POSITIVE, 1.0/f32::MAX, 1.0/f32::INFINITY == 0.0, 1.0/0.0, 1.0/f32::NAN)
    if sum_sqr_inv.is_finite() {
        // Equivalent to using cos-sin of atan2(sum)
        *left = sum.scale((left_sqr * sum_sqr_inv).sqrt());
        *right = sum.scale((right_sqr * sum_sqr_inv).sqrt());
    } else {
        let sum_norm_inv = sum.norm().recip();
        let left_norm = left_sqr.sqrt();
        let right_norm = right_sqr.sqrt();
        // Check if sum_sqr_inv is recoverable by taking the square root before .recip()
        if sum_norm_inv.is_finite() {
            *left = sum.scale(left_norm * sum_norm_inv);
            *right = sum.scale(right_norm * sum_norm_inv);
        } else {
            // If we get here, the sum coordinate is near the origin of 0.0 + 0.0i, so
            //   the left and right channels are either silence or are completely out-of-phase
            // Since we want to keep the zero-crossings, we just have the right copy the left
            // Implicitly, *left = *left
            *right = *left;
        }
    }
}

/// EBU R 128 Integrated Loudness calculation
/// Basically a two-pass windowed RMS.
///   First pass is used to detect and ignore silence at -70dB
///   Second pass is used to detect and ignore quieter points at the first-pass RMS minus 10dB
fn gated_rms(samples: &Vec<f32>, rate: u32) -> f64 {
    let mut ebur128 = EbuR128::new(1_u32, rate, Mode::I).expect("Shouldn't happen");
    // Planar sucks since it requires an array of channel arrays, so for one channel, it needs an array around it
    // f64 samples are not needed
    _ = ebur128.add_frames_f32(&samples.to_owned());
    let loudness = ebur128.loudness_global().expect("Shouldn't happen");

    10.0_f64.powf(loudness / 20.0_f64)
}

// TODO: break up into smaller functions
/// All three processing steps into one function
fn process_samples(
    planner: &mut RealFftPlanner<f32>,
    data: (Vec<f32>, Vec<f32>),
    rate: u32,
) -> (Box<[f32]>, Box<[f32]>) {
    let mut left_channel = data.0;
    let mut right_channel = data.1;
    // Force minimum reconstructed frequency to 20 hz
    let sample_rate = (f64::from(rate) / 20.0_f64) as usize;

    // Best to do a small FFT to prevent transient smearing
    let fft_size = next_fast_fft(sample_rate);
    let recip_fft = (fft_size as f64).recip() as f32;
    let r2c = planner.plan_fft_forward(fft_size);
    let c2r = planner.plan_fft_inverse(fft_size);
    let fft_complex_size = r2c.complex_len();

    // Pre-calculate window function
    let window = window(sample_rate);

    // For the original signal, silence is added to the end, while the
    //   offset signal's silence is added to the beginning
    // When using 20 hz, 44.1khz needs 1102.5 samples of silence, which becomes 1102
    // The effect is insignificant anyways, giving a 0.0712% or 0.00619dB difference
    // TODO: offset can technically fail original_length is near usize::MAX
    //   would happen if usize == u32 and decoded a 12-hour 96khz file
    let original_length = left_channel.len();
    let offset = sample_rate >> 1_usize;
    let offset_vec = vec![0.0_f32; offset];
    let mut not_left_channel = offset_vec.clone();
    let mut not_right_channel = offset_vec;
    not_left_channel.append(&mut left_channel.clone());
    not_right_channel.append(&mut right_channel.clone());
    let offset_total = not_left_channel.len();
    // assert_eq!(offset_total, original_length + offset);
    left_channel.resize(offset_total, 0.0);
    right_channel.resize(offset_total, 0.0);

    // The algorithm I want to use will chunk each signal by sample_rate, so it's better to round up
    //   to the next multiple so we can use ChunksExact and have no remainder
    let fft_total = offset_total.next_multiple_of(sample_rate);
    left_channel.resize(fft_total, 0.0);
    right_channel.resize(fft_total, 0.0);
    not_left_channel.resize(fft_total, 0.0);
    not_right_channel.resize(fft_total, 0.0);

    // Turning the Vec's into Box'es will shrink them and prevent further allocations and mutations
    let left_channel_box = left_channel.into_boxed_slice();
    let right_channel_box = right_channel.into_boxed_slice();
    let not_left_channel_box = not_left_channel.into_boxed_slice();
    let not_right_channel_box = not_right_channel.into_boxed_slice();
    // Chunking for later
    let left_channel_chunks = left_channel_box.chunks_exact(sample_rate);
    let right_channel_chunks = right_channel_box.chunks_exact(sample_rate);
    let not_left_channel_chunks = not_left_channel_box.chunks_exact(sample_rate);
    let not_right_channel_chunks = not_right_channel_box.chunks_exact(sample_rate);

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

    // Create scratch FFT for `RealFFT`
    // `RealFFT` uses `RustFFT`'s .process_with_scratch() for its .process() function
    let mut left_fft = r2c.make_output_vec();
    let mut right_fft = r2c.make_output_vec();
    let mut not_left_fft = r2c.make_output_vec();
    let mut not_right_fft = r2c.make_output_vec();

    for (left_chunk, right_chunk, not_left_chunk, not_right_chunk) in izip!(
        left_channel_chunks,
        right_channel_chunks,
        not_left_channel_chunks,
        not_right_channel_chunks
    ) {
        // Seems a bit faster than using .resize()
        // Repetition is used instead of cloning a premade Vec since .clone() clogs up the flamegraph
        let mut left = vec![0.0_f32; fft_size.strict_sub(sample_rate)];
        let mut right = vec![0.0_f32; fft_size.strict_sub(sample_rate)];
        let mut not_left = vec![0.0_f32; fft_size.strict_sub(sample_rate)];
        let mut not_right = vec![0.0_f32; fft_size.strict_sub(sample_rate)];
        left.extend(left_chunk);
        right.extend(right_chunk);
        not_left.extend(not_left_chunk);
        not_right.extend(not_right_chunk);

        for index in 0..sample_rate {
            // SAFETY: index never exceeds sample_rate - 1
            let window_multiplier = *unsafe { window.get_unchecked(index) };
            // SAFETY: index never exceeds sample_rate - 1
            *unsafe { left.get_unchecked_mut(index) } *= window_multiplier;
            // SAFETY: index never exceeds sample_rate - 1
            *unsafe { right.get_unchecked_mut(index) } *= window_multiplier;
            // SAFETY: index never exceeds sample_rate - 1
            *unsafe { not_left.get_unchecked_mut(index) } *= window_multiplier;
            // SAFETY: index never exceeds sample_rate - 1
            *unsafe { not_right.get_unchecked_mut(index) } *= window_multiplier;
        }

        // Ignore errors by `RealFFT`
        // `RustFFT` does not return a Result after processing,
        //   but `RealFFT` does return Results due to some zero-check
        //   `RealFFT` author says to just ignore these in the meantime.
        // https://github.com/HEnquist/realfft/issues/41#issuecomment-2050347470
        _ = r2c.process(&mut left, &mut left_fft);
        _ = r2c.process(&mut right, &mut right_fft);
        _ = r2c.process(&mut not_left, &mut not_left_fft);
        _ = r2c.process(&mut not_right, &mut not_right_fft);

        // Remove DC offset in case the noise changes over time
        // The first bin is the DC bin, which is the average unheard noise
        // From what we've seen, this should be a real number (C + 0.0i), but it's better to be safe
        //   by zeroing it out in both axes
        // SAFETY: first element must exist
        *unsafe { left_fft.get_unchecked_mut(0) } = num_complex::Complex::zero();
        // SAFETY: first element must exist
        *unsafe { right_fft.get_unchecked_mut(0) } = num_complex::Complex::zero();
        // SAFETY: first element must exist
        *unsafe { not_left_fft.get_unchecked_mut(0) } = num_complex::Complex::zero();
        // SAFETY: first element must exist
        *unsafe { not_right_fft.get_unchecked_mut(0) } = num_complex::Complex::zero();

        for index in 1..fft_complex_size {
            // SAFETY: index guaranteed to be less than fft length, which is fft_complex_size
            let left_fft_index = unsafe { left_fft.get_unchecked_mut(index) };
            // SAFETY: index guaranteed to be less than fft length, which is fft_complex_size
            let right_fft_index = unsafe { right_fft.get_unchecked_mut(index) };
            // SAFETY: index guaranteed to be less than fft length, which is fft_complex_size
            let not_left_fft_index = unsafe { not_left_fft.get_unchecked_mut(index) };
            // SAFETY: index guaranteed to be less than fft length, which is fft_complex_size
            let not_right_fft_index = unsafe { not_right_fft.get_unchecked_mut(index) };
            align(left_fft_index, right_fft_index);
            align(not_left_fft_index, not_right_fft_index);
        }

        _ = c2r.process(&mut left_fft, &mut left);
        _ = c2r.process(&mut right_fft, &mut right);
        _ = c2r.process(&mut not_left_fft, &mut not_left);
        _ = c2r.process(&mut not_right_fft, &mut not_right);

        // FFT zero-padding is ignored with this `for`` loop
        for index in 0..sample_rate {
            // SAFETY: sample rate is less than left.len, so index should always work
            let left_index = *unsafe { left.get_unchecked(index) };
            // SAFETY: sample rate is less than left.len, so index should always work
            let right_index = *unsafe { right.get_unchecked(index) };
            // SAFETY: sample rate is less than left.len, so index should always work
            let not_left_index = *unsafe { not_left.get_unchecked(index) };
            // SAFETY: sample rate is less than left.len, so index should always work
            let not_right_index = *unsafe { not_right.get_unchecked(index) };
            // Normalize FFT values to finish them off
            processed_left.push(left_index * recip_fft);
            processed_right.push(right_index * recip_fft);
            processed_not_left.push(not_left_index * recip_fft);
            processed_not_right.push(not_right_index * recip_fft);
        }
    }

    drop(left_fft);
    drop(right_fft);
    drop(not_left_fft);
    drop(not_right_fft);
    drop(r2c);
    drop(c2r);

    // Add the original and offset signals together to get the unwindowed level
    for index in 0..original_length {
        // TODO: add checking for usize length audio
        // SAFETY: index + offset < fft_total
        let not_index = unsafe { index.unchecked_add(offset) };
        // SAFETY: index + offset < fft_total
        *unsafe { processed_left.get_unchecked_mut(index) } +=
            *unsafe { processed_not_left.get_unchecked(not_index) };
        // SAFETY: index + offset < fft_total
        *unsafe { processed_right.get_unchecked_mut(index) } +=
            *unsafe { processed_not_right.get_unchecked(not_index) };
    }
    drop(processed_not_left);
    drop(processed_not_right);

    // Remove chunking zero-padding
    processed_left.truncate(original_length);
    processed_right.truncate(original_length);
    processed_left.shrink_to(original_length);
    processed_right.shrink_to(original_length);

    // Remove overall DC after all local DC was removed
    // DC is just the average of the whole signal
    let left_dc = processed_left.clone().iter().sum::<f32>() / original_length as f32;
    let right_dc = processed_right.clone().iter().sum::<f32>() / original_length as f32;
    for index in 0..original_length {
        // SAFETY: index bounds check never fails
        *unsafe { processed_left.get_unchecked_mut(index) } -= left_dc;
        // SAFETY: index bounds check never fails
        *unsafe { processed_right.get_unchecked_mut(index) } -= right_dc;
    }

    // Average out the loudness of the left and right channels
    // Need to .sqrt() the RMS to get the per-sample multiplier instead of the per-RMS multiplier
    let left_rms_sqrt = gated_rms(&processed_left, rate).sqrt();
    let right_rms_sqrt = gated_rms(&processed_right, rate).sqrt();
    let left_equalizer = (right_rms_sqrt / left_rms_sqrt) as f32;
    let right_equalizer = (left_rms_sqrt / right_rms_sqrt) as f32;
    for index in 0..original_length {
        // SAFETY: index bounds check never fails
        *unsafe { processed_left.get_unchecked_mut(index) } *= left_equalizer;
        // SAFETY: index bounds check never fails
        *unsafe { processed_right.get_unchecked_mut(index) } *= right_equalizer;
    }

    // Overall processing is done
    // pack it up
    (
        processed_left.into_boxed_slice(),
        processed_right.into_boxed_slice(),
    )
}

/// Save processed audio to the output using `hound`
fn save_audio(file_path: &Path, audio: &(Box<[f32]>, Box<[f32]>), rate: u32) {
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

    for index in 0..length {
        // SAFETY: index guaranteed to be within length
        writer
            .write_sample(*unsafe { audio.0.get_unchecked(index) })
            .expect("Could not write sample");
        // SAFETY: index guaranteed to be within length
        writer
            .write_sample(*unsafe { audio.1.get_unchecked(index) })
            .expect("Could not write sample");
    }
    writer.finalize().expect("Could not finalize WAV file");
}

/// Recursive file path retriever from `StackOverflow`
/// TODO: check if there's a newer std function or crate to do this
fn get_paths(directory: path::PathBuf) -> io::Result<Vec<path::PathBuf>> {
    let mut entries: Vec<path::PathBuf> = vec![];
    let folder_read = fs::read_dir(directory)?;

    for entry in folder_read {
        let unwrapped_entry = entry?;
        let meta = unwrapped_entry.metadata()?;

        if meta.is_dir() {
            let mut subdir = get_paths(unwrapped_entry.path())?;
            entries.append(&mut subdir);
        }

        if meta.is_file() {
            entries.push(unwrapped_entry.path());
        }
    }
    Ok(entries)
}

/// Main function to execute
fn main() -> Result<(), Error> {
    // Keeping the time for benchmarking
    let time = time::Instant::now();

    // Check if INPUT_DIR exists, or create it if it doesn't
    match fs::exists(INPUT_DIR) {
        Ok(true) => {}
        Ok(false) => {
            _ = fs::create_dir_all(INPUT_DIR);
            println!("Notice: Inputs folder created. Copy audio files here to process them.");
            return Ok(());
        }
        // `Symphonia` has a wrapper for IoErrors
        Err(err) => return Err(Error::IoError(err)),
    }

    // Get list of files in INPUT_DIR
    let entries: Vec<path::PathBuf> = get_paths(INPUT_DIR.into())?;

    let mut real_planner = RealFftPlanner::<f32>::new();

    println!("Setup and file-exploring time: {:#?}", time.elapsed());
    for entry in entries {
        println!("Found file: {}", entry.display());
        print!("    Decoding...");
        io::stdout().flush()?; // Show print instantly
        let mut output_path =
            path::PathBuf::from(OUTPUT_DIR).join(entry.strip_prefix(INPUT_DIR).unwrap());

        let channels: (Vec<f32>, Vec<f32>);
        let sample_rate: u32;
        // If we can't properly decode the data, it's likely not audio data
        // In this case, we just send it to OUTPUT_DIR so it's not spitting the warning every run
        match get_samples_and_metadata(&entry) {
            Ok(data) => {
                channels = data.0;
                sample_rate = data.1;
            }
            // The following errors usually happen if `Symphonia` attempts to open a .jpg or .png
            Err(Error::IoError(err)) => {
                if err.kind() == io::ErrorKind::UnexpectedEof {
                    println!("  Invalid or unsupported audio, sent to output.");
                    fs::create_dir_all(output_path.parent().unwrap())?;
                    fs::rename(entry, output_path)?;
                    continue;
                }
                return Err(Error::IoError(err)); // Except here, where an actual audio file failed decoding
            }
            Err(Error::Unsupported(_) | Error::DecodeError(_)) => {
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
