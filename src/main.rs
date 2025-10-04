// Imports
use core::f64::consts::PI;
use ebur128::{EbuR128, Mode};
use itertools::{Itertools as _, izip};
use realfft::{RealFftPlanner, num_complex::Complex};
use std::{
    fs,
    io::{self, Write as _},
    path::{self, Path},
    time,
};
use symphonia::{
    core::{
        codecs::audio::AudioDecoderOptions,
        errors::Error,
        formats::{FormatOptions, TrackType, probe::Hint},
        io::{MediaSourceStream, MediaSourceStreamOptions},
        meta::MetadataOptions,
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
    let mut format = probe.probe(
        Hint::new().with_extension("flac"),
        mss,
        FormatOptions::default(),
        MetadataOptions::default(),
    )?;

    // 7
    let track = format.default_track(TrackType::Audio).unwrap();

    // 8
    let mut decoder = code_registry.make_audio_decoder(
        // Don't know why the two unwraps are
        track.codec_params.as_ref().unwrap().audio().unwrap(),
        &AudioDecoderOptions::default(),
    )?;

    let track_id = track.id;

    let mut left_samples: Vec<f32> = vec![];
    let mut right_samples: Vec<f32> = vec![];

    let mut sample_buf: Vec<f32> = vec![];
    let mut sample_rate = 0;

    // 9
    // 10
    // 11
    while let Ok(Some(packet)) = format.next_packet() {
        if packet.track_id() == track_id {
            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    if sample_rate == 0 {
                        sample_rate = audio_buf.spec().rate();
                    }
                    // The API for planar samples sucks
                    sample_buf.resize(audio_buf.samples_interleaved(), 0.0);
                    audio_buf.copy_to_slice_interleaved(&mut sample_buf);
                    sample_buf.chunks_exact(2).for_each(|chunk| {
                        // SAFETY: chunk.len() > 1
                        left_samples.push(*unsafe { chunk.get_unchecked(0) });
                        // SAFETY: chunk.len() > 1
                        right_samples.push(*unsafe { chunk.get_unchecked(1) });
                    });
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
    Ok(((left_samples, right_samples), sample_rate))
}

// TODO: add algorithm for arbitrary-length FFT?
/// Given a sample rate, get the best FFT size for `RustFFT` to process
/// `RustFFT` likes FFT lengths which are powers of 2 multiplied with powers of 3
/// We'll zero-pad the seconds anyway
fn next_fast_fft(rate: usize) -> usize {
    match rate {
        4410 => 4608,   // 44_100
        4800 => 5184,   // 48_000
        8820 => 9216,   // 88_200
        9600 => 10368,  // 96_000
        17640 => 18432, // 176_400
        19200 => 19683, // 192_000
        _ => unimplemented!(),
    }
}

/// Windowing is used to make the signal chunk fade in and out
///   to prevent discontinuities, which causes spectral leakage (noise tuned to the music).
/// This function uses the Hann window, which is considered to be a jack-of-all-trades.
/// One useful property is that 50% overlapping (i.e. window(n) + window(n + rate/2)) == 1.0, so
///   we get back to the original level of the input if we overlap two FFTs with this offset between them.
/// Zero points are added to the beginning and end of the window at the FFT so some perodicity property is satisfied
/// Here, no values should equal zero to prevent lost information
fn window(rate: usize) -> Box<[f32]> {
    (0..=rate)
        .map(|n| {
            (PI * n as f64 / rate.strict_add(1_usize) as f64)
                .sin()
                .powi(2) as f32
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

/// Two-channel FFT processing
fn fft_process(
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

    // Turning the Vec's into Box'es will shrink them and prevent further allocations and mutations
    let left_channel_box = left_channel.into_boxed_slice();
    let right_channel_box = right_channel.into_boxed_slice();
    // Chunking for later
    let left_channel_chunks = left_channel_box.chunks_exact(rate);
    let right_channel_chunks = right_channel_box.chunks_exact(rate);

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

    izip!(left_channel_chunks, right_channel_chunks,).for_each(|(left_chunk, right_chunk)| {
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
        izip!(left_fft.iter_mut(), right_fft.iter_mut(),)
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

        izip!(left.iter_mut(), right.iter_mut(),).for_each(|(left_samp, right_samp)| {
            *left_samp *= recip_fft;
            *right_samp *= recip_fft;
        });

        // Scratch Vec's are cleared by these lines
        processed_left.append(&mut left);
        processed_right.append(&mut right);
    });

    (processed_left, processed_right)
}

/// Specific overlapping
fn overlap(
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

/// All three processing steps into one function
fn process_samples(data: (Vec<f32>, Vec<f32>), sample_rate: u32) -> (Box<[f32]>, Box<[f32]>) {
    let mut left_channel = data.0;
    let mut right_channel = data.1;
    let original_length = left_channel.len();
    // Force minimum reconstructed frequency to N hertz
    let f64_rate = f64::from(sample_rate) / 10.0_f64; // Testing with 10 Hz
    let rate = f64_rate as usize;

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
    // For some reason, even denominators should be used, as
    //   odd denominators (e.g. 3) cause a vibrato effect
    // Also for some reason reduces peak levels???
    // TODO: offset can technically fail original_length is near usize::MAX
    //   would happen if usize == u32 and decoded a 12-hour 96khz file
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
    processed_left.shrink_to(original_length);
    processed_right.shrink_to(original_length);

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
    (
        processed_left.into_boxed_slice(),
        processed_right.into_boxed_slice(),
    )
}

/// Save processed audio to the output using `hound`
fn save_audio(file_path: &Path, audio: &(Box<[f32]>, Box<[f32]>), sample_rate: u32) {
    // TODO: add simple functionality for mono signals?
    // Might be a lot of work for something you can re-render to stereo in foobar2000
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32, // hound only supports 32-bit float
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(file_path, spec).expect("Could not create writer");

    izip!(audio.0.iter(), audio.1.iter()).for_each(|(left, right)| {
        writer.write_sample(*left).expect("Could not write sample");
        writer.write_sample(*right).expect("Could not write sample");
    });
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
        let modified_audio = process_samples(channels, sample_rate);

        print!("    Saving...");
        io::stdout().flush()?;
        output_path.set_extension("wav");
        fs::create_dir_all(output_path.parent().unwrap())?;
        save_audio(&output_path, &modified_audio, sample_rate);

        println!("    T+{:#?} ", time.elapsed());
    }
    Ok(())
}
