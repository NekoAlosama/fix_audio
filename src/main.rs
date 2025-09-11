// Imports
// Currently prefering to have one or two namespaces used in the functions
use realfft::{RealFftPlanner, num_complex, num_traits::Zero};
use std::{
    fs,
    io::{self, Write},
    path, time,
};
use symphonia::{core, default};

// Hard-coded directories
const INPUT_DIR: &str = "./inputs/";
const OUTPUT_DIR: &str = "./outputs/";

type AudioMatrix = ((Vec<f64>, Vec<f64>), u32);
fn get_samples_and_metadata(path: &path::PathBuf) -> Result<AudioMatrix, core::errors::Error> {
    // TODO: check updated example code for Symphonia dev-0.6.0
    // Numbers are from the Symphonia basic proceedures in its docs.rs

    // 1
    let code_registry = default::get_codecs();
    // 2
    let probe = default::get_probe();

    // 3
    // 4
    let mss = core::io::MediaSourceStream::new(Box::new(fs::File::open(path)?), Default::default());

    // 5
    // 6
    let probe_result = probe.format(
        core::probe::Hint::new().with_extension("flac"),
        mss,
        &Default::default(),
        &Default::default(),
    )?;
    let mut format = probe_result.format;

    // 7
    let track = format.default_track().unwrap();

    // 8
    let mut decoder = code_registry.make(&track.codec_params, &Default::default())?;

    let track_id = track.id;

    let mut left_samples: Vec<f64> = vec![];
    let mut right_samples: Vec<f64> = vec![];

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

                        sample_buf = Some(core::audio::SampleBuffer::<f64>::new(duration, spec));
                    }

                    if let Some(buf) = &mut sample_buf {
                        // Doesn't seem like plannar (first half is left samples, second half is right) is working
                        buf.copy_interleaved_ref(audio_buf);
                        let reservation = buf.samples().len() / 2;
                        left_samples.reserve(reservation);
                        right_samples.reserve(reservation);

                        for sample in buf.samples().chunks_exact(2) {
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
        // 20 hz
        2205 => 2304,
        2400 => 2592,
        // 4800 => 5184 // appears later in 10 hz

        // 15 hz
        2940 => 3072,
        3200 => 3456,
        6400 => 6561,

        // 10 hz
        4410 => 4608,
        4800 => 5184, // also 20 hz for 96khz
        9600 => 10368,

        // 1 hz
        44100 => 46656,
        48000 => 49152,
        96000 => 98304,
        _ => 0,
    }
}

fn window(n: usize, rate: usize) -> f64 {
    // Windowing is used to make the signal fade in and out
    //   to prevent discontinuities, which causes spectral leakage (noise tuned to the music).
    // This function uses the Hann window, which is unoptimized but good enough for most sounds.
    // One benefit is that window(n) + window(n + rate/2) == 1.0, so we get back to the original
    //   level of the input if we overlap two FFTs with this offset between them.

    // If we wanted this to be continuous, do (n % rate) to ensure discontinuities
    // I.e. window(47999, 48000) == window(48000, 48000) == 0.0
    (std::f64::consts::PI * n as f64 / (rate - 1) as f64)
        .sin()
        .powi(2)
}

fn process_samples(
    planner: &mut RealFftPlanner<f64>,
    data: (Vec<f64>, Vec<f64>),
    rate: u32,
) -> (Vec<f64>, Vec<f64>) {
    let mut left_channel = data.0;
    let mut right_channel = data.1;
    // TODO: change name of modified sample rate
    // Force minimum reconstructed frequency to 20 hz
    let sample_rate = rate as usize / 20;

    // It's better to add the offset between channels as silence here
    // For the original signal, silence is added to the end, while the
    //   offset signal's silence is added to the beginning
    let original_length = left_channel.len();
    let offset = sample_rate / 2; // If using 20 hz, 44.1khz gives 1102.5 samples. Probably insignificant?
    let offset_vec = vec![0.0_f64; offset];
    let mut not_left_channel = offset_vec.clone();
    let mut not_right_channel = offset_vec.clone();
    not_left_channel.append(&mut left_channel.clone());
    not_right_channel.append(&mut right_channel.clone());
    left_channel.resize(original_length + offset, 0.0);
    right_channel.resize(original_length + offset, 0.0);
    debug_assert_eq!(left_channel.len(), not_right_channel.len());

    // Best to do a small FFT to prevent transient smearing
    let fft_size = next_fast_fft(sample_rate);
    let recip_fft = (fft_size as f64).recip();
    let r2c = planner.plan_fft_forward(fft_size);
    let c2r = planner.plan_fft_inverse(fft_size);
    let fft_complex_size = r2c.complex_len();

    // The algorithm I want to use will chunk each signal by sample_rate, so it's better to round up
    //   to the next multiple so we can use ChunksExact and have no remainder
    let fft_total = left_channel.len().next_multiple_of(sample_rate);
    left_channel.resize(fft_total, 0.0);
    right_channel.resize(fft_total, 0.0);
    not_left_channel.resize(fft_total, 0.0);
    not_right_channel.resize(fft_total, 0.0);
    debug_assert_eq!(left_channel.len() % sample_rate, 0);

    // Chunks idk
    let window = (0..sample_rate)
        .map(|s| window(s, sample_rate))
        .collect::<Vec<f64>>();

    // Failed attempt at reducing memory footprint
    let left_channel = left_channel.chunks_exact(sample_rate);
    let right_channel = right_channel.chunks_exact(sample_rate);
    let not_left_channel = not_left_channel.chunks_exact(sample_rate);
    let not_right_channel = not_right_channel.chunks_exact(sample_rate);

    // The first bin is the DC bin, which is the average unheard noise
    // From what we've seen, this should be a real number (C + 0.0i), but it's better to be safe
    //   by zeroing it out in both axes
    let new_dc = num_complex::Complex::zero();

    // Saving samples for later
    // TODO: check if Vec::with_capacity(fft_total); is faster than .clone();
    let mut processed_left: Vec<f64> = Vec::with_capacity(fft_total);
    let mut processed_right: Vec<f64> = Vec::with_capacity(fft_total);
    let mut processed_not_left: Vec<f64> = Vec::with_capacity(fft_total);
    let mut processed_not_right: Vec<f64> = Vec::with_capacity(fft_total);

    for chunk in left_channel
        .zip(right_channel)
        .zip(not_left_channel.zip(not_right_channel))
    {
        let mut left = chunk.0.0.to_vec();
        let mut right = chunk.0.1.to_vec();
        let mut not_left = chunk.1.0.to_vec();
        let mut not_right = chunk.1.1.to_vec();

        for index in 0..sample_rate {
            let window_multiplier = window[index];
            left[index] *= window_multiplier;
            right[index] *= window_multiplier;
            not_left[index] *= window_multiplier;
            not_right[index] *= window_multiplier;
        }

        // Zero-pad signal for FFT
        left.resize(fft_size, 0.0);
        right.resize(fft_size, 0.0);
        not_left.resize(fft_size, 0.0);
        not_right.resize(fft_size, 0.0);

        // Create scratch FFT for RealFFT
        // RealFFT uses RustFFT's .process_with_scratch() for its .process() function
        let mut left_fft = r2c.make_output_vec();
        let mut right_fft = r2c.make_output_vec();
        let mut not_left_fft = r2c.make_output_vec();
        let mut not_right_fft = r2c.make_output_vec();

        // Ignore errors by RealFFT
        // RustFFT does not return a Result after processing,
        //   but RealFFT does return Results due to some zero-check
        //   RealFFT author says to just ignore these in the meantime.
        // https://github.com/HEnquist/realfft/issues/41#issuecomment-2050347470
        let _ = r2c.process(&mut left, &mut left_fft);
        let _ = r2c.process(&mut right, &mut right_fft);
        let _ = r2c.process(&mut not_left, &mut not_left_fft);
        let _ = r2c.process(&mut not_right, &mut not_right_fft);

        // Remove local DC offset
        left_fft[0] = new_dc;
        right_fft[0] = new_dc;
        not_left_fft[0] = new_dc;
        not_right_fft[0] = new_dc;

        for index in 1..fft_complex_size {
            let left_r = left_fft[index].norm();
            let right_r = right_fft[index].norm();
            let not_left_r = not_left_fft[index].norm();
            let not_right_r = not_right_fft[index].norm();

            // TODO: doing atan2 is expensive. is there something better?
            // Align the phase of the left and right channels using the circular mean
            // Usually, the side with the greater magnitude to not rotate much while the lesser one rotates more
            // The regular mean would make both sides rotate equally, which seems to cause issues among FFTs
            let phase = (left_fft[index].im + right_fft[index].im)
                .atan2(left_fft[index].re + right_fft[index].re);
            let not_phase = (not_left_fft[index].im + not_right_fft[index].im)
                .atan2(not_left_fft[index].re + not_right_fft[index].re);

            left_fft[index] = num_complex::Complex::from_polar(left_r, phase);
            right_fft[index] = num_complex::Complex::from_polar(right_r, phase);
            not_left_fft[index] = num_complex::Complex::from_polar(not_left_r, not_phase);
            not_right_fft[index] = num_complex::Complex::from_polar(not_right_r, not_phase);
        }

        let _ = c2r.process(&mut left_fft, &mut left);
        let _ = c2r.process(&mut right_fft, &mut right);
        let _ = c2r.process(&mut not_left_fft, &mut not_left);
        let _ = c2r.process(&mut not_right_fft, &mut not_right);

        drop(left_fft);
        drop(right_fft);
        drop(not_left_fft);
        drop(not_right_fft);

        // FFT zero-padding is ignored with this `for`` loop
        for index in 0..sample_rate {
            // Normalize FFT values
            let new_left = left[index] * recip_fft;
            let new_right = right[index] * recip_fft;
            let new_not_left = not_left[index] * recip_fft;
            let new_not_right = not_right[index] * recip_fft;

            // FFT processing has finished
            processed_left.push(new_left);
            processed_right.push(new_right);
            processed_not_left.push(new_not_left);
            processed_not_right.push(new_not_right);
        }
    }
    drop(r2c);
    drop(c2r);

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

    // Remove overall DC after all local DC was removed
    // DC is just the average of the whole signal
    let left_dc = processed_left.clone().iter().sum::<f64>() / original_length as f64;
    let right_dc = processed_right.clone().iter().sum::<f64>() / original_length as f64;
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
        .fold(0.0, |acc, s| acc + s * s);
    let right_s = processed_right
        .clone()
        .iter()
        .fold(0.0, |acc, s| acc + s * s);
    // First square root is to get the multiplier when applied to s^2,
    //   second square root is to get the multiplier when applied to just s.
    let equalizer = (left_s / right_s).sqrt().sqrt();

    for index in 0..original_length {
        processed_left[index] /= equalizer;
        processed_right[index] *= equalizer;
    }

    // Overall processing is done
    (processed_left, processed_right)
}

fn save_audio(file_path: &std::path::Path, audio: (Vec<f64>, Vec<f64>), rate: u32) {
    // TODO: add simple functionality for mono signals?
    // Might be a lot of work for something you can re-render to stereo in foobar2000
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: rate,
        bits_per_sample: 32, // hound only supports 32-bit float
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(file_path, spec).expect("Could not create writer");
    for index in 0..audio.0.len() {
        writer
            .write_sample(audio.0[index] as f32)
            .expect("Could not write sample");
        writer
            .write_sample(audio.1[index] as f32)
            .expect("Could not write sample");
    }
    writer.finalize().expect("Could not finalize WAV file");
}

fn get_paths(directory: path::PathBuf) -> io::Result<Vec<path::PathBuf>> {
    // Recursive file path retriever
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

    println!("File setup time: {:#?}", time.elapsed());

    let mut real_planner = RealFftPlanner::<f64>::new();
    for entry in entries {
        println!("Found file: {}", entry.display());
        let mut output_path = path::PathBuf::new();
        let unprefix_output = entry.strip_prefix(INPUT_DIR).unwrap();
        output_path.push(OUTPUT_DIR);
        output_path.push(unprefix_output);

        let channels: (Vec<f64>, Vec<f64>);
        let sample_rate: u32;
        print!("    Decoding...");
        io::stdout().flush()?; // Show print instantly
        // If we can't properly decode the data, it's likely not audio data
        // In this case, we just send it to OUTPUT_DIR so it's not spitting the warning every run
        match get_samples_and_metadata(&entry) {
            Ok(data) => {
                channels = data.0;
                sample_rate = data.1;
            }
            Err(core::errors::Error::IoError(err)) => {
                if err.kind() == io::ErrorKind::UnexpectedEof {
                    println!("  Invalid audio, sent to output.");
                    fs::create_dir_all(output_path.parent().unwrap())?;
                    fs::rename(entry, output_path)?;
                    continue;
                } else {
                    return Err(core::errors::Error::IoError(err));
                }
            }
            // Usually happens if Symphonia attempts to open a .jpg or .png
            Err(core::errors::Error::Unsupported(_) | core::errors::Error::DecodeError(_)) => {
                println!("    Invalid audio, sent to output");
                fs::create_dir_all(output_path.parent().unwrap())?;
                fs::rename(entry, output_path)?;
                continue;
            }
            Err(other) => return Err(other),
        };

        print!("    Processing... ");
        io::stdout().flush()?;
        let modified_audio = process_samples(&mut real_planner, channels, sample_rate);

        print!("    Saving...");
        io::stdout().flush()?;
        output_path.set_extension("wav");
        fs::create_dir_all(output_path.parent().unwrap())?;
        save_audio(&output_path, modified_audio, sample_rate);

        println!("    T+{:#?} ", time.elapsed());
    }
    Ok(())
}
