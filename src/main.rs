// Imports
// Currently prefering to have one or two namespaces used in the functions
use realfft::{RealFftPlanner, num_complex};
use std::{fs, io::{self, Write}, path, time};
use symphonia::{core, default};

// Hard-coded files
const INPUT_DIR: &str = "./inputs/";
const OUTPUT_DIR: &str = "./outputs/";

type AudioMatrix = ((Vec<f32>, Vec<f32>), core::audio::SignalSpec);
// TODO: add error handling if file isn't audio
// Currently handling io errors as
fn get_samples_and_metadata(path: &path::PathBuf) -> Result<AudioMatrix, core::errors::Error> {
    // TODO: check updated example code for Symphonia dev-0.6.0

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
    // left_track.codec_params == right_track.codec_params, but are separate here for symmetry
    let mut decoder = code_registry.make(&track.codec_params, &Default::default())?;

    let track_id = track.id;

    let mut left_samples: Vec<f32> = vec![];
    let mut right_samples: Vec<f32> = vec![];

    let mut sample_buf = None;
    let mut meta_spec = core::audio::SignalSpec::new(0, core::audio::Channels::FRONT_LEFT);

    // 9
    // 10
    // 11
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() == track_id {
            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    if sample_buf.is_none() {
                        let spec = *audio_buf.spec();
                        meta_spec = spec;

                        let duration = audio_buf.capacity() as u64;

                        sample_buf = Some(core::audio::SampleBuffer::<f32>::new(duration, spec));
                    }

                    if let Some(buf) = &mut sample_buf {
                        // Might be better to
                        buf.copy_interleaved_ref(audio_buf);

                        for sample in buf.samples().chunks_exact(2) {
                            left_samples.push(sample[0]);
                            right_samples.push(sample[1]);
                        }
                    }
                }
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
        40000 => 41472,   // 2**9 * 3**4
        44100 => 46656,   // 2**6 * 3**6
        48000 => 49152,   // 2**14 * 3**1
        96000 => 98304,   // 2**15 * 3**1
        192000 => 196608, // 2**16 * 3**1
        _ => 0,
    }
}

fn window(n: usize, rate: usize) -> f32 {
    // Windowing is used to make the signal fade in and out
    //   to prevent discontinuities, which causes spectral leakage (noise tuned to the music).
    // This function uses the Hann window, which is unoptimized but good enough for most sounds.
    // One benefit is that window(n) + window(n + rate/2) == 1.0, so it's good to overlap
    //   two FFTs in this form to return to the loudness of the original signal. However,
    //   we're just doing window(n) + (1.0 - window(n)), which seems to have the same effect?

    // If we wanted this to be continuous, do (n % rate) to ensure discontinuities
    // I.e. window(47999, 48000) == window(48000, 48000) == 0.0
    (std::f32::consts::PI * n as f32 / (rate - 1) as f32)
        .sin()
        .powi(2)
}

fn process_samples(
    data: &mut ((Vec<f32>, Vec<f32>), core::audio::SignalSpec),
    planner: &mut RealFftPlanner<f32>,
) -> (Vec<f32>, Vec<f32>) {
    let left_channel = &mut data.0.0;
    let right_channel = &mut data.0.1;
    let sample_rate = data.1.rate as usize;

    let original_length = left_channel.len();
    let offset = sample_rate / 2;
    let offset_vec = vec![0.0_f32; offset];
    left_channel.append(&mut offset_vec.clone());
    right_channel.append(&mut offset_vec.clone());
    let mut not_left_channel = offset_vec.clone();
    let mut not_right_channel = offset_vec.clone();
    not_left_channel.append(&mut left_channel.clone());
    not_right_channel.append(&mut right_channel.clone());

    // Best FFT size is probably just the sample rate since it'll encompass all frequencies up to Nyquist (size/2 as hz)
    // Any larger will give us garbage data in the ultrasonics (>20,000 hz) and worse timing resolution in exchange for better subsonic resolution (<20 hz)
    // Might be faster to do size == 40000, so Nyquist is exactly 20,000 hz?
    // TODO: check how much FFT reduces noise above Nyquist, since we know that size == 4096 is still good for some purposes
    let fft_size = next_fast_fft(sample_rate);
    let recip_fft = (fft_size as f32).recip();
    let r2c = planner.plan_fft_forward(fft_size);
    let c2r = planner.plan_fft_inverse(fft_size);

    // Since we're using one FFT per second, we add silence up to the next second
    // We can then remove this silence later to get the original length
    let silence_total = left_channel.len().next_multiple_of(sample_rate) - left_channel.len();
    let silence = vec![0.0_f32; silence_total];
    left_channel.append(&mut silence.clone());
    right_channel.append(&mut silence.clone());
    not_left_channel.append(&mut silence.clone());
    not_right_channel.append(&mut silence.clone());

    // The FFTs are larger than one second for performance reasons, so we add silence to each second
    // This slightly improves frequency resolution, but does not add any additional info (i.e. more points of the same shape)
    let zero_padding_total = fft_size - sample_rate;
    let zero_pad = vec![0.0_f32; zero_padding_total];

    let window = (0..sample_rate).map(|s| window(s, sample_rate));
    let left_chunks = left_channel.chunks_exact(sample_rate);
    let right_chunks = right_channel.chunks_exact(sample_rate);
    let not_left_chunks = not_left_channel.chunks_exact(sample_rate);
    let not_right_chunks = not_right_channel.chunks_exact(sample_rate);

    // The first bin is the DC bin, which is the average unheard noise
    // From what we've seen, this should be a real number (C + 0.0i), but it's better to be safe
    //   by zeroing it out in both axes
    let new_dc = num_complex::Complex::new(0.0_f32, 0.0_f32);

    let mut processed_left: Vec<f32> = vec![];
    let mut processed_right: Vec<f32> = vec![];
    let mut processed_not_left: Vec<f32> = vec![];
    let mut processed_not_right: Vec<f32> = vec![];

    for chunk in left_chunks
        .zip(right_chunks)
        .zip(not_left_chunks.zip(not_right_chunks))
    {
        let left_chunk = chunk.0.0.iter().zip(window.clone());
        let right_chunk = chunk.0.1.iter().zip(window.clone());
        let not_left_chunk = chunk.1.0.iter().zip(window.clone());
        let not_right_chunk = chunk.1.1.iter().zip(window.clone());

        let mut left: Vec<f32> = left_chunk.clone().map(|(s, w)| s * w).collect();
        let mut right: Vec<f32> = right_chunk.clone().map(|(s, w)| s * w).collect();
        let mut not_left: Vec<f32> = not_left_chunk.clone().map(|(s, w)| s * w).collect();
        let mut not_right: Vec<f32> = not_right_chunk.clone().map(|(s, w)| s * w).collect();

        // Zero-pad signal for FFT
        left.append(&mut zero_pad.clone());
        right.append(&mut zero_pad.clone());
        not_left.append(&mut zero_pad.clone());
        not_right.append(&mut zero_pad.clone());

        // Create scratch FFT for RealFFT
        // RustFFT uses .process_with_scratch() to get the same functionality
        let mut left_fft = r2c.make_output_vec();
        let mut right_fft = left_fft.clone();
        let mut not_left_fft = left_fft.clone();
        let mut not_right_fft = left_fft.clone();

        // Ignore errors by RealFFT
        // RustFFT does not return a Result after processing,
        //   but RealFFT does return Results from numbers close to zero.
        //   RealFFT author says to just ignore these in the meantime.
        let _ = r2c.process(&mut left, &mut left_fft);
        let _ = r2c.process(&mut right, &mut right_fft);
        let _ = r2c.process(&mut not_left, &mut not_left_fft);
        let _ = r2c.process(&mut not_right, &mut not_right_fft);

        // Remove local DC offset
        left_fft[0] = new_dc;
        right_fft[0] = new_dc;
        not_left_fft[0] = new_dc;
        not_right_fft[0] = new_dc;

        for index in 1..left_fft.len() {
            let left_r = left_fft[index].norm();
            let right_r = right_fft[index].norm();
            let not_left_r = not_left_fft[index].norm();
            let not_right_r = not_right_fft[index].norm();

            // Align the phase of the left and right channels using the circular mean
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

        // Add per-channel FFTs and normalize values after forward and backward FFT
        // Zero-padding ignored with this for-loop, since the actual size would be fft_size
        for index in 0..sample_rate {
            let new_left = left[index] * recip_fft;
            let new_right = right[index] * recip_fft;
            let new_not_left = not_left[index] * recip_fft;
            let new_not_right = not_right[index] * recip_fft;

            processed_left.push(new_left);
            processed_right.push(new_right);
            processed_not_left.push(new_not_left);
            processed_not_right.push(new_not_right);
        }
    }

    // Remove silence by using samples up to the original length of the signal
    for index in 0..original_length {
        processed_left[index] += processed_not_left[index + offset];
        processed_right[index] += processed_not_right[index + offset];
    }

    processed_left = processed_left[0..original_length].to_vec();
    processed_right = processed_right[0..original_length].to_vec();

    assert_eq!(processed_left.len(), original_length);

    // Remove overall DC after all local DC was removed
    // DC is just the average of the whole signal
    let left_dc = processed_left.clone().iter().sum::<f32>() / original_length as f32;
    let right_dc = processed_right.clone().iter().sum::<f32>() / original_length as f32;
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

    (processed_left, processed_right)
}

fn save_audio(
    audio: (Vec<f32>, Vec<f32>),
    file_path: &std::path::Path,
    metadata: &core::audio::SignalSpec,
) {
    // TODO: add simple functionality for mono signals?
    // Might be a lot of work for something you can re-render to stereo in foobar2000
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: metadata.rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(file_path, spec).expect("Could not create writer");
    for index in 0..audio.0.len() {
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

// Possible io::Error from checking if INPUT_DIR and OUTPUT_DIR folders exist
fn main() -> Result<(), core::errors::Error> {
    // Keeping the time for benchmarking
    let time = time::Instant::now();

    match fs::exists(INPUT_DIR) {
        Ok(true) => {}
        Ok(false) => {
            let _ = fs::create_dir(INPUT_DIR);
            println!("Notice: Inputs folder created. Copy audio files here to process them.");
            return Ok(());
        }
        Err(err) => return Err(core::errors::Error::IoError(err)),
    }

    // Get list of files in INPUT_DIR
    // Currently
    let entries: Vec<path::PathBuf> = get_paths(INPUT_DIR.into())?;

    println!("File setup time: {:#?}", time.elapsed());

    let mut real_planner = RealFftPlanner::<f32>::new();
    for entry in entries {
        println!("Found file: {}", entry.display());

        print!("    Decoding...");
        io::stdout().flush()?;
        let mut output_path = path::PathBuf::new();
        let unprefix_output = entry.strip_prefix(INPUT_DIR).unwrap();
        output_path.push(OUTPUT_DIR);
        output_path.push(unprefix_output);

        let mut collected_data = match get_samples_and_metadata(&entry) {
            Ok(data) => data,
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
        let modified_audio = process_samples(&mut collected_data, &mut real_planner);

        output_path.set_extension("wav");
        print!("    Saving...");
        io::stdout().flush()?;
        fs::create_dir_all(output_path.parent().unwrap())?;
        save_audio(modified_audio, &output_path, &collected_data.1);

        println!("    T+{:#?} ", time.elapsed());
    }
    Ok(())
}
