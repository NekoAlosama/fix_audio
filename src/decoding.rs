use std::{fs, path};
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

/// Seperated here due to Clippy lint
type AudioMatrix = ((Vec<f64>, Vec<f64>), u32);
/// Get samples and metadata for a given file using `Symphonia`
pub fn get_samples_and_metadata(path: &path::PathBuf) -> Result<AudioMatrix, Error> {
    // Based on `Symphonia`'s docs.rs page and example code (mix of 0.5.4 and dev-0.6)
    // Numbers are from the `Symphonia` basic proceedures in its docs.rs

    // Early exit if file doesn't have an extension indicating audio, but could still be read by Symphonia
    // .png files could have .jpg data if converted from that
    // .zip files with no compression would have Symphonia decode the first track it sees
    if let Some(extension) = path.extension()
        && let Some("zip" | "png") = extension.to_str()
    {
        return Err(Error::Unsupported("Not an audio file."));
    }
    // 1
    let codec_registry = default::get_codecs();
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
    let mut decoder = codec_registry.make_audio_decoder(
        // Don't know why the two unwraps are needed
        track.codec_params.as_ref().unwrap().audio().unwrap(),
        &AudioDecoderOptions::default(),
    )?;

    let track_id = track.id;

    let mut left_samples: Vec<f64> = vec![];
    let mut right_samples: Vec<f64> = vec![];

    let mut sample_buf: Vec<Vec<f64>> = vec![];
    let mut sample_rate = 0;
    let mut channel_count = 0;

    // 9
    // 10
    // 11
    while let Ok(Some(packet)) = format.next_packet() {
        if packet.track_id() == track_id {
            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    if sample_rate == 0 {
                        sample_rate = audio_buf.spec().rate();
                        channel_count = audio_buf.num_planes();

                        // .copy_to_vecs_planar() requires a Vec containing the channels
                        match channel_count {
                            1 => {
                                sample_buf = vec![vec![0_f64; audio_buf.samples_planar()]];
                            }
                            2 => {
                                sample_buf = vec![
                                    vec![0_f64; audio_buf.samples_planar()],
                                    vec![0_f64; audio_buf.samples_planar()],
                                ];
                            }
                            _ => {
                                return Err(Error::Unsupported("Too many channels"));
                            }
                        }
                    }

                    audio_buf.copy_to_vecs_planar(&mut sample_buf);
                    match channel_count {
                        1 => {
                            // SAFETY: sample_buf has one channel
                            left_samples.extend(unsafe { sample_buf.get_unchecked(0) });
                        }
                        2 => {
                            // SAFETY: sample_buf has two channels
                            left_samples.extend(unsafe { sample_buf.get_unchecked(0) });
                            // SAFETY: sample_buf has two channels
                            right_samples.extend(unsafe { sample_buf.get_unchecked(1) });
                        }
                        _ => {
                            unreachable!();
                        }
                    }
                }
                // For some reason, `Symphonia` is fine if the decode doesn't work?
                // like with malformed data or something
                Err(Error::DecodeError(_)) => (),
                Err(_) => break,
            }
        }
    }

    if channel_count == 1 {
        right_samples.copy_from_slice(&left_samples);
    }

    left_samples.shrink_to_fit();
    right_samples.shrink_to_fit();
    // TODO: return error if fft_total would be larger than usize::MAX
    Ok(((left_samples, right_samples), sample_rate))
}
