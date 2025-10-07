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
type AudioMatrix = ((Vec<f32>, Vec<f32>), u32);
/// Get samples and metadata for a given file using `Symphonia`
pub fn get_samples_and_metadata(path: &path::PathBuf) -> Result<AudioMatrix, Error> {
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

    left_samples.shrink_to_fit();
    right_samples.shrink_to_fit();
    // TODO: return error if fft_total would be larger than usize::MAX
    Ok(((left_samples, right_samples), sample_rate))
}
