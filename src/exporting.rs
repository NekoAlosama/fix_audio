use core::hint;
use std::{fs::File, io::BufWriter, path::Path};

use hound::{SampleFormat, WavSpec, WavWriter};
use lofty::{
    config::WriteOptions,
    tag::{Tag, TagExt as _},
};

/// Export processed audio to the output using `hound`.
pub fn export_audio(file_path: &Path, audio: (Box<[f32]>, Box<[f32]>), sample_rate: u32) {
    // TODO: add simple functionality for mono signals?
    let spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32, // hound only supports 32-bit float
        sample_format: SampleFormat::Float,
    };
    let mut writer: WavWriter<BufWriter<File>> =
        WavWriter::create(file_path, spec).expect("Could not create writer");

    audio.0.into_iter().zip(audio.1).for_each(|(left, right)| {
        // cold_path() used since we're not expecting an error
        writer.write_sample(left).unwrap_or_else(|_| {
            hint::cold_path();
            panic!("Could not write sample")
        });
        writer.write_sample(right).unwrap_or_else(|_| {
            hint::cold_path();
            panic!("Could not write sample")
        });
    });
    writer.finalize().expect("Could not finalize WAV file");
}

/// Write tags to exported audio.
/// Requires `export_audio()` to be executed first.
/// Unfortunately doubles Exporting time and memory since `hound` clears all tags when calling `.finalize()`.
pub fn write_tags(file_path: &Path, tags: Box<[Tag]>) {
    // We don't particularly care if tags are written
    for tag in tags {
        _ = tag.save_to_path(file_path, WriteOptions::default());
    }
}
