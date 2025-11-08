use std::path::Path;

use hound::{SampleFormat, WavSpec, WavWriter};
use itertools::izip;
use lofty::{
    config::WriteOptions,
    tag::{ItemKey, Tag, TagExt as _, TagType},
};

/// Export processed audio to the output using `hound`
pub fn export_audio(file_path: &Path, audio: (Box<[f64]>, Box<[f64]>), sample_rate: u32) {
    // TODO: add simple functionality for mono signals?
    let spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32, // hound only supports 32-bit float
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(file_path, spec).expect("Could not create writer");

    izip!(audio.0.into_iter(), audio.1.into_iter()).for_each(|(left, right)| {
        writer
            .write_sample(left as f32)
            .expect("Could not write sample");
        writer
            .write_sample(right as f32)
            .expect("Could not write sample");
    });
    writer.finalize().expect("Could not finalize WAV file");
}

/// Write tags to exported audio.
/// Requires `export_audio()` to be executed first.
pub fn write_tags(file_path: &Path, maybe_tags: Option<Tag>) {
    // We don't particularly care if tags are written
    if let Some(mut tags) = maybe_tags {
        // For compatibility, all tags are written to WAV as Id3v2.4
        tags.re_map(TagType::Id3v2);

        // File peak will change due to processing
        tags.remove_key(&ItemKey::ReplayGainAlbumPeak);
        tags.remove_key(&ItemKey::ReplayGainTrackPeak);

        // Added due to `.re_map()`
        tags.remove_key(&ItemKey::EncoderSoftware); // Associates with the "ENCODING SETTINGS" tag lol
        tags.remove_key(&ItemKey::EncoderSettings);

        _ = tags.save_to_path(file_path, WriteOptions::default());
    }
}
