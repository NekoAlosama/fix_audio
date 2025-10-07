use hound::{SampleFormat, WavSpec, WavWriter};
use itertools::izip;
use std::path::Path;

/// Export processed audio to the output using `hound`
pub fn export_audio(file_path: &Path, audio: &(Vec<f32>, Vec<f32>), sample_rate: u32) {
    // TODO: add simple functionality for mono signals?
    // Might be a lot of work for something you can re-render to stereo in foobar2000
    let spec = WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 32, // hound only supports 32-bit float
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(file_path, spec).expect("Could not create writer");

    izip!(audio.0.iter(), audio.1.iter()).for_each(|(left, right)| {
        writer.write_sample(*left).expect("Could not write sample");
        writer.write_sample(*right).expect("Could not write sample");
    });
    writer.finalize().expect("Could not finalize WAV file");
}
