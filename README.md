# Some set of DSP effects
Pet project of processing audio files by and for NekoAlosma to learn FFT processing

Currently, this program takes in stereo audio files (input folder created on first run) and:
* Aligns the phase angle between the left and right channel
  * Concept based on Thimeo Stereo Tool's "Image phase amplifier: 0%", automated through Thimeo WatchCat
    * [Researching how to replicate its results](./research/)
  * Use case: switching between a mono speaker to a car stereo
    * Prevents per-frequency phase cancellation for a better downmix to mono
    * Heavily reduces the perceived stereo width, but instrument placement / channel-specific sounds are preserved
* Rotates the phase of the result from the above step
  * Concept based on iZotope RX 11's "Phase" module, can't be automated
  * Use case: Reduce signal peak levels, especially ones amplified due to the above step
    * RX 11's algorithm usually increases peak levels for no good reason
    * May be removed if the alignment algorithm is changed to one that inherently produces lower signal levels, making this step redundant
* Averages the loudness of the left and right channel
  * Concept based on iZotope RX 11's "Azimuth" module, can't be automated
  * Use case: ensure that one channel doesn't overpower the other over the course of a track
    * Uses the EBU R 128 Integrated Loudness, while RX 11 uses plain RMS
    * Plain RMS is affected by DC bias and does not account for human hearing

Processed audio files are sent to the output folder as 32-bit floating-point .wav files with tags and embedded covers transfered over. Non-audio files (covers, documents, etc.) are transfered to the output folder. The original audio files are kept in the input folder, so remember to delete them if you don't need to re-run the program with changes.

## Reflection
### Known problems I can't seem to fix:
* Symphonia dev-0.6 doesn't support certain codecs and features
  * Try converting unsupported music files to 32-bit .wav
    * Video files with an audio track
    * .opus files
    * .mp3 files: does not support invalid CRC checksums, so output files will have added silence
* FFT alignment algorithm produces issues
  * [Research is ongoing](./research/)
* FFT does not exactly preserve the shape of waveforms below 20hz
  * Side effect: FFT produces relatively minor frequency smearing / pre-echo depending on chosen frequency
    * Mainly affects very short hi-hats and sounds delayed in one channel
  * Stereo Tool suggests that it uses ~11hz, but no frequency smearing is detected?

### Things to do:
* Add option and confirmation to delete input files after processing
* Improve program efficiency
  * Approximate performance:
    * about 2.41 minutes of runtime per 1 hour of 44.1khz audio (5.15 minutes to process 2.14 hours of music (339 million samples))
  * Reduce memory usage
    * Seems like there's hidden clones/duplicates? Unsure if it's just bad logic on my part or my dependencies. Clippy isn't saying much in this regard.
    * Need a memory profiler
  * Possible slowdown due to CPU affinity (`rayon` does not implement CPU pinning or similar) or other applications
  * `mimalloc` being used as an alternative allocator. Minor overallocation and may give better performance on other platforms
  * (Windows only) Set the program's priority class (Idle -> Above Normal) and I/O priority (Normal -> High)
    * Approximate 50% speedup (90s to 60s on an old test suite) using System Informer to apply priorities
  * Add shortcut for mono files (remove DC noise only)
* Add more error-checking
  * Handle all existing `.unwrap()`s and `.expect()`s
  * Vec memory allocation on 32-bit builds for long files of audio
    * Could just suggest cutting down the audio into smaller bits
  * Test files that shorter than FFT (sound effects?)
  * Mono files are converted to stereo files
* Make functions generic over floats (allow `f32` or `f64` in case more precision is needed)
  * Considering always using `f64` over `f32` to ensure no quantization noise/imprecision

![performance](./Screenshot%202025-12-21%20205638.png)