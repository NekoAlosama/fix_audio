# Some set of DSP effects
Pet project of processing audio files by and for NekoAlosma to learn FFT processing

Currently, this program takes in stereo audio files (input folder created on first run) and:
* Aligns phase information between the left and right channel
  * Concept based on Thimeo Stereo Tool's "Image phase amplifier: 0%", automated through Thimeo WatchCat
  * Use case: switching between a mono speaker to a car stereo
    * Prevents per-frequency phase cancellation for a better downmix to mono
    * Heavily reduces the perceived stereo width, but instrument placement / channel-specific sounds are preserved
* Averages the loudness of the left and right channel
  * Concept based on iZotope RX 11's "Azimuth" module, can't be automated
  * Use case: ensure that one channel doesn't overpower the other over the course of a track
    * Uses the EBU R 128 Integrated Loudness, while RX 11 uses plain RMS
    * Plain RMS is affected by DC bias
* (Unimplemented: Removes DC bias / inaudible 0.0 hz noise)
  * Current known implementation causes large clicks/jumps/artifacts
  * Concept based on iZotope RX 11's "Filter DC Offset" option in its "De-hum" module, automated with RX 11's "Batch Processor"
  * Use case: noise may cause audible noise/clicks/artifacts/imbalances with further processing
    * Loudness stats like RMS and peak level will be incorrect because of DC noise
    * Generally caused by amateur recording, mixing, and mastering (i.e. YAYAYI - track 13 from self-titled)
* Add DC noise to reduce peak levels
  * Currently being used to test where and when DC noise is noticable
  * Use case: Reduce peak levels while keeping the same loudness

Processed audio files are sent to the output folder as 32-bit floating-point .wav files. Non-audio files (covers, documents, etc.) are transfered to the output folder. The original audio files are kept in the input folder, so remember to delete them if you don't need to re-run the program with changes.

## Reflection
__Known problems I can't seem to fix__:
* Symphonia dev-0.6 doesn't support certain features
  * Try converting music files to .wav
    * .opus files or video files containing audio in general
    * .mp3 file output is longer than it should
* FFT alignment algorithm introduces clicks in certian audio
  * Likely cause is with the use of a short-time Fourier transform (STFT)
    * Experiments with lots of overlapped DTFTs don't have clicks, but have noticable spectral noise that can't be removed without lots of processing
      * Might be mitigated through better windows, but I don't know how to implement those (Dolph-Chebyshev, DPSS/Kaiser)
  * Phase changes are not preserved between chunks, so large differences in phase could cause jumps?
    * Average the phase of each consecutive FFT?
  * Also seems to be affected by windowing and overlap-adding
    * Seems that Hann and Blackman windows aren't good enough?
  * Example album with introduced clicks: SOPHIE - "PRODUCT" (2025 Reissue)
* FFT introduces relatively minor transient smearing / pre-echo
  * Mainly affects very short hi-hats and sounds delayed in one channel

__Things to do__:
* Copy tags from input to output
  * Best library seems to be `lofty-rs`
* Add support for mono files
  * Force upmixing to stereo?
  * Make sure to bypass phase alignment
* Make code more idiomatic
* Increase program efficiency
  * The current memory usage is good, so the main feature to implement is multithreading
  * Main bottlenecks also seem to be Sympohonia decoding (I/O reading) and hound .wav file-saving (I/O writing)
    * Parallelism via `rayon` doesn't seem to improve times
  * Another improvement would be to set the program's priority class (Idle -> Above Normal) and I/O priority (Normal -> High)
    * Approximate 50% speedup (90s to 60s on an old test suite) using System Informer to apply priorities
* Add more error-checking
  * e.g. Vec memory allocation on 32-bit targets for long files (12-hours of audio)
    * could just suggest cutting down the audio into smaller bits

![flamegraph](flamegraph.svg)