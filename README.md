# Some set of DSP effects
Pet project of processing audio files by and for NekoAlosma to learn FFT processing

Currently, this program takes in stereo audio files (input folder created on first run) and:
* Aligns the phase angle between the left and right channel
  * Concept based on Thimeo Stereo Tool's "Image phase amplifier: 0%", automated through Thimeo WatchCat
  * Use case: switching between a mono speaker to a car stereo
    * Prevents per-frequency phase cancellation for a better downmix to mono
    * Heavily reduces the perceived stereo width, but instrument placement / channel-specific sounds are preserved
* Low-cuts audio 10Hz and below / High-passes audio 10Hz and above (Enabled by default, feature flag `subsonic_removal`)
  * Not based on anything in particular
  * Use case: Remove some subsonic noise (<20Hz audio) that is not worth capturing in music.
    * Very few stereo systems and headphones allow audio <20Hz audio playback, which would be vibrations felt rather than a tone heard.
    * Will obviously change the shape of the waveform, but the actual heard content should be the same
* Rotates the phase of the result from the above step (Enabled by default, feature flag `final_rotation`)
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
    * .mp3 files: does not support invalid CRC checksums, so output files will have added silence or are cut prematurely based on padding
* Lofty is used to remaps tags to .wav's ID3v2 and RIFF INFO tags, so the conversion is usually lossy; non-standard tags like LYRICS and UNSYNCEDLYRICS/UNSYNCED_LYRICS/'UNSYNCED LYRICS' are likely not copied over.
  * Since this project exports files as .wav, you can try converting input files to 32-bit .wav while keeping tags using another program.
* FFT does not exactly preserve the shape of waveforms below 20hz
  * Side effect: FFT produces relatively minor frequency smearing / pre-echo depending on chosen frequency
    * Mainly affects very short hi-hats and sounds delayed in one channel
  * Stereo Tool suggests that it uses ~11hz, but no frequency smearing is detected?

### Things to do:
* Add option and confirmation to delete input files after processing
* Make all steps optional through feature flags or command-line options
* Improve program efficiency
  * Approximate performance on my workstation:
    * ~4.31 minutes of runtime per 1 hour of 44.1khz audio
  * Possible slowdown due to CPU affinity (`rayon` does not implement CPU pinning or similar) or other applications
  * (Windows only) Set the program's priority class (Idle -> Above Normal) and I/O priority (Normal -> High)
    * Approximate 50% speedup (90s to 60s on an old test suite) using System Informer to apply priorities
  * Add shortcut for mono files (remove DC noise only)
* Add more error-checking
  * Handle all existing `.unwrap()`s and `.expect()`s
  * Vec memory allocation on 32-bit builds for long files of audio
    * Could just suggest cutting down the audio into smaller bits
  * Test files that shorter than FFT (sound effects?)
  * Mono files are converted to stereo files

![performance](<Screenshot 2026-03-31 201506.png>)