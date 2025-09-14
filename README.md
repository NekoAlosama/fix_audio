# Some set of DSP effects
Pet project of processing audio files by and for NekoAlosma to learn FFT processing

Currently, this program takes in stereo audio files (input folder created on first run) and:
* Aligns phase information between the left and right channel
  * Idea based on Thimeo Stereo Tool's "Image phase amplifier: 0%", automated through Thimeo WatchCat
  * (Heavily) reduces perceived stereo width in exchange for a better mono downmix
  * I listen to music both on one earbud and in my car stereo, so this prevents phase cancellation and keeps side/stereo information
* Removes DC bias
  * Idea based on iZotope RX 11's "Filter DC Offset" option in the "De-hum" module, automated with RX 11's "Batch Processor"
  * Removes inaudible 0.0 hz noise
  * The noise artifically modifies loudness statistics and is carried through further processing
* Averages the RMS of the left and right channel
  * Idea based on iZotope RX 11's "Azimuth" module, can't be automated
  * Generally ensures that one channel doesn't overpower the other over the course of a track
  * In RX 11, the module's "Suggest" button only modifies the right channel

Processed audio files are sent to the output folder as 32-bit floating-point .wav files. Non-audio files (covers, documents, etc.) are transfered to the output folder. The original audio files are kept in the input folder, so remember to delete them if you don't need to re-run the program with changes.

## Reflection
__Known problems I can't fix__:
* Symphonia 0.5.4 doesn't support files above 96khz
  * Fixed in dev-0.6 branch, but it's unstable.
* Cannot copy tags from input to output (lack of ecosystem support?)
  * Symphonia only supports tag reading
  * hound does not support writing .wav Vorbis tags
* FFT introduces relatively minor transient smearing / pre-echo
  * Problem tracks:
    * SOPHIE - "MSMSMSM": Transient hi-hat of <0.1s, right channel delayed by ~0.02s


__Things to do__:
* Add support for mono files
  * Force upmixing to stereo?
  * Make sure to bypass phase alignment
* Add support for videos with an audio track (.webm, .mkv)
  * Symphonia doesn't have a demuxer example/tutorial?
* Make code more idiomatic
* Increase program efficiency
  * The current memory usage is good, so the main feature to implement is multithreading
  * Main bottlenecks also seem to be Sympohonia decoding (I/O reading) and hound .wav file-saving (I/O writing)
  * Another improvement would be to set the program's priority class (Idle -> Above Normal) and I/O priority (Normal -> High)
    * Approximate 50% speedup (90s to 60s on test suite) using System Informer to apply priorities
* and more...


![flamegraph](flamegraph.svg)