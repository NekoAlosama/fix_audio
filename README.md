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

Non-audio files (covers, documents, etc.) are transfered to the output folder. The original audio files are kept in the input folder, so remember to delete them if you don't need to re-run the program with changes.

## Issues
Known problems which I kinda plan on fixing:
* Does not support mono files
  * Force upmixing to stereo?
  * Make sure to bypass phase alignment
* Cannot copy tags from input to output (lack of ecosystem support?)
  * Symphonia only supports tag reading
  * hound does not support writing .wav Vorbis tags
* FFT introduces relatively minor transient smearing / pre-echo
  * Problem tracks:
    * SOPHIE - "MSMSMSM": Transient hi-hat of <0.1s, right channel delayed by ~0.02s
  * Can't really fix?
* Make code more idiomatic 
* Make code faster
  * Benchmarks on my laptop:
    * Ground-Zero - "Consume Red" (421MB 16-bit .flac, then 1.12GB 32-bit-floating-point .wav, done twice):
      * both files decode to and are saved using 2.26GB of memory (likely because of 32-bit-floating -> 64-bit-floating)
      * .flac took ~27.5s and used 13.02GB while processing
      * .wav took ~32.5s and used 11.27GB while processing
      * ideal memory while processing should be ~8.96GB? (64-bit data + not_data + processed_data + processed_not_data + fft setup)
        * .wav has a fifth copy somewhere, .flac almost has a sixth copy somewhere?
  * Focus on memory bottleneck first, then multithread. If I can't multithread, next spot would be to optimize atan2.
* and more...

![flamegraph](flamegraph.svg)