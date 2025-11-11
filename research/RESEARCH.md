# Research
Attempts at reverse-engineering Thimeo Stereo Tool's ["Stereo Image"](https://www.thimeo.com/documentation/stereo.html) section.

## Notes
Thimeo Stereo Tool is a real-time audio processor that can come in VST form. As such, it processes audio in blocks, where the highest settings are with 4,096-sample blocks and an abstract 150% audio quality. The "audio quality" setting may refer to internal quality increases such as short FFT size and/or filters, but it could either/also mean the audio is resampled based on the setting. Thimeo WatchCat is a standalone audio file processor that can apply Stereo Tool presets, where my configuration uses a preset with "Image phase amplifier: 0%" and high settings ([Preset file (.sts)](./NekoAlosama%20-%20zero%20phase%20only%20-%202025-08-20.sts)), exports in 32-bit floating-point .wav files with most tags transfered, and deletes the input files.

This project is based around the idea of reimplementing the "Image phase amplifier: 0%" setting with as much quality as possible. Actual reverse-engineering through code decompilation is outside of my skillset. While Stereo Tool does have modules that clearly use short-time Fourier Transform (STFT) processing like the Delossifier with its spectrogram display, I am now considering the idea that this setting could be using another time-frequency analysis technique such as complex wavelet transforms or constant-Q transforms, or could be using multiple all-pass filters with varying coefficients.

I think I may need outside help with this problem. I encourage someone out there to ask Thimeo about how the algorithm works, since I'm unsure why they seem to have the only plugin in the world to do this. I'm not asking them since I'm scared about follow-up questions.

### List of potential improvements:
* __Larger or variable block sizes:__ while 4,096 samples are good for processing on 44.1khz and 48khz audio, it theoretically becomes bad for processing low-frequencies with 96khz or 192khz audio since the minimum frequency that can be depicted will be 23.4hz and 46.8hz, respectively. This project currently uses a minimmum frequency of 10hz over any sample rate.
  * Side note: Stereo Tool might use variable block sizes already with 4,096 samples as the maximum, since certain songs are have shorter transients/spectrasl leakage using Stereo Tool compared to the equivalent setting in `fix_audio`.
* __More accurate/precise processing:__ This project uses 64-bit floating-point numbers in its calculations in order to prevent precision loss that might come with 32-bit floating-point numbers. For example, equalizing the RMS between the left and right channels using 32-bit floating-point numbers results in an error of about 0.02dB.
* __Window function:__ assuming that this Stereo Tool setting uses STFT, the varying output level of its audio implies that the window function being used introduces scalloping loss. A flat-top window would somewhat help with this in exchange with increasing runtime.
* __Better algorithm:__ assuming that this Stereo Tool Setting uses STFT, we could implement a different algorithm which might be better than STFT. If it does not use STFT, the different algorithm could be close to the actual algorithm.
  * Complex wavelet transform: no known implementation of the inverse complex wavelet transform
  * Constant-Q transform: current known implementation uses too much memory and runtime to be viable
  * Multiple all-pass filters: haven't tried yet, currently researching if it's feasable
* __Better transfer of tags:__ WatchCat fails to transfer the tags of audio that contains special Unicode characters or are just too long. Covers are also not transfered at all.
* __Better detection of non-audio files:__ WatchCat, like Symphonia, detects some non-audio files like certain pictures as audio.

### Current issues:
* __Reminder: Stereo Tool runs in real-time and can utilize multiple threads.__ This implies that the solution is simple, at least enough to parallelize.
* __Clicking noise in certain songs:__ SOPHIE's "JUST LIKE WE NEVER SAID GOODBYE" has significant clicking when processed with `fix_audio`.
  * Sample files are marked `11`, `12`, and `13`; reduced to the first 10% of the song (18.852 seconds)
  * Mainly uncorrelated/out-of-phase/varying phase(?) low frequencies.
* __High peak in certain songs:__ DJ Screw's "Backstreets (Screwed)" has a high peak level of 9.26dB with `fix_audio` compared to 3.65dB with WatchCat.
  * Sample files are marked `21`, `22`, and `23`; reduced to the last 30 seconds
  * `fix_audio` result is 9.26dB even with DC offset being applied to reduce peak levels.
  * The peak level does not significantly change when using different window functions (i.e. similar peak level with rectangular window)
  * Implies some sort of fundamental difference between the `fix_audio` algorithm and the Stereo Tool algorithm?