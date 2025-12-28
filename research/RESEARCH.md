# Research
Attempts at reverse-engineering Thimeo Stereo Tool's ["Stereo Image"](https://www.thimeo.com/documentation/stereo.html) section.

## Notes
Thimeo Stereo Tool is a real-time audio processor that can come in VST form. As such, it processes audio in blocks, where the highest settings are with 4,096-sample blocks and an abstract 150% audio quality. The "audio quality" setting may refer to internal quality increases such as short FFT size and/or filters, but it could either/also mean the audio is resampled based on the setting. Thimeo WatchCat is a standalone audio file processor that can apply Stereo Tool presets, where my configuration uses a preset with `Image phase amplifier: 0%` and high settings ([Preset file (.sts)](./NekoAlosama%20-%20zero%20phase%20only%20-%202025-08-20.sts)), exports in 32-bit floating-point .wav files with most tags transferred, and deletes the input files.

This project is based around the idea of reimplementing the `Image phase amplifier: 0%` setting with as much quality as possible. Actual reverse-engineering through code decompilation is outside of my skillset. While Stereo Tool does have modules that clearly use short-time Fourier Transform (STFT) processing like the Delossifier with its spectrogram display, I am now considering the idea that this setting could be using another time-frequency analysis technique such as complex wavelet transforms or constant-Q transforms, or could be using multiple all-pass filters with varying coefficients.

I think I may need outside help with this problem. I encourage someone out there to ask Thimeo about how the algorithm works, since I am unsure why they have the only plugin in the world to do this. I am not asking them since I am scared about follow-up questions.

### List of potential improvements:
* __Larger or variable block sizes:__ while 4,096 samples are good for processing on 44.1khz and 48khz audio, it theoretically becomes bad for processing low-frequencies with 96khz or 192khz audio since the minimum frequency that can be depicted will be 23.4hz and 46.8hz, respectively. This project currently uses a minimum frequency of 10hz over any sample rate.
  * Side note: Stereo Tool might use variable block sizes already with 4,096 samples as the maximum, since certain songs have shorter transients/spectral leakage using Stereo Tool compared to the equivalent setting in `fix_audio`.
* __More accurate/precise processing:__ This project uses 64-bit floating-point numbers in its calculations in order to prevent precision loss that might come with 32-bit floating-point numbers. For example, equalizing the RMS between the left and right channels using 32-bit floating-point numbers results in an error of about 0.02dB.
* __Window function:__ assuming that this Stereo Tool setting uses STFT, the varying output level of its audio implies that the window function being used introduces scalloping loss, such that the prominent frequencies will be underestimated. A flat-top window would help with this in exchange with increasing runtime.
* __Better algorithm:__ assuming that this Stereo Tool Setting uses STFT, we could implement a different algorithm which might be better than STFT. If it does not use STFT, the different algorithm could be close to the actual algorithm.
  * Complex wavelet transform: have not tried yet, current known Rust implementation is not well documented and might be abandoned
  * Constant-Q transform: current known Rust implementation uses too much memory and runtime to be viable, have not checked whether it's intrinsic to the algorithm or implementation is just not optimized well
  * Analytic signal / Hilbert transform: sounds awful and produces large clicks, does have theoretically optimal properties like having equal polarity/zero-crossings for both channels
  * Full-song FFTs instead of STFT: spectral leakage/noise is obvious, even when using better windows or more zero-padding, is theoretically optimal besides that
  * Multiple all-pass filters: Likely candidate, have not tried yet, currently researching how to even implement it
* __Better transfer of tags:__ WatchCat fails to transfer the tags of audio that contain special Unicode characters or are just too long. Covers are also not transferred at all.
* __Better detection of non-audio files:__ WatchCat, like Symphonia, detects some non-audio files like certain pictures as audio.

### Current issues:
* __Reminder: Stereo Tool runs in real-time and can utilize multiple threads.__ This implies that the solution is simple, at least enough to parallelize.
* __High-pitched zipper noise in certain songs:__ SOPHIE's "JUST LIKE WE NEVER SAID GOODBYE" produces a noticable noise when low "anti-aliasing" (minimum required overlaps) is applied.
  * Sample files are marked `11`, `12`, and `13`; reduced to the first 10% of the song (18.852 seconds)
  * Sample `13` file uses no "anti-aliasing"; actual program currently uses 32x "anti-aliasing" (almost inaudible)
    * When "anti-aliasing" is applied, the frequency seems to increase, but the magnitude of it heavily decreases.
  * Uncorrelated/out-of-phase/varying phase(?) low frequencies.
* __High peak in certain songs:__ DJ Screw's "Backstreets (Screwed)" has a high peak level of 10.04dB with `fix_audio` compared to 3.65dB with WatchCat.
  * Sample files are marked `21`, `22`, and `23`; reduced to the last 30 seconds
  * The peak level does not significantly change when using different window functions or when increasing the number of overlaps
  * Currently, the project tries to rotate the final result to reduce peak levels, making this sample file slightly outdated. However, this treats the symptom, not the cause.
  * Implies some sort of fundamental difference between the `fix_audio` algorithm and the Stereo Tool algorithm
* __Lack of partial alignment:__ `Image phase amplifier: 0%` to `Image phase amplifier: 200%` is possible in Stereo Tool, but this projects essentially implements `Image phase amplifier: 0%` only. For correctness, multiple passes through `Image phase amplifier: 50%` and its equivalent for this project should eventually converge to using `Image phase amplifier: 0%` once, where this operation is idempotent.
  * The current `align()` function could be modified to allow partial alignment if all of the phase angles are calculated using `Complex::arg()`, but this might be too slow.
  * Unsure how partial alignment could be implemented for the other algorithms, mainly with multiple all-pass filters