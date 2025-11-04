/// Decoding module
mod decoding;
/// Exporting module
mod exporting;
/// FFT-specific module for processing.rs
mod fft;
/// Processing module
mod processing;

use realfft::RealFftPlanner;
use std::{
    fs,
    io::{self, Write as _},
    path::{self},
    time,
};
use symphonia::core::errors::Error;

use crate::decoding::get_samples_and_metadata;
use crate::exporting::export_audio;
use crate::processing::process_samples;

// Hard-coded directories
/// Input directory, created on first run
const INPUT_DIR: &str = "./inputs/";
/// Output directory
const OUTPUT_DIR: &str = "./outputs/";

/// Recursive file path retriever from `StackOverflow`
/// TODO: check if there's a newer std function or crate to do this
fn get_paths(directory: path::PathBuf) -> io::Result<Vec<path::PathBuf>> {
    let mut entries: Vec<path::PathBuf> = vec![];
    let folder_read = fs::read_dir(directory)?;

    for entry in folder_read {
        let unwrapped_entry = entry?;
        let meta = unwrapped_entry.metadata()?;

        if meta.is_dir() {
            let mut subdir = get_paths(unwrapped_entry.path())?;
            entries.append(&mut subdir);
        }

        if meta.is_file() {
            entries.push(unwrapped_entry.path());
        }
    }
    Ok(entries)
}

/// Main function to execute
fn main() -> Result<(), Error> {
    // Keeping the time for benchmarking
    let time = time::Instant::now();

    // Check if INPUT_DIR exists, or create it if it doesn't
    match fs::exists(INPUT_DIR) {
        Ok(true) => {}
        Ok(false) => {
            _ = fs::create_dir_all(INPUT_DIR);
            println!("Notice: Inputs folder created. Copy audio files here to process them.");
            return Ok(());
        }
        // `Symphonia` has a wrapper for IoErrors
        Err(err) => return Err(Error::IoError(err)),
    }

    // Get list of files in INPUT_DIR
    let entries: Vec<path::PathBuf> = get_paths(INPUT_DIR.into())?;

    let mut planner = RealFftPlanner::new();
    println!("Setup and file-exploring time: {:#?}", time.elapsed());
    for entry in entries {
        let stripped_entry = entry.strip_prefix(INPUT_DIR).unwrap();
        println!("Found file: {}", stripped_entry.display());
        print!("	Decoding...");
        io::stdout().flush()?; // Show print instantly
        let mut output_path = path::PathBuf::from(OUTPUT_DIR).join(stripped_entry);

        let channels: (Vec<f64>, Vec<f64>);
        let sample_rate: u32;
        // If we can't properly decode the data, it's likely not audio data
        // In this case, we just send it to OUTPUT_DIR so it's not spitting the warning every run
        match get_samples_and_metadata(&entry) {
            Ok(data) => {
                channels = data.0;
                sample_rate = data.1;
            }
            // Catches most non-audio files such as pictures
            // Also catches detected-but-unimplemented audio like Opus files
            Err(Error::Unsupported(_)) => {
                println!("	Not audio or just unsupported; sent to output.");
                fs::create_dir_all(output_path.parent().unwrap())?;
                fs::rename(entry, output_path)?;
                continue;
            }
            // Catches a few non-audio files such as pictures with interesting metadata (.png or .gif with .jpg data?)
            Err(Error::IoError(err)) => {
                if err.kind() == io::ErrorKind::UnexpectedEof {
                    println!("	Not audio; sent to output.");
                    fs::create_dir_all(output_path.parent().unwrap())?;
                    fs::rename(entry, output_path)?;
                    continue;
                }
                return Err(Error::IoError(err)); // Except here, where an actual audio file failed decoding
            }
            // Unsure if this even happens
            Err(Error::DecodeError(err)) => {
                println!("	Decoding error ({err}); sent to output.");
                fs::create_dir_all(output_path.parent().unwrap())?;
                fs::rename(entry, output_path)?;
                continue;
            }
            Err(other) => return Err(other), // some other unknown error
        }

        print!("	Processing... ");
        io::stdout().flush()?;
        let modified_audio = process_samples(&mut planner, channels, sample_rate);

        print!("	Exporting...");
        io::stdout().flush()?;
        output_path.set_extension("wav");
        fs::create_dir_all(output_path.parent().unwrap())?;
        export_audio(&output_path, &modified_audio, sample_rate);

        println!("	T+{:#?} ", time.elapsed());
    }
    Ok(())
}
