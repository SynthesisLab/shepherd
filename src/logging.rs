
//! This module provides functionality for setting up logging

use env_logger::Builder;
use log::LevelFilter;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

/// Sets up the logger based on verbosity and optional log file path.
pub fn setup_logger(verbosity: u8, log_output: Option<PathBuf>) {
    let mut builder = Builder::from_default_env();
    builder.format_timestamp(None);

    builder.filter_level(match verbosity {
        0 => LevelFilter::Warn,
        1 => LevelFilter::Info,
        2 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    });

    if let Some(log_path) = log_output {
        match File::create(&log_path) {
            Ok(file) => {
                let writer = Box::new(file) as Box<dyn Write + Send>;
                builder.target(env_logger::Target::Pipe(writer));
            }
            Err(_) => {
                eprintln!("Could not create log file at {}. Defaulting to stderr.", log_path.display());
                builder.target(env_logger::Target::Stderr);
            }
        }
    } else {
        builder.target(env_logger::Target::Stderr);
    }

    builder.init();
}
