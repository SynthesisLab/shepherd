//! This module defines the command line interface (CLI) for the application.

use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use crate::solver;
use crate::nfa;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum OutputFormat {
    Plain,
    Tex,
    Csv,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[arg(value_name = "AUTOMATON_FILE", help = "Path to the input")]
    pub filename: String,

    #[arg(
        short = 'f',
        long = "from",
        value_enum,
        default_value = "tikz",
        help = "The input format"
    )]
    pub input_format: nfa::InputFormat,

    #[arg(
           short = 'v',
           long = "verbose",
           action = clap::ArgAction::Count,
           help = "Increase verbosity level"
       )]
    pub verbosity: u8,

    #[arg(
        long,
        short = 'l',
        value_name = "LOG_FILE",
        help = "Optional path to the log file. Defaults to stdout if not specified."
    )]
    pub log_output: Option<PathBuf>,

    #[arg(
        value_enum,
        short = 't',
        long = "to",
        default_value = "plain",
        help = "The output format"
    )]
    pub output_format: OutputFormat,

    #[arg(
        short = 'o',
        long = "output",
        value_name = "OUTPUT_FILE",
        help = "Where to write the strategy; defaults to stdout."
    )]
    pub output_path: Option<PathBuf>,

    #[arg(
        short,
        long,
        value_enum,
        default_value = "input",
        help = "The state reordering type."
    )]
    pub state_ordering: nfa::StateOrdering,

    #[arg(
        long,
        value_enum,
        default_value = "strategy",
        help = "Solver output specification."
    )]
    pub solver_output: solver::SolverOutput,
}
