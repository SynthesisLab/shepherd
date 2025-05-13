use clap::Parser;
use std::path::PathBuf;
use log::info;

use shepherd::nfa;
mod logging;



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

}

pub fn main() {
    // parse CLI arguments
    let args = Args::parse();

    // set up logging
    logging::setup_logger(args.verbosity, args.log_output);

    // parse the input file
    let nfa = nfa::Nfa::load_from_file(&args.filename, &args.input_format, &nfa::StateOrdering::Input);        

    // print the input automaton
    info!("{}", nfa);

    // compute the solution

}
