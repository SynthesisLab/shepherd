use clap::Parser;
use std::fs::File;
use std::io;
use std::io::Write;
use log::info;

use shepherd::solver;
use shepherd::nfa;

mod cli;
mod logging;

pub fn main() {
    // parse CLI arguments
    let args = cli::Args::parse();

    // set up logging
    logging::setup_logger(args.verbosity, args.log_output);

    // parse the input file
    let nfa = nfa::Nfa::load_from_file(&args.filename, &args.input_format, &args.state_ordering);

    // print the input automaton
    info!("{}", nfa);

    // compute the solution
    let solution = solver::solve(&nfa, &args.solver_output);

    // print the solution in any case.
    // This now only prints the status: controllable or not.
    match args.solver_output {
        solver::SolverOutput::Strategy => println!("\nMaximal winning strategy;\n{}", solution),
        solver::SolverOutput::YesNo => {
            println!("\nSolution\n{}", solution);
            if solution.is_controllable {
                println!(
                    "\nStrategy winning from the initial positions (might not be maximal)\n{}",
                    solution.winning_strategy
                );
            }
        }
    }

    // only if the answer was positive, format the winning strategy
    let output_strategy = match args.solver_output {
        solver::SolverOutput::Strategy => true,
        solver::SolverOutput::YesNo => solution.is_controllable,
    };
    if output_strategy {
        // create a writer were we later print the output.
        // This is either a file or simply stdout.
        let mut out_writer = match args.output_path {
            Some(path) => {
                // Open a file in write-only mode, returns `io::Result<File>`
                let file = match File::create(&path) {
                    Err(why) => panic!("couldn't create {}: {}", path.display(), why),
                    Ok(file) => file,
                };
                Box::new(file) as Box<dyn Write>
            }
            None => Box::new(io::stdout()) as Box<dyn Write>,
        };

        // prepare output string
        let output = match args.output_format {
            cli::OutputFormat::Tex => {
                let is_tikz = args.input_format == nfa::InputFormat::Tikz;
                let latex_content =
                    solution.as_latex(if is_tikz { Some(&args.filename) } else { None });
                latex_content.to_string()
            }
            cli::OutputFormat::Plain => {
                format!(
                    "States: {}\n {}",
                    nfa.states_str(),
                    solution.winning_strategy
                )
            }
            cli::OutputFormat::Csv => {
                format!(
                    "Σ, {}\n{}\n",
                    nfa.states().join(","),
                    solution.winning_strategy.as_csv()
                )
            }
        };

        // Write the winning strategy to the output
        write!(out_writer, "{}", output).expect("Couldn’t write");
    }
}
