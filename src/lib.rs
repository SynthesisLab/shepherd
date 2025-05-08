use clap::Parser;
use std::fs::File;
use std::io;
use std::io::Write;
use log::info;

pub mod coef;
pub mod downset;
pub mod flow;
pub mod graph;
pub mod ideal;
pub mod memoizer;
pub mod nfa;
pub mod partitions;
pub mod semigroup;
pub mod solution;
pub mod solver;
pub mod strategy;

mod cli;
mod logging;

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coef::{C0, C1, C2, OMEGA};
    use crate::downset::DownSet;
    use crate::ideal::Ideal;

    const EXAMPLE1: &str = include_str!("../examples/bottleneck-1-ab.tikz");
    const EXAMPLE1_COMPLETE: &str = include_str!("../examples/bottleneck-1-ab-complete.tikz");
    const EXAMPLE2: &str = include_str!("../examples/bottleneck-2.tikz");
    const EXAMPLE_BUG12: &str = include_str!("../examples/bug12.tikz");

    #[test]
    fn test_example_1() {
        let nfa = nfa::Nfa::from_tikz(EXAMPLE1);
        let solution = solver::solve(&nfa, &solver::SolverOutput::YesNo);
        print!("{}", solution);
        assert!(!solution.is_controllable);
        assert_eq!(solution.winning_strategy.iter().count(), 2);
        let downseta = solution
            .winning_strategy
            .iter()
            .filter(|x| x.0 == "a")
            .map(|x| x.1)
            .next()
            .unwrap();
        let downsetb = solution
            .winning_strategy
            .iter()
            .filter(|x| x.0 == "b")
            .map(|x| x.1)
            .next()
            .unwrap();

        assert_eq!(
            *downseta,
            DownSet::from_vecs(&[&[C1, C0, C0, C0, C0], &[C0, OMEGA, C0, C0, C0]])
        );
        assert_eq!(*downsetb, DownSet::from_vecs(&[&[C0, C0, OMEGA, C0, C0]]));
    }

    #[test]
    fn test_example_1bis() {
        let nfa = nfa::Nfa::from_tikz(EXAMPLE1_COMPLETE);
        let solution = solver::solve(&nfa, &solver::SolverOutput::YesNo);
        print!("{}", solution);
        assert!(!solution.is_controllable);
        assert_eq!(solution.winning_strategy.iter().count(), 2);
        let downseta = solution
            .winning_strategy
            .iter()
            .filter(|x| x.0 == "a")
            .map(|x| x.1)
            .next()
            .unwrap();
        let downsetb = solution
            .winning_strategy
            .iter()
            .filter(|x| x.0 == "b")
            .map(|x| x.1)
            .next()
            .unwrap();

        assert_eq!(
            *downseta,
            DownSet::from_vecs(&[&[C1, OMEGA, C0, OMEGA, C0]])
        );
        assert_eq!(
            *downsetb,
            DownSet::from_vecs(&[&[C0, C0, OMEGA, OMEGA, C0]])
        );
    }

    #[test]
    fn test_example_2() {
        let nfa = nfa::Nfa::from_tikz(EXAMPLE2);
        let solution = solver::solve(&nfa, &solver::SolverOutput::Strategy);
        print!("{}", solution);
        assert!(!solution.is_controllable);
        assert_eq!(solution.winning_strategy.iter().count(), 4);
        let downseta = solution
            .winning_strategy
            .iter()
            .filter(|x| x.0 == "a")
            .map(|x| x.1)
            .next()
            .unwrap();

        assert_eq!(*downseta, DownSet::from_vecs(&[&[C2, C0, C0, C0, C0]]));
    }

    #[test]
    fn test_example_2_sorted_alpha() {
        let mut nfa = nfa::Nfa::from_tikz(EXAMPLE2);
        nfa.sort(&nfa::StateOrdering::Alphabetical);
        let solution = solver::solve(&nfa, &solver::SolverOutput::Strategy);
        assert!(!solution.is_controllable);
        assert_eq!(solution.winning_strategy.iter().count(), 4);
        let downseta = solution
            .winning_strategy
            .iter()
            .filter(|x| x.0 == "a")
            .map(|x| x.1)
            .next()
            .unwrap();

        assert_eq!(*downseta, DownSet::from_vecs(&[&[C0, C0, C0, C0, C2]]));
    }

    #[test]
    fn test_example_2_sorted_topo() {
        let mut nfa = nfa::Nfa::from_tikz(EXAMPLE2);
        nfa.sort(&nfa::StateOrdering::Topological);
        let solution = solver::solve(&nfa, &solver::SolverOutput::Strategy);
        assert!(!solution.is_controllable);
        assert_eq!(solution.winning_strategy.iter().count(), 4);
        let downseta = solution
            .winning_strategy
            .iter()
            .filter(|x| x.0 == "a")
            .map(|x| x.1)
            .next()
            .unwrap();

        assert_eq!(*downseta, DownSet::from_vecs(&[&[C2, C0, C0, C0, C0]]));
    }

    #[test]
    fn test_bug12() {
        let mut nfa = nfa::Nfa::from_tikz(EXAMPLE_BUG12);
        nfa.sort(&nfa::StateOrdering::Topological);
        let solution = solver::solve(&nfa, &solver::SolverOutput::Strategy);
        let downsetb = solution
            .winning_strategy
            .iter()
            .filter(|x| x.0 == "b")
            .map(|x| x.1)
            .next()
            .unwrap();
        println!("{}", downsetb);
        assert!(downsetb.contains(&Ideal::from_vec(vec![C2, C0, C0, C0, C0, C0, C0, C0])));
    }
}
