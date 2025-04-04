use clap::{Parser, ValueEnum};
use std::fs::File;
use std::io;
use std::io::Write;
use std::path::PathBuf;
mod coef;
mod downset;
mod flow;
mod graph;
mod ideal;
mod memoizer;
mod nfa;
mod partitions;
mod semigroup;
mod solution;
mod solver;
mod strategy;
use log::LevelFilter;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum OutputFormat {
    Plain,
    Tex,
    Csv,
}

#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(value_name = "AUTOMATON_FILE", help = "path to the input")]
    filename: String,

    #[arg(
        short = 'f',
        long = "from",
        value_enum,
        default_value = "tikz",
        help = "The input format"
    )]
    input_format: nfa::InputFormat,

    #[arg(
        value_enum,
        short = 't',
        long = "to",
        default_value = "plain",
        help = "The output format"
    )]
    output_format: OutputFormat,

    /// path to write the strategy
    #[arg(
        short = 'o',
        long = "output",
        value_name = "OUTPUT_FILE",
        help = "where to write the strategy; defaults to stdout."
    )]
    output_path: Option<PathBuf>,

    #[arg(
        short,
        long,
        value_enum,
        default_value = "input",
        help = format!("The state reordering type: preserves input order, sorts alphabetically or topologically.")
    )]
    state_ordering: nfa::StateOrdering,

    #[arg(
        long,
        value_enum,
        default_value = "strategy",
        help = format!("The solver output. Either yes/no and a winning strategy (the faster). Or the full maximal winning strategy.")
    )]
    solver_output: solver::SolverOutput,
}

fn main() {
    #[cfg(debug_assertions)]
    env_logger::Builder::new()
        .filter_level(LevelFilter::Debug)
        .init();

    #[cfg(not(debug_assertions))]
    env_logger::Builder::new()
        .filter_level(LevelFilter::Info)
        .init();

    // parse CLI arguments
    let args = Args::parse();

    // parse the input file
    let nfa = nfa::Nfa::load_from_file(&args.filename, &args.input_format, &args.state_ordering);

    // print the input automaton
    println!("{}", nfa);

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
            OutputFormat::Tex => {
                let is_tikz = args.input_format == nfa::InputFormat::Tikz;
                let latex_content =
                    solution.as_latex(if is_tikz { Some(&args.filename) } else { None });
                latex_content.to_string()
            }
            OutputFormat::Plain => {
                format!(
                    "States: {}\n {}",
                    nfa.states_str(),
                    solution.winning_strategy
                )
            }
            OutputFormat::Csv => {
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
