use clap::{Parser, ValueEnum};
use std::fs::{write, File};
use std::io::{self, Write};
use std::path::PathBuf;
use std::process;
mod coef;
mod flow;
mod graph;
mod ideal;
mod memoizer;
mod nfa;
mod partitions;
mod semigroup;
mod sheep;
mod solution;
mod solver;
mod strategy;
use log::LevelFilter;


#[derive(Debug,Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum OutputFormat {
    Plain,
    Tex,
}

#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The path to the input file
    filename: String,

    #[arg(
        short='f',
        long="from",
        value_enum,
        default_value = "tikz",
        help = "The input format"
    )]
    input_format: nfa::InputFormat,
    
    #[arg(
        value_enum,
        short='t',
        long="to",
        default_value = "plain",
        help = "The output format"
    )]
    output_format: OutputFormat,

    /// path to write the strategy
    #[arg(
        short='o',
        long="output",
        value_name = "STRATEGY_FILE",
        help = "where to write the strategy; defaults to stdout."
    )]
    output_path: Option<PathBuf>,

    #[arg(
        short,
        long,
        value_enum,
        default_value = "input",
        help = format!("The state reordering type.\n'{:?}' preserves input order.\n\
        '{:?}' sorts by label.\n\
        '{:?}' sorts states topologically.\n", nfa::StateOrdering::Input, nfa::StateOrdering::Alphabetical, nfa::StateOrdering::Topological)
    )]
    state_ordering: nfa::StateOrdering,
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

    let args = Args::parse();


    let nfa = nfa::Nfa::load_from_file(&args.filename, &args.input_format, &args.state_ordering);
    
    // print the input automaton
    println!("{}", nfa);

    let solution = solver::solve(&nfa);

    // print the solution in any case
    println!("{}", solution);
 
    // only if the answer was positive, format the winning strategy
    if solution.result {
        // create a writer were we later print the output
        let mut out_writer = match args.output_path {
            Some(path) => {
                // Open a file in write-only mode, returns `io::Result<File>`
                let mut file = match File::create(&path) {
                    Err(why) => panic!("couldn't create {}: {}", path.display(), why),
                    Ok(file) => file,
                };
                Box::new(file) as Box<dyn Write>
            },
            None => {
                Box::new(io::stdout()) as Box<dyn Write>
            },
        };
    
        // prepare output string
        let output = match args.output_format {
            OutputFormat::Tex => {
                let is_tikz = args.input_format == nfa::InputFormat::Tikz;
                let latex_content = solution.as_latex(
                    if is_tikz { Some(&args.filename) } else { None },
                );
                format!("{}", latex_content)
            }
            OutputFormat::Plain => {
                format!("States: {}\n {}", nfa.states_str(),solution.maximal_winning_strategy)
            }
        };

        // Write the winning strategy to the output
        write!(out_writer, "{}", output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coef::{C0, C1, C2, OMEGA};
    use crate::ideal::Ideal;

    const EXAMPLE1: &str = include_str!("../examples/bottleneck-1-ab.tikz");
    const EXAMPLE1_COMPLETE: &str = include_str!("../examples/bottleneck-1-ab-complete.tikz");
    const EXAMPLE2: &str = include_str!("../examples/bottleneck-2.tikz");

    #[test]
    fn test_example_1() {
        let nfa = nfa::Nfa::from_tikz(&EXAMPLE1);
        let solution = solver::solve(&nfa);
        print!("{}", solution);
        assert!(!solution.result);
        assert_eq!(solution.maximal_winning_strategy.iter().count(), 2);
        let ideala = solution
            .maximal_winning_strategy
            .iter()
            .filter(|x| x.0 == "a")
            .map(|x| x.1)
            .next()
            .unwrap();
        let idealb = solution
            .maximal_winning_strategy
            .iter()
            .filter(|x| x.0 == "b")
            .map(|x| x.1)
            .next()
            .unwrap();

        assert_eq!(
            *ideala,
            Ideal::from_vecs(&[&[C1, C0, C0, C0, C0], &[C0, OMEGA, C0, C0, C0]])
        );
        assert_eq!(*idealb, Ideal::from_vecs(&[&[C0, C0, OMEGA, C0, C0]]));
    }

    #[test]
    fn test_example_1bis() {
        let nfa = nfa::Nfa::from_tikz(&EXAMPLE1_COMPLETE);
        let solution = solver::solve(&nfa);
        print!("{}", solution);
        assert!(!solution.result);
        assert_eq!(solution.maximal_winning_strategy.iter().count(), 2);
        let ideala = solution
            .maximal_winning_strategy
            .iter()
            .filter(|x| x.0 == "a")
            .map(|x| x.1)
            .next()
            .unwrap();
        let idealb = solution
            .maximal_winning_strategy
            .iter()
            .filter(|x| x.0 == "b")
            .map(|x| x.1)
            .next()
            .unwrap();

        assert_eq!(*ideala, Ideal::from_vecs(&[&[C1, OMEGA, C0, OMEGA, C0]]));
        assert_eq!(*idealb, Ideal::from_vecs(&[&[C0, C0, OMEGA, OMEGA, C0]]));
    }

    #[test]
    fn test_example_2() {
        let nfa = nfa::Nfa::from_tikz(&EXAMPLE2);
        let solution = solver::solve(&nfa);
        print!("{}", solution);
        assert!(!solution.result);
        assert_eq!(solution.maximal_winning_strategy.iter().count(), 4);
        let ideala = solution
            .maximal_winning_strategy
            .iter()
            .filter(|x| x.0 == "a")
            .map(|x| x.1)
            .next()
            .unwrap();

        assert_eq!(*ideala, Ideal::from_vecs(&[&[C2, C0, C0, C0, C0]]));
    }

    #[test]
    fn test_example_2_sorted_alpha() {
        let mut nfa = nfa::Nfa::from_tikz(&EXAMPLE2);
        nfa.sort(&nfa::StateOrdering::Alphabetical);
        let solution = solver::solve(&nfa);
        assert!(!solution.result);
        assert_eq!(solution.maximal_winning_strategy.iter().count(), 4);
        let ideala = solution
            .maximal_winning_strategy
            .iter()
            .filter(|x| x.0 == "a")
            .map(|x| x.1)
            .next()
            .unwrap();

        assert_eq!(*ideala, Ideal::from_vecs(&[&[C0, C0, C0, C0, C2]]));
    }

    #[test]
    fn test_example_2_sorted_topo() {
        let mut nfa = nfa::Nfa::from_tikz(&EXAMPLE2);
        nfa.sort(&nfa::StateOrdering::Topological);
        let solution = solver::solve(&nfa);
        assert!(!solution.result);
        assert_eq!(solution.maximal_winning_strategy.iter().count(), 4);
        let ideala = solution
            .maximal_winning_strategy
            .iter()
            .filter(|x| x.0 == "a")
            .map(|x| x.1)
            .next()
            .unwrap();

        assert_eq!(*ideala, Ideal::from_vecs(&[&[C2, C0, C0, C0, C0]]));
    }
}
