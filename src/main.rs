use clap::Parser;
use std::fs::write;
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
        short='t',
        long="to",
        value_enum,
        default_value = "tikz",
        help = "The output format"
    )]
    output_format: nfa::InputFormat,

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

    //adds an explanation to the help message
    #[arg(long, action, help = "Do not generate tex output")]
    no_tex_output: bool,

    #[arg(long, action, help = "Do not generate pdf output")]
    no_pdf_output: bool,

    #[arg(long, default_value = "pdflatex", help = "The latex processor to use")]
    latex_processor: String,
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

    let solution = solver::solve(&nfa);

    println!("{}", solution);

    if !args.no_tex_output {
        //remove trailing path from filename
        let filename = args.filename.split('/').last().unwrap();
        let output_path_tex = format!("{}.solution.tex", filename);
        let output_path_pdf = format!("{}.solution.pdf", filename);
        let is_tikz = args.input_format == nfa::InputFormat::Tikz;
        solution.generate_latex(
            &output_path_tex,
            if is_tikz { Some(&args.filename) } else { None },
        );
        println!("Solution written to tex file './{}'", output_path_tex);
        if !args.no_pdf_output {
            print!("\nRunning pdflatex...");
            //run pdflatex on the generated file and redirect output to a log file
            let output = process::Command::new(&args.latex_processor)
                .arg("-interaction=nonstopmode")
                .arg(&output_path_tex)
                .output()
                .expect("Failed to execute pdflatex");
            println!("{}", String::from_utf8_lossy(&output.stderr));
            //check whether file output_path_pdf exists
            if !std::path::Path::new(&output_path_pdf).exists() {
                write("pdflatex_stdout.log", &output.stdout).expect("Failed to write stdout log");
                write("pdflatex_stderr.log", &output.stderr).expect("Failed to write stderr log");
                eprintln!(
                "error occurred. Check pdflatex_stdout.log and pdflatex_stderr.log for details."
            );
                process::exit(1);
            } else {
                println!("Solution written to pdf file './{}'", output_path_pdf);
            }
        }
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
