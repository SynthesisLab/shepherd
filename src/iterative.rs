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

    #[arg(value_name = "N",
           help = "dimension of the product"
       )]
    pub dim: usize,

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
    let mut nfa = nfa::Nfa::load_from_file(&args.filename, &args.input_format, &nfa::StateOrdering::Alphabetical);

    // print the input automaton
    info!("{}", nfa);

    // compute the solution
    if !nfa.is_complete() {
        info!("The automaton is not complete. Completing it...");

        match nfa.add_state("SINK") {
            Ok(sink) => {
                info!("Added sink state");
                nfa.complete(Some(sink));
            },
            Err(e) => {
                info!("Error adding sink state: {}", e);
                return;  // TODO:  handle this error properly
            }
        }
    }
    // print the complete automaton again
    info!("{}", nfa);

    // TODO: create prism input file
    let prism_model = nfa_to_prism(&nfa, args.dim);
    print!("{}", prism_model);
    // TODO: call prism
}

fn nfa_to_prism(nfa: &nfa::Nfa, n: usize) -> String {
    let mut prism_input = String::new();
    prism_input.push_str("mdp\n\n");

    // module M1 will be our NFA.
    prism_input.push_str("module M1\n");
    // define states, assume that state 0 is the initial state
    prism_input.push_str(&format!("s1 : [0..{}] init 0;\n", nfa.nb_states()-1));

    // define transitions
    for (act,am) in nfa.get_edges().iter() {    // for every alphabet letter
        for src in 0..am.dim()  {             // for all states
            let succs = am.get_successors(src); // get successors
            // prism requires explicit floating point numbers to represent distributions.
            // here we represent a uniform dist among successors.
            let prob = 1.0/succs.len() as f64;
            let update = succs.iter()
                            .map(|trg| format!("{}:(s1'={})", prob, trg))
                            .collect::<Vec<String>>()
                            .join(" + ");
            prism_input.push_str(&format!("[{act}] s1={} -> {};\n", src, update));
        }
    }
    prism_input.push_str("endmodule\n\n");


    // Add a copy of the MDP for every power up to n
    for i in 2..=n {
        prism_input.push_str(&format!("module M{i} = M1 [s1=s{i}, s{i}=s1] endmodule\n"));
    }


    // define a label representing global reachability target:
    // every component is in one of its final states.
    let mut final_line = String::from("\nlabel \"final\" = ");

    let mut conj = Vec::new();
    for i in 1..=n {
        conj.push(nfa.final_states().iter()
            .map(|f| format!("s{i}={f}"))
            .collect::<Vec<_>>()
            .join("| "));
    }
    final_line.push_str(&conj.iter()
        .map(|f| format!("( {f} )"))
        .collect::<Vec<_>>()
        .join(" & "));
    final_line.push_str(";\n");
    prism_input.push_str(&final_line);


    // define the global system as the product of all n many copies.
    // This uses prisms syntax for parallel composition.
    prism_input.push_str("\nsystem\n");
    let prod_string = (1..=n)
                    .map(|i| format!("M{i}"))
                    .collect::<Vec<String>>()
                    .join(" || ");
    prism_input.push_str(&prod_string);
    prism_input.push_str("\nendsystem\n");

    prism_input
}
