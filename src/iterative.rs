use clap::Parser;
use log::{debug, info};
use std::path::PathBuf;

use shepherd::nfa;
mod logging;
use regex::Regex;
use std::fs::File;
use std::io;
use std::io::{Read, Write};
use std::process::{Command, Stdio};

const PRISM_CMD: &str = "prism";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[arg(value_name = "AUTOMATON_FILE", help = "Path to the input")]
    pub filename: String,

    #[arg(
        value_name = "TMP_DIR",
        help = "directory where to store the prism files"
    )]
    pub tmp_dir: PathBuf,

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
    let mut nfa = nfa::Nfa::load_from_file(
        &args.filename,
        &args.input_format,
        &nfa::StateOrdering::Alphabetical,
    );

    // print the input automaton
    info!("{}", nfa);

    // compute the solution
    if !nfa.is_complete() {
        info!("The automaton is not complete. Completing it...");

        match nfa.add_state("SINK") {
            Ok(sink) => {
                info!("Added sink state");
                nfa.complete(Some(sink));
            }
            Err(e) => {
                info!("Error adding sink state: {}", e);
                return; // TODO:  handle this error properly
            }
        }
    }
    // print the complete automaton again
    info!("{}", nfa);

    let mut i = 1;
    loop {
        // create prism input string
        let prism_model = nfa_to_prism(&nfa, i);
        debug!("{}", prism_model);

        // write prism input to file
        let prism_input_path = args.tmp_dir.join(format!("model-{}.pm", i));
        if let Err(e) = write_string_to_file(&prism_model, &prism_input_path) {
            eprintln!("Error writing to file: {}", e);
            return;
        }
        info!("Wrote prism input to file: {}", prism_input_path.display());

        let value = call_prism(&[
            "-pf",
            "Pmax=? [ F \"final\" ]",
            &prism_input_path.to_string_lossy(),
        ])
        .unwrap();
        println!("n={} -> {:.3}", i, value);

        if value < 1.0 {
            println!("The value is less than 1.0, stopping the search.");
            break;
        }
        i += 1;
    }
}

fn write_string_to_file(content: &str, file_path: &PathBuf) -> io::Result<()> {
    let mut file = File::create(file_path)?;
    file.write_all(content.as_bytes())?;
    Ok(())
}

fn call_prism(args: &[&str]) -> Result<f32, ()> {
    let child = Command::new(PRISM_CMD)
        .args(args)
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to call prism");

    let stdout = child.stdout.expect("Failed to capture stdout");

    // Using stdout to read the output from the child process
    let mut output = String::new();
    io::BufReader::new(stdout)
        .read_to_string(&mut output)
        .expect("Failed to read from stdout");
    info!(
        "PRISM OUTPUT\n---------------\n{}\n-----------------",
        output
    );

    // Compile the regular expression once
    let re = Regex::new(r"Value in the initial state: (\d+\.\d+)").unwrap();

    for line in output.lines() {
        if let Some(captures) = re.captures(&line) {
            if let Some(value) = captures.get(1) {
                return value.as_str().parse::<f32>().map_err(|_| ());
            }
        }
    }
    Err(())
}

fn nfa_to_prism(nfa: &nfa::Nfa, n: usize) -> String {
    let mut prism_input = String::new();
    prism_input.push_str("mdp\n\n");

    // module M1 will be our NFA.
    prism_input.push_str("module M1\n");

    // we assume that there is only one initial state; get it.
    let initial: nfa::State = nfa.initial_states().iter().cloned().next().unwrap();

    // define states string for prism
    prism_input.push_str(&format!(
        "s1 : [0..{}] init {initial};\n",
        nfa.nb_states() - 1
    ));

    // define transitions
    for (act, am) in nfa.get_edges().iter() {
        // for every alphabet letter
        for src in 0..am.dim() {
            // for all states
            let succs = am.get_successors(src); // get successors
                                                // prism requires explicit floating point numbers to represent distributions.
                                                // here we represent a uniform dist among successors.
            let prob = 1.0 / succs.len() as f64;
            let update = succs
                .iter()
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
        conj.push(
            nfa.final_states()
                .iter()
                .map(|f| format!("s{i}={f}"))
                .collect::<Vec<_>>()
                .join("| "),
        );
    }
    final_line.push_str(
        &conj
            .iter()
            .map(|f| format!("( {f} )"))
            .collect::<Vec<_>>()
            .join(" & "),
    );
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
