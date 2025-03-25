use clap::Parser;
use std::fs::File;
use std::io::{self, Read};
use std::process;
mod coef;
mod flow;
mod graph;
mod ideal;
mod nfa;
mod partitions;
mod semigroup;
mod sheep;
mod solution;
mod solver;
mod strategy;

#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    filename: String,

    #[arg(short, long, default_value = "tikz")]
    input_type: String,

    #[arg(short, long, action)]
    no_tex_output: bool,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    // Get the arguments

    let filename = args.filename;

    let nfa = match read_file(&filename) {
        Ok(content) => match args.input_type.as_str() {
            "tikz" => nfa::Nfa::from_tikz(&content),
            _ => {
                eprintln!("Invalid format: {}", args.input_type);
                eprintln!("Known formats: [tikz]");
                process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("Error reading file '{}': '{}'", &filename, e);
            process::exit(1);
        }
    };

    let solution = solver::solve(&nfa);

    println!("{}", solution);

    if !args.no_tex_output {
        let output_path = format!("{}.solution.tex", filename);
        solution.generate_latex(&output_path, Some(filename.as_str()));
    }
}

/// Reads the content of the file
fn read_file(filename: &str) -> io::Result<String> {
    let mut file = File::open(filename)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    Ok(content)
}
