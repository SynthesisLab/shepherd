use clap::Parser;
use std::fs::{write, File};
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
    #[arg(short, long, help = "The path to the input file")]
    filename: String,

    #[arg(short, long, default_value = "tikz", help = "The input format")]
    input_type: String,

    //adds an explanation to the help message
    #[arg(long, action, help = "Do not generate tex output")]
    no_tex_output: bool,

    #[arg(long, action, help = "Do not generate pdf output")]
    no_pdf_output: bool,

    #[arg(long, default_value = "pdflatex", help = "The latex processor to use")]
    latex_processor: String,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    // Get the arguments

    let tikz_path = args.filename;

    let nfa = match read_file(&tikz_path) {
        Ok(content) => match args.input_type.as_str() {
            "tikz" => nfa::Nfa::from_tikz(&content),
            _ => {
                eprintln!("Invalid format: {}", args.input_type);
                eprintln!("Known formats: [tikz]");
                process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("Error reading file '{}': '{}'", &tikz_path, e);
            process::exit(1);
        }
    };

    let solution = solver::solve(&nfa);

    println!("{}", solution);

    if !args.no_tex_output {
        //remove trailing path from filename
        let filename = tikz_path.split('/').last().unwrap();
        let output_path_tex = format!("{}.solution.tex", filename);
        let output_path_pdf = format!("{}.solution.pdf", filename);
        solution.generate_latex(&output_path_tex, Some(&tikz_path));
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

/// Reads the content of the file
fn read_file(filename: &str) -> io::Result<String> {
    let mut file = File::open(filename)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    Ok(content)
}
