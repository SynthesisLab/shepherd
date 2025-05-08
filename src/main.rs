use shepherd;

pub fn main() {
    if let Err(e) = shepherd::run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

