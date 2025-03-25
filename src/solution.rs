use crate::nfa::Nfa;
use crate::strategy::Strategy;
use std::fmt;
use std::fs;
use tera::{Context, Tera};

/// A solution to the population control problem.
pub struct Solution {
    pub nfa: Nfa,
    pub nb_states_added: usize,
    pub nb_transitions_added: usize,
    pub result: bool,
    pub maximal_winning_strategy: Strategy,
}

impl Solution {
    pub fn generate_latex(&self, output_path: &str, tikz_path: Option<&str>) {
        let template_content = include_str!("../latex/solution.template.tex");

        // Create Tera instance
        let mut tera = Tera::default();
        tera.add_raw_template("template", template_content).unwrap();

        // Create context with values
        let mut context = Context::new();

        context.insert("is_tikz_input", &tikz_path.is_some());
        if let Some(path) = tikz_path {
            context.insert("tikz_input", path);
        };
        context.insert("states", &self.nfa.states_str());
        context.insert("initial", &self.nfa.initial_states_str());
        context.insert("accepting", &self.nfa.accepting_states_str());
        context.insert("transitions", &self.nfa.transitions_str());
        context.insert(
            "answer",
            if self.result {
                "YES (controllable)"
            } else {
                "NO (uncontrollable)"
            },
        );
        context.insert("strategy", &self.maximal_winning_strategy.to_string());

        // Render template
        let rendered = tera
            .render("template", &context)
            .expect("Template rendering failed");

        //Replace the utf8 symbol omega by \omega in therendered string
        let rendered = rendered.replace("ω", "w");
        // Write to output file
        fs::write(output_path, rendered).expect("Failed to write file");
    }
}

impl fmt::Display for Solution {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let answer = if self.result {
            "controllable"
        } else {
            "uncontrollable"
        };
        writeln!(f, "Answer: {}", answer)?;
        let was_completed = self.nb_states_added > 0 || self.nb_transitions_added > 0;
        if was_completed {
            writeln!(
                f,
                "The NFA was not complete. It was turned into the following complete NFA."
            )?;
            writeln!(f, "Number of states added: {}", self.nb_states_added)?;
            writeln!(
                f,
                "Number of transitions added
            : {}",
                self.nb_transitions_added
            )?;
        } else {
            writeln!(f, "The NFA was complete.")?;
        }
        writeln!(f, "\n\nAutomaton:\n{}\n\n", self.nfa)?;
        writeln!(f, "Maximal winning random walk:\n")?;
        writeln!(f, "States:\n\t{}", self.nfa.states_str())?;
        writeln!(f, "\n{}", self.maximal_winning_strategy)
    }
}
