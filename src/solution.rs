use crate::nfa::Nfa;
use crate::strategy::Strategy;
use std::fmt;
use tera::{Context, Tera};

/// A solution to the population control problem.
pub struct Solution {
    pub nfa: Nfa,
    pub result: bool,
    pub maximal_winning_strategy: Strategy,
}

impl Solution {
    pub fn as_latex(&self, tikz_path: Option<&str>) -> String {
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
        rendered.replace("ω", "w")
    }
}

impl fmt::Display for Solution {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let answer = if self.result {
            "controllable"
        } else {
            "uncontrollable"
        };
        writeln!(f, "Answer: {}", answer)
    }
}
