/*
authors @GBathie + @Numero7
 */

use crate::graph::Graph;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fmt;

pub type State = usize;
pub type Letter = String;

#[derive(Clone, Debug)]
pub struct Transition {
    pub from: State,
    pub label: Letter,
    pub to: State,
}

#[derive(Debug, Clone)]
pub struct Nfa {
    states: Vec<String>,
    initial: HashSet<State>,
    accepting: HashSet<State>,
    transitions: Vec<Transition>,
}

impl Nfa {
    pub fn from_size(nb_states: usize) -> Self {
        Nfa {
            states: (0..nb_states).map(|index| index.to_string()).collect(),
            initial: HashSet::new(),
            accepting: HashSet::new(),
            transitions: vec![],
        }
    }

    #[allow(dead_code)]
    pub fn from_states(states: &[&str]) -> Self {
        Nfa {
            states: states.iter().map(|&l| l.to_string()).collect(),
            initial: HashSet::new(),
            accepting: HashSet::new(),
            transitions: vec![],
        }
    }

    pub fn from_tikz(input: &str) -> Self {
        let state_re = Regex::new(
            r"\\node\[(?P<attrs>[^\]]*)\]\s*at\s*\([^)]+\)\s*\((?P<id>\w+)\)\s*\{\$(?P<name>[^$]+)\$\}",
        )
        .unwrap();
        let edge_re =
            Regex::new(r"\((?P<from>\w+)\)\s*edge.*?\{\$(?P<label>[^$]+)\$\}\s*\((?P<to>\w+)\)")
                .unwrap();

        let mut states: Vec<String> = Vec::new(); //preserves appearance order in file
        let mut names: HashMap<String, String> = HashMap::new();
        let mut initials: HashSet<String> = HashSet::new();
        let mut finals: HashSet<String> = HashSet::new();
        let mut transitions: Vec<(String, String, String)> = Vec::new();

        for cap in state_re.captures_iter(input) {
            let id = cap["id"].to_string();
            let name = cap["name"].to_string();
            if !states.contains(&id) {
                states.push(id.clone());
            }
            names.insert(id.clone(), name);

            let attrs = &cap["attrs"];
            if attrs.contains("initial") {
                initials.insert(id.clone());
            }
            if attrs.contains("accepting") {
                finals.insert(id);
            }
        }

        for cap in edge_re.captures_iter(input) {
            let from = cap["from"].to_string();
            let to = cap["to"].to_string();
            let label = cap["label"].to_string();
            //split label according to ',' separator, and trim the result
            let labels: Vec<&str> = label.split(',').map(|x| x.trim()).collect();
            for label in labels {
                transitions.push((from.clone(), label.to_string(), to.clone()));
            }
        }

        let mut nfa = Nfa {
            states: states.iter().map(|s| names[s].to_string()).collect(),
            initial: HashSet::new(),
            accepting: HashSet::new(),
            transitions: vec![],
        };
        for state in initials {
            nfa.add_initial(&names[&state]);
        }
        for state in finals {
            nfa.add_final(&names[&state]);
        }
        for (from, label, to) in transitions {
            nfa.add_transition(&names[&from], &names[&to], &label);
        }
        nfa
    }

    pub fn get_alphabet(&self) -> Vec<&str> {
        let mut letters = Vec::new();
        self.transitions.iter().for_each(|t| {
            let label = t.label.as_str();
            if !letters.contains(&label) {
                letters.push(label);
            }
        });
        letters
    }

    pub fn add_transition_by_index(&mut self, from: State, to: State, label: char) {
        self.check_state(from);
        self.check_state(to);
        self.transitions.push(Transition {
            from,
            label: label.to_string(),
            to,
        });
    }

    pub fn add_transition(&mut self, from: &str, to: &str, label: &str) {
        let from = self.get_state_index(from);
        let to = self.get_state_index(to);
        self.check_state(from);
        self.check_state(to);
        self.transitions.push(Transition {
            from,
            label: label.to_string(),
            to,
        });
    }

    fn check_state(&self, q: State) {
        assert!(q < self.nb_states(), "State {} is not in the NFA", q)
    }

    pub fn add_initial_by_index(&mut self, q: State) {
        self.check_state(q);
        self.initial.insert(q);
    }

    pub fn add_final_by_index(&mut self, q: State) {
        self.check_state(q);
        self.accepting.insert(q);
    }

    pub fn add_initial(&mut self, q: &str) {
        self.initial.insert(self.get_state_index(q));
    }

    pub fn add_final(&mut self, q: &str) {
        self.accepting.insert(self.get_state_index(q));
    }

    pub fn nb_states(&self) -> usize {
        self.states.len()
    }

    pub(crate) fn initial_states(&self) -> HashSet<State> {
        self.initial.clone()
    }

    pub(crate) fn final_states(&self) -> Vec<State> {
        self.accepting.iter().cloned().collect()
    }

    ///completes the automata by adding an extra sink state, if needed.
    /// and missing transitions are redrected to the sink state.
    /// returns true iff the automaton was modified
    pub(crate) fn turn_into_complete_nfa(nfa: &Nfa) -> Option<Self> {
        if nfa.is_complete() {
            None
        } else {
            let mut nfa = nfa.clone();
            nfa.complete_with_sink();
            Some(nfa)
        }
    }

    fn complete_with_sink(&mut self) -> bool {
        if self.is_complete() {
            return false;
        }
        //look for a state from which there is no sequence of transitons to a final state
        let mut not_sink_states = self.final_states().into_iter().collect::<HashSet<_>>();
        loop {
            let mut changed = false;
            for t in self.transitions.iter() {
                if not_sink_states.contains(&t.to) && !not_sink_states.contains(&t.from) {
                    not_sink_states.insert(t.from);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let existing_sink_state =
            (0..self.nb_states()).find(|state| !not_sink_states.contains(state));
        let sink_index = match existing_sink_state {
            Some(s) => s,
            None => {
                let mut sink_name = "sink".to_string();
                while self.states.contains(&sink_name) {
                    sink_name.push('#');
                }
                self.states.push(sink_name);
                self.nb_states() - 1
            }
        };
        //bad performance, not a big deal for now
        let missing_transitions: Vec<(State, Letter)> = self
            .get_alphabet()
            .iter()
            .flat_map(|letter| {
                (0..self.nb_states())
                    .filter(|&state| {
                        !self
                            .transitions
                            .iter()
                            .any(|t| t.from == state && t.label == *letter)
                    })
                    .map(|state| (state, letter.to_string()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(
            !missing_transitions.is_empty(),
            "No missing transitions found, although the NFA failed the complete test"
        );

        for (state, letter) in missing_transitions {
            self.add_transition_by_index(state, sink_index, letter.chars().next().unwrap());
        }
        true
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.get_alphabet().iter().all(|letter| {
            (0..self.nb_states()).all(|state| {
                self.transitions
                    .iter()
                    .any(|t| t.from == state && t.label == *letter)
            })
        })
    }

    //overload [] opertor to turn state labels to state index
    pub fn get_state_index(&self, label: &str) -> State {
        self.states
            .iter()
            .position(|x| x == label)
            .expect("State not found")
    }

    pub(crate) fn get_nfa(name: &str) -> Nfa {
        match name {
            "((a#b){a,b})#" => {
                let mut nfa = Nfa::from_size(6);
                nfa.add_initial_by_index(0);
                nfa.add_final_by_index(4);
                nfa.add_transition_by_index(0, 0, 'a');
                nfa.add_transition_by_index(0, 1, 'a');
                nfa.add_transition_by_index(1, 0, 'a');
                nfa.add_transition_by_index(1, 1, 'a');
                nfa.add_transition_by_index(4, 4, 'a');
                nfa.add_transition_by_index(5, 5, 'a');

                nfa.add_transition_by_index(0, 0, 'b');
                nfa.add_transition_by_index(4, 4, 'b');
                nfa.add_transition_by_index(5, 5, 'b');

                nfa.add_transition_by_index(1, 2, 'b');
                nfa.add_transition_by_index(1, 3, 'b');

                nfa.add_transition_by_index(2, 4, 'a');
                nfa.add_transition_by_index(2, 5, 'b');
                nfa.add_transition_by_index(3, 4, 'b');
                nfa.add_transition_by_index(3, 5, 'a');
                nfa
            }
            _ => panic!("Unknown NFA"),
        }
    }

    pub(crate) fn get_support(&self, action: &str) -> crate::graph::Graph {
        Graph::new(
            &self
                .transitions
                .iter()
                .filter(|t| t.label == *action)
                .map(|t| (t.from, t.to))
                .collect::<Vec<_>>(),
        )
    }
}

impl fmt::Display for Nfa {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "NFA\n")?;
        writeln!(f, "States: {:?}", self.states)?;
        writeln!(f, "Initial: {:?}", self.initial)?;
        writeln!(f, "Accepting: {:?}", self.accepting)?;
        writeln!(f, "Transitions: {:?}", self.transitions)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn create() {
        let mut nfa = Nfa::from_states(&["toto", &"titi"]);
        nfa.add_transition("toto", "titi", "label1");
        nfa.add_transition("titi", "toto", "label2");
        nfa.add_initial("toto");
        nfa.add_final("titi");
    }

    #[test]
    fn parity() {
        let mut nfa = Nfa::from_size(2);
        nfa.add_transition_by_index(0, 1, 'a');
        nfa.add_transition_by_index(1, 0, 'a');
        nfa.add_transition_by_index(0, 0, 'b');
        nfa.add_transition_by_index(1, 1, 'b');
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(0);

        let letters = nfa.get_alphabet();
        assert!(letters.contains(&"a"));
        assert!(letters.contains(&"b"));
        assert!(letters.len() == 2);
    }

    #[test]
    fn make_complete1() {
        let mut nfa = Nfa::from_size(2);
        nfa.add_transition_by_index(0, 1, 'a');
        nfa.add_transition_by_index(1, 1, 'a');
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(0);
        assert!(nfa.is_complete());
        let changed = nfa.complete_with_sink();
        assert!(!changed);
        assert!(nfa.nb_states() == 2);
        assert_eq!(nfa.transitions.len(), 2);
    }

    #[test]
    fn make_complete2() {
        let mut nfa = Nfa::from_size(2);
        nfa.add_transition_by_index(0, 1, 'a');
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(0);
        assert!(!nfa.is_complete());
        let changed = nfa.complete_with_sink();
        assert!(changed);
        assert!(nfa.is_complete());
        //no state was added
        assert!(nfa.nb_states() == 2);
        //a transition from 1 to 1 was added
        assert_eq!(nfa.transitions.len(), 2);
    }

    #[test]
    fn make_complete3() {
        let mut nfa = Nfa::from_size(2);
        nfa.add_transition_by_index(0, 1, 'a');
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(1);
        assert!(!nfa.is_complete());
        let changed = nfa.complete_with_sink();
        assert!(changed);
        assert!(nfa.is_complete());
        //a new state was added
        assert!(nfa.nb_states() == 3);
        assert!(nfa.states.contains(&"sink".to_string()));
        //a transition from 1 to 2 was added
        assert_eq!(nfa.transitions.len(), 3);
    }

    #[test]
    fn a_b_star() {
        let mut nfa = Nfa::from_size(2);
        nfa.add_transition_by_index(0, 1, 'a');
        nfa.add_transition_by_index(1, 0, 'b');
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(0);
    }

    #[test]
    fn tikz() {
        let nfa = Nfa::from_tikz(
            r#"
            %% Machine generated by https://finsm.io
%% 2025-3-21-5:56:39
%% include in preamble:
%% \usepackage{tikz}
%% \usetikzlibrary{automata,positioning,arrows}
\begin{center}
\begin{tikzpicture}[]
\node[initial,thick,state] at (-3.175,4.95) (1fa0116c) {$ini$};
\node[thick,state] at (1.275,4.825) (4c126865) {$ready$};
\node[thick,accepting,state] at (6.85,5.1) (b8befb7d) {$barn$};
\node[thick,state] at (4.125,6.2) (316b0ce4) {$left$};
\node[thick,state] at (4.175,3.475) (6e65ff45) {$right$};
\node[thick,state] at (6.5,8) (8a7c360d) {$wolf$};
\node[thick,state] at (6.775,2.075) (8a7c360d) {$wolf$};
\path[->, thick, >=stealth]
(1fa0116c) edge [loop,min distance = 1.25cm,above,in = 121, out = 59] node {$a,b$} (1fa0116c)
(1fa0116c) edge [above,in = 153, out = 24] node {$a$} (4c126865)
(4c126865) edge [loop,min distance = 1.25cm,above,in = 121, out = 59] node {$a$} (4c126865)
(4c126865) edge [below,in = -24, out = -160] node {$a$} (1fa0116c)
(4c126865) edge [right,in = -154, out = 26] node {$b$} (316b0ce4)
(4c126865) edge [left,in = 155, out = -25] node {$b$} (6e65ff45)
(b8befb7d) edge [loop,min distance = 1.25cm,above,in = 121, out = 59] node {$a,b$} (b8befb7d)
(316b0ce4) edge [left,in = 158, out = -22] node {$a$} (b8befb7d)
(316b0ce4) edge [right,in = -143, out = 37] node {$b$} (8a7c360d)
(6e65ff45) edge [right,in = -149, out = 31] node {$b$} (b8befb7d)
(6e65ff45) edge [left,in = 152, out = -28] node {$a$} (8a7c360d)
(8a7c360d) edge [loop,min distance = 1.25cm,above,in = 121, out = 59] node {$a,b$} (8a7c360d)
(8a7c360d) edge [loop,min distance = 1.25cm,above,in = 121, out = 59] node {$a,b$} (8a7c360d)
;
\end{tikzpicture}
\end{center}
            "#,
        );
        //print!("{:?}", nfa);
        assert_eq!(nfa.states.len(), 6);
        for state in nfa.states.iter() {
            //allow duplicates of state with different tikz ids but same label
            assert!(["ini", "ready", "barn", "left", "right", "wolf"].contains(&state.as_str()));
        }
        assert_eq!(nfa.initial_states().len(), 1);
        assert_eq!(nfa.final_states().len(), 1);
        assert!(nfa.is_complete());
        assert_eq!(nfa.get_alphabet(), ["a", "b"]);

        let mut succ_a_0 = nfa.get_support("a").get_successors(0);
        succ_a_0.sort();
        assert_eq!(succ_a_0, vec![0, 1]);
    }
}
