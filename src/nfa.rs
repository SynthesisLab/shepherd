/*
authors @GBathie + @Numero7
 */
use crate::graph::Graph;
use clap::ValueEnum;
use dot_parser::*;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{self, Read};

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

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
pub enum InputFormat {
    Dot,
    Tikz,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum StateOrdering {
    Input,
    Alphabetical,
    Topological,
}

impl Nfa {
    #[allow(dead_code)]
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

    pub fn from_dot(input: &str) -> Self {
        // intermediate boxes to hold values
        let mut states: Vec<String> = Vec::new(); //preserves appearance order in file
        let mut names: HashMap<String, String> = HashMap::new();
        let mut initials: HashSet<String> = HashSet::new();
        let mut finals: HashSet<String> = HashSet::new();
        let mut transitions: Vec<(String, String, String)> = Vec::new();

        // get a graph from the DOT string
        let graph = canonical::Graph::from(ast::Graph::try_from(input).unwrap());

        // extract nodes with labels:
        // - ignore state with label "init"
        // - interpret nodes with attribute "shape:doublecircle" as accepting states
        for (id, node) in graph.nodes.set {
            //println!("{:#?}", node);  // help me debug

            // skip over artificial "init" state
            if id.eq("init") {
                continue;
            }

            //println!("state {}", id);
            states.push(id.clone()); // keep node.id as state id

            for (k, v) in node.attr.elems {
                //println!("  {k}:{v}");
                if k.eq("label") {
                    // remove double quotes around node labels
                    let l = v.trim_matches(|c| c == '"');
                    //println!("  Label: {}", l);
                    names.insert(node.id.clone(), l.to_string());
                }
                if k.eq("shape") && v.eq("doublecircle") {
                    //println!("state {} is accepting", node.id);
                    finals.insert(node.id.clone());
                }
            }
        }

        // interpret edges with labels
        // also define set of initial states as those where "init" has an edge into.
        for edge in graph.edges.set {
            //println!("{:#?}", edge);

            // if an edge from init to X exists then X interpreted as initial state.
            if edge.from.eq("init") {
                //println!("{:#?}", edge);
                initials.insert(edge.to.clone());
            }
            for (k, v) in edge.attr.elems {
                //println!("  {k}:{v}");
                if k.eq("label") {
                    // remove double quotes around labels
                    let l = v.trim_matches(|c| c == '"');
                    transitions.push((edge.from.clone(), l.to_string(), edge.to.clone()));
                    //println!("{} --{}--> {} ", edge.from, l, edge.to);
                }
            }
        }

        // Create NFA struct and filling it with data from auxiliary boxes
        let mut nfa = Nfa {
            states,
            initial: HashSet::new(),
            accepting: HashSet::new(),
            transitions: vec![],
        };
        for state in initials.iter() {
            //println!("IN {:#?}", state);
            nfa.add_initial(state);
        }
        for state in finals {
            nfa.add_final(&state);
        }
        for (from, label, to) in transitions {
            nfa.add_transition(&from, &to, &label);
        }
        nfa
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

    #[allow(dead_code)]
    pub fn add_transition_by_index1(&mut self, from: State, to: State, label: char) {
        self.check_state(from);
        self.check_state(to);
        self.transitions.push(Transition {
            from,
            label: label.to_string(),
            to,
        });
    }

    #[allow(dead_code)]
    pub fn add_transition_by_index2(&mut self, from: State, to: State, label: &str) {
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

    #[allow(dead_code)]
    pub fn add_initial_by_index(&mut self, q: State) {
        self.check_state(q);
        self.initial.insert(q);
    }

    #[allow(dead_code)]
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

    pub fn states_str(&self) -> String {
        format!("( {} )", self.states.join(" , "))
    }

    pub fn initial_states_str(&self) -> String {
        self.initial
            .iter()
            .map(|&i| self.states[i].as_str())
            .collect::<Vec<_>>()
            .join(" , ")
    }

    pub fn accepting_states_str(&self) -> String {
        self.accepting
            .iter()
            .map(|&i| self.states[i].as_str())
            .collect::<Vec<_>>()
            .join(" , ")
    }

    pub fn transitions_str(&self) -> String {
        self.transitions
            .iter()
            .map(|t| {
                format!(
                    "\t{} --{}--> {}",
                    self.states[t.from], t.label, self.states[t.to]
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn get_edges(&self) -> HashMap<Letter, Graph> {
        self.get_alphabet()
            .iter()
            .map(|action| (action.to_string(), self.get_support(action)))
            .collect()
    }

    pub(crate) fn initial_states(&self) -> HashSet<State> {
        self.initial.clone()
    }

    pub(crate) fn final_states(&self) -> Vec<State> {
        self.accepting.iter().cloned().collect()
    }

    //overload [] opertor to turn state labels to state index
    pub fn get_state_index(&self, label: &str) -> State {
        self.states
            .iter()
            .position(|x| x == label)
            .expect("State not found")
    }

    pub(crate) fn get_support(&self, action: &str) -> crate::graph::Graph {
        Graph::new(
            self.states.len(),
            &self
                .transitions
                .iter()
                .filter(|t| t.label == *action)
                .map(|t| (t.from, t.to))
                .collect::<Vec<_>>(),
        )
    }

    /// Reads the content of the file
    fn read_file(filename: &str) -> io::Result<String> {
        let mut file = File::open(filename)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        Ok(content)
    }

    pub(crate) fn load_from_file(
        path: &str,
        input_type: &InputFormat,
        state_ordering: &StateOrdering,
    ) -> Self {
        let mut nfa = match Self::read_file(path) {
            Ok(content) => match input_type {
                InputFormat::Tikz => Self::from_tikz(&content),
                InputFormat::Dot => Self::from_dot(&content),
            },
            Err(e) => {
                panic!("Error reading file '{}': '{}'", &path, e);
            }
        };
        nfa.sort(state_ordering);
        nfa
    }

    //allow useless pub
    #[allow(unused)]
    pub fn sort(&mut self, state_ordering: &StateOrdering) {
        match state_ordering {
            StateOrdering::Input => {}
            StateOrdering::Alphabetical => {
                let mut states_indices = (0..self.nb_states()).collect::<Vec<_>>();
                states_indices.sort_by_key(|&i| self.states[i].as_str());
                self.apply_reordering(&states_indices);
            }
            StateOrdering::Topological => {
                self.sort_states_topologically();
            }
        }
    }

    fn apply_reordering(&mut self, new_order: &[usize]) {
        //blue monday
        let old_to_new: Vec<_> = new_order
            .iter()
            .map(|&i| new_order.iter().position(|&x| new_order[x] == i).unwrap())
            .collect();
        self.states = new_order.iter().map(|&i| self.states[i].clone()).collect();
        self.transitions.iter_mut().for_each(|t| {
            t.from = old_to_new[t.from];
            t.to = old_to_new[t.to];
        });
        self.initial = self.initial.iter().map(|i| old_to_new[*i]).collect();
        self.accepting = self.accepting.iter().map(|i| old_to_new[*i]).collect();
    }

    fn sort_states_topologically(&mut self) {
        //we want to sort states topologically
        let mut successor_relation = HashMap::new();
        for state in (0..self.nb_states()).collect::<Vec<_>>() {
            //create aset at index state with state as its only element
            successor_relation
                .entry(state)
                .or_insert_with(HashSet::new)
                .insert(state);
        }
        loop {
            let mut changed = false;
            for t in self.transitions.iter() {
                for successors in successor_relation.values_mut() {
                    if successors.contains(&t.from) {
                        changed |= successors.insert(t.to);
                    }
                }
            }
            if !changed {
                break;
            }
        }
        //reorder the vector state, first according to successor relation of its indices
        //and then according to alphabetical order
        let mut states_indices = (0..self.nb_states()).collect::<Vec<_>>();
        states_indices.sort_by(|&a, &b| {
            let succa = successor_relation.get(&a).unwrap();
            let succb = successor_relation.get(&b).unwrap();
            if succa.contains(&b) && succb.contains(&a) {
                std::cmp::Ordering::Equal
            } else if succa.contains(&b) {
                std::cmp::Ordering::Less
            } else if succb.contains(&a) {
                std::cmp::Ordering::Greater
            } else {
                self.states[a].cmp(&self.states[b])
            }
        });
        self.apply_reordering(&states_indices);
    }
}

impl fmt::Display for Nfa {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "NFA\n")?;
        writeln!(f, "States: {{ {} }}", self.states.join(" , "))?;
        writeln!(f, "Initial: {{ {} }}", self.initial_states_str())?;
        writeln!(f, "Accepting: {{ {} }}", self.accepting_states_str())?;
        writeln!(f, "Transitions:\n{}", self.transitions_str())
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
        nfa.add_transition_by_index1(0, 1, 'a');
        nfa.add_transition_by_index1(1, 0, 'a');
        nfa.add_transition_by_index1(0, 0, 'b');
        nfa.add_transition_by_index1(1, 1, 'b');
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(0);

        let letters = nfa.get_alphabet();
        assert!(letters.contains(&"a"));
        assert!(letters.contains(&"b"));
        assert!(letters.len() == 2);
    }

    #[test]
    fn a_b_star() {
        let mut nfa = Nfa::from_size(2);
        nfa.add_transition_by_index1(0, 1, 'a');
        nfa.add_transition_by_index1(1, 0, 'b');
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
        assert_eq!(nfa.get_alphabet(), ["a", "b"]);

        let mut succ_a_0 = nfa.get_support("a").get_successors(0);
        succ_a_0.sort();
        assert_eq!(succ_a_0, vec![0, 1]);
    }
}
