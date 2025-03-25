use crate::coef::C0;
use crate::coef::OMEGA;
use crate::flow;
use crate::graph::Graph;
use crate::nfa;
use crate::nfa::Nfa;
use crate::semigroup;
use crate::sheep::Sheep;
use crate::solution::Solution;
use crate::strategy::Strategy;
use log::{debug, warn};
use std::collections::HashMap;
use std::collections::HashSet;

pub fn solve(original_nfa: &nfa::Nfa) -> Solution {
    let complete_nfa;
    let mut nb_states_added = 0;
    let mut nb_transitions_added = 0;
    let nfa = match original_nfa.is_complete() {
        true => original_nfa,
        false => {
            let nb_states_before = original_nfa.nb_states();
            let nb_transitions_before = original_nfa.nb_transitions();
            complete_nfa = nfa::Nfa::turn_into_complete_nfa(original_nfa).unwrap();
            nb_states_added = complete_nfa.nb_states() - nb_states_before;
            nb_transitions_added = complete_nfa.nb_transitions() - nb_transitions_before;
            warn!(
                "The NFA was not complete. It was turned into the following complete NFA {}.",
                complete_nfa
            );
            &complete_nfa
        }
    };

    let dim = nfa.nb_states();
    let source = get_omega_sheep(
        dim,
        &nfa.initial_states().iter().cloned().collect::<Vec<_>>(),
    );
    let final_states = nfa.final_states();
    let final_ideal = get_omega_sheep(dim, &final_states);

    let edges = get_edges(nfa);
    let mut strategy = Strategy::get_maximal_strategy(dim, &nfa.get_alphabet());
    let mut result = true;

    while result {
        //convert strategy to flows
        let action_flows = compute_action_flows(&strategy, &edges);
        debug!("\nAction flows:\n{}", flows_to_string(&action_flows));
        let semigroup = semigroup::FlowSemigroup::compute(&action_flows);
        debug!("Semigroup:\n{}", semigroup);
        let mut winning_ideal = semigroup.get_path_problem_solution(&final_states);
        winning_ideal.insert(&final_ideal);
        debug!("Winning ideal for the path problem:\n{}", winning_ideal);
        debug!("Strategy before restriction:\n{}", strategy);
        let changed = strategy.restrict_to(winning_ideal, &edges);
        debug!("Strategy after restriction:\n{}", strategy);
        if !changed {
            break;
        }
        result = strategy.is_defined_on(&source);
    }
    Solution {
        nfa: nfa.clone(),
        nb_states_added,
        nb_transitions_added,
        result,
        maximal_winning_strategy: strategy,
    }
}

fn get_omega_sheep(dim: usize, states: &[usize]) -> Sheep {
    let mut sheep = Sheep::new(dim, C0);
    for state in states {
        sheep.set(*state, OMEGA);
    }
    sheep
}

fn get_edges(nfa: &Nfa) -> HashMap<nfa::Letter, Graph> {
    if !nfa.is_complete() {
        panic!("The NFA is not complete");
    }
    nfa.get_alphabet()
        .iter()
        .map(|action| (action.to_string(), nfa.get_support(action)))
        .collect()
}

fn compute_action_flows(
    strategy: &Strategy,
    edges: &HashMap<nfa::Letter, Graph>,
) -> HashSet<flow::Flow> {
    let mut action_flows = HashSet::new();
    for (action, ideal) in strategy.iter() {
        let edges_for_action = edges.get(action).unwrap();
        for sheep in ideal.sheeps() {
            let flows = flow::Flow::from_domain_and_edges(sheep, edges_for_action);
            for flow in flows {
                action_flows.insert(flow);
            }
        }
    }
    action_flows
}

fn flows_to_string(flows: &HashSet<flow::Flow>) -> String {
    let mut vec: Vec<String> = flows.iter().map(|x| x.to_string()).collect();
    vec.sort();
    vec.join("\r\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::Flow;
    use crate::nfa::Nfa;
    use crate::strategy::Strategy;

    //test compute_action_flows
    #[test]
    fn test_nfa_1() {
        let mut nfa = Nfa::from_size(2);
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(1);
        nfa.add_transition_by_index(0, 1, 'a');
        nfa.add_transition_by_index(1, 1, 'a');
        let strategy = Strategy::get_maximal_strategy(2, &["a"]);
        let edges = get_edges(&nfa);
        let action_flows = compute_action_flows(&strategy, &edges);
        //a single action flow
        let flow: flow::Flow = Flow::from_entries(2, &[C0, OMEGA, C0, OMEGA]);
        assert_eq!(action_flows, HashSet::from([flow]));

        let edges = get_edges(&nfa);
        assert_eq!(edges, {
            let mut map = HashMap::new();
            map.insert("a".to_string(), nfa.get_support("a"));
            map
        });
    }

    #[test]
    fn test_nfa_2() {
        let mut nfa = Nfa::from_size(2);
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(1);
        nfa.add_transition_by_index(0, 0, 'b');
        nfa.add_transition_by_index(0, 1, 'a');
        nfa.add_transition_by_index(1, 0, 'b');
        nfa.add_transition_by_index(1, 1, 'a');
        nfa.add_transition_by_index(1, 1, 'b');
        let strategy = Strategy::get_maximal_strategy(2, &["a", "b"]);
        let edges = get_edges(&nfa);
        let computed = compute_action_flows(&strategy, &edges);
        //a single action flow
        assert_eq!(
            computed,
            HashSet::from([
                Flow::from_entries(2, &[C0, OMEGA, C0, OMEGA]),
                Flow::from_entries(2, &[OMEGA, C0, OMEGA, OMEGA]),
            ])
        );
        let edges = get_edges(&nfa);
        assert_eq!(edges.len(), 2);
        assert_eq!(edges.get("a").unwrap(), &nfa.get_support(&"a"));
        assert_eq!(edges.get("b").unwrap(), &nfa.get_support(&"b"));
    }

    #[test]
    fn test_solve_positive_mono_letter() {
        let mut nfa = Nfa::from_size(2);
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(1);
        nfa.add_transition_by_index(0, 0, 'a');
        nfa.add_transition_by_index(0, 1, 'a');
        nfa.add_transition_by_index(1, 1, 'a');
        let solution = solve(&nfa);
        assert_eq!(solution.result, true);
        assert_eq!(
            solution.maximal_winning_strategy,
            Strategy::get_maximal_strategy(2, &["a"])
        );
    }

    #[test]
    fn test_solve_is_completing_nfa() {
        let nb_states = 3;
        let mut nfa = Nfa::from_size(nb_states);
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(2);
        nfa.add_transition_by_index(0, 1, 'a');
        nfa.add_transition_by_index(0, 2, 'a');
        nfa.add_transition_by_index(1, 2, 'a');
        assert!(!nfa.is_complete());
        let solution = solve(&nfa);
        assert_eq!(
            solution.maximal_winning_strategy.dim().unwrap(),
            nb_states + 1
        );
    }

    #[test]
    fn test_solve_negative_mono_letter() {
        let mut nfa = Nfa::from_size(3);
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(2);
        nfa.add_transition_by_index(0, 1, 'a');
        nfa.add_transition_by_index(1, 1, 'a');
        nfa.add_transition_by_index(0, 2, 'a');
        nfa.add_transition_by_index(2, 2, 'a');
        let solution = solve(&nfa);
        print!("{}", solution);
        assert_eq!(solution.result, false);
    }

    pub(crate) fn get_nfa(name: &str) -> Nfa {
        match name {
            "((a#b){a,b})#" => {
                let mut nfa = Nfa::from_states(&["0", "1", "2", "3", "4", "5"]);
                nfa.add_initial("0");
                nfa.add_final("4");
                nfa.add_transition("0", "0", "a");
                nfa.add_transition("0", "1", "a");
                nfa.add_transition("1", "0", "a");
                nfa.add_transition("1", "1", "a");
                nfa.add_transition("4", "4", "a");
                nfa.add_transition("5", "5", "a");

                nfa.add_transition("0", "0", "b");
                nfa.add_transition("4", "4", "b");
                nfa.add_transition("5", "5", "b");

                nfa.add_transition("1", "2", "b");
                nfa.add_transition("1", "3", "b");

                nfa.add_transition("2", "4", "a");
                nfa.add_transition("2", "5", "b");
                nfa.add_transition("3", "4", "b");
                nfa.add_transition("3", "5", "a");
                nfa.add_transition("1", "0", "a");
                nfa.add_transition("1", "1", "a");
                nfa.add_transition("4", "4", "a");
                nfa.add_transition("5", "5", "a");

                nfa.add_transition("0", "0", "b");
                nfa.add_transition("4", "4", "b");
                nfa.add_transition("5", "5", "b");

                nfa.add_transition("1", "2", "b");
                nfa.add_transition("1", "3", "b");

                nfa.add_transition("2", "4", "a");
                nfa.add_transition("2", "5", "b");
                nfa.add_transition("3", "4", "b");
                nfa.add_transition("3", "5", "a");
                nfa
            }
            _ => panic!("Unknown NFA"),
        }
    }

    #[test]
    fn test_solve_positive_two_letters() {
        let nfa = get_nfa("((a#b){a,b})#");

        let solution = solve(&nfa);
        print!("{}", solution);
        assert_eq!(solution.result, true);
    }
}
