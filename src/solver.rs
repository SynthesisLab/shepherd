use crate::coef::C0;
use crate::coef::OMEGA;
use crate::flow;
use crate::graph::Graph;
use crate::nfa;
use crate::semigroup;
use crate::sheep::Sheep;
use crate::solution::Solution;
use crate::strategy::Strategy;
use log::{debug, info};
use std::collections::HashMap;
use std::collections::HashSet;

pub fn solve(nfa: &nfa::Nfa) -> Solution {
    let dim = nfa.nb_states();
    let source = get_omega_sheep(
        dim,
        &nfa.initial_states().iter().cloned().collect::<Vec<_>>(),
    );
    let final_states = nfa.final_states();
    let final_ideal = get_omega_sheep(dim, &final_states);

    let edges = nfa.get_edges();
    let mut strategy = Strategy::get_maximal_strategy(dim, &nfa.get_alphabet());
    let mut result = true;
    let mut step = 1;
    while result {
        //convert strategy to flows
        info!(
            "Step {} states\n{}\nstrategy\n{}",
            step,
            nfa.states_str(),
            strategy
        );
        step += 1;
        let action_flows = compute_action_flows(&strategy, &edges);
        debug!("\nAction flows:\n{}", flows_to_string(&action_flows));
        println!("Computing semigroup");
        let semigroup = semigroup::FlowSemigroup::compute(&action_flows, dim as u16);
        debug!("Semigroup:\n{}", semigroup);
        println!("Computing winning ideal");
        let mut winning_ideal = semigroup.get_path_problem_solution(&final_states);
        winning_ideal.insert(&final_ideal);
        //non-omega stay below dim
        let dim16: u16 = dim.try_into().unwrap();
        debug!(
            "Winning ideal for the path problem before rounding down\n{}",
            winning_ideal
        );
        winning_ideal.round_down(dim16, dim); //backed by the small constants theorem
        debug!(
            "Winning ideal for the path problem before minimize\n{}",
            winning_ideal
        );
        winning_ideal.minimize();
        debug!("Winning ideal for the path problem:\n{}", winning_ideal);
        println!("Restricting strategy");
        let changed = strategy.restrict_to(winning_ideal, &edges);
        debug!("Strategy after restriction:\n{}", strategy);
        if !changed {
            break;
        }
        result = strategy.is_defined_on(&source);
    }
    Solution {
        nfa: nfa.clone(),
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
    vec.join("\n")
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
        let dim = 2;
        let mut nfa = Nfa::from_size(dim);
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(1);
        nfa.add_transition_by_index1(0, 1, 'a');
        nfa.add_transition_by_index1(1, 1, 'a');
        let strategy = Strategy::get_maximal_strategy(2, &["a"]);
        let edges = nfa.get_edges();
        let action_flows = compute_action_flows(&strategy, &edges);
        //a single action flow
        let flow: flow::Flow = Flow::from_entries(dim, dim, &[C0, OMEGA, C0, OMEGA]);
        assert_eq!(action_flows, HashSet::from([flow]));

        let edges = nfa.get_edges();
        let grapha = edges
            .get("a")
            .unwrap()
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        let grapha_from_nfa = nfa.get_support("a").iter().cloned().collect::<HashSet<_>>();
        assert_eq!(grapha, grapha_from_nfa);
    }

    #[test]
    fn test_nfa_2() {
        let dim = 2;
        let mut nfa = Nfa::from_size(dim);
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(1);
        nfa.add_transition_by_index1(0, 0, 'b');
        nfa.add_transition_by_index1(0, 1, 'a');
        nfa.add_transition_by_index1(1, 0, 'b');
        nfa.add_transition_by_index1(1, 1, 'a');
        nfa.add_transition_by_index1(1, 1, 'b');
        let strategy = Strategy::get_maximal_strategy(2, &["a", "b"]);
        let edges = nfa.get_edges();
        let computed = compute_action_flows(&strategy, &edges);
        //a single action flow
        assert_eq!(
            computed,
            HashSet::from([
                Flow::from_entries(dim, dim, &[C0, OMEGA, C0, OMEGA]),
                Flow::from_entries(dim, dim, &[OMEGA, C0, OMEGA, OMEGA]),
            ])
        );
        let edges = nfa.get_edges();
        assert_eq!(edges.len(), 2);
        let grapha = edges
            .get("a")
            .unwrap()
            .iter()
            .cloned()
            .collect::<HashSet<_>>();

        let graphb = edges
            .get("b")
            .unwrap()
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        let grapha_from_nfa = nfa.get_support("a").iter().cloned().collect::<HashSet<_>>();
        let graphb_from_nfa = nfa.get_support("b").iter().cloned().collect::<HashSet<_>>();
        assert_eq!(grapha, grapha_from_nfa);
        assert_eq!(graphb, graphb_from_nfa);
    }

    #[test]
    fn test_solve_positive_mono_letter() {
        let mut nfa = Nfa::from_size(2);
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(1);
        nfa.add_transition_by_index1(0, 0, 'a');
        nfa.add_transition_by_index1(0, 1, 'a');
        nfa.add_transition_by_index1(1, 1, 'a');
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
        nfa.add_transition_by_index1(0, 1, 'a');
        nfa.add_transition_by_index1(0, 2, 'a');
        nfa.add_transition_by_index1(1, 2, 'a');
        let solution = solve(&nfa);
        assert!(!solution.result);
    }

    #[test]
    fn test_solve_negative_mono_letter() {
        let mut nfa = Nfa::from_size(3);
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(2);
        nfa.add_transition_by_index1(0, 1, 'a');
        nfa.add_transition_by_index1(1, 1, 'a');
        nfa.add_transition_by_index1(0, 2, 'a');
        nfa.add_transition_by_index1(2, 2, 'a');
        let solution = solve(&nfa);
        print!("{}", solution);
        assert_eq!(solution.result, false);
    }

    #[test]
    fn test_solve_positive_two_letters() {
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

        //enter the <= 1 bottleneck
        nfa.add_transition("1", "2", "b");
        nfa.add_transition("1", "3", "b");

        //chhose sides
        nfa.add_transition("2", "4", "a");
        nfa.add_transition("2", "5", "b");
        nfa.add_transition("3", "4", "b");
        nfa.add_transition("3", "5", "a");

        let solution = solve(&nfa);
        print!("{}", solution);
        assert_eq!(solution.result, true);
    }
}
