use crate::coef::{coef, C0, OMEGA};
use crate::flow;
use crate::graph::Graph;
use crate::ideal::Ideal;
use crate::nfa;
use crate::semigroup::{self, FlowSemigroup};
use crate::solution::Solution;
use crate::strategy::Strategy;
use clap::ValueEnum;
use log::{debug, info};
use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Debug, Clone, ValueEnum)]
pub enum SolverOutput {
    YesNo,
    Strategy,
}

pub fn solve(nfa: &nfa::Nfa, output: &SolverOutput) -> Solution {
    let dim = nfa.nb_states();
    let source = get_omega_ideal(
        dim,
        &nfa.initial_states().iter().cloned().collect::<Vec<_>>(),
    );
    let final_states = nfa.final_states();
    let edges = nfa.get_edges();
    let letters = nfa.get_alphabet();
    let (strategy, semigroup) = match output {
        SolverOutput::Strategy => {
            compute_maximal_winning_strategy(dim, &final_states, edges, &letters)
        }
        SolverOutput::YesNo => {
            compute_control_problem_solution(dim, &source, &final_states, edges, &letters)
        }
    };
    let is_controllable = strategy.is_defined_on(&source);
    Solution {
        nfa: nfa.clone(),
        is_controllable,
        winning_strategy: strategy,
        semigroup,
    }
}

fn compute_maximal_winning_strategy(
    dim: usize,
    final_states: &[usize],
    edges: HashMap<String, Graph>,
    letters: &[&str],
) -> (Strategy, FlowSemigroup) {
    let maximal_finite_value = dim as coef;

    let mut strategy = Strategy::get_maximal_strategy(dim, letters);

    let mut step = 1;
    loop {
        //convert strategy to flows
        info!("Computing the maximal winning strategy step {}", step);
        step += 1;

        let (changed, semigroup) = update_strategy(
            dim,
            &mut strategy,
            final_states,
            &edges,
            maximal_finite_value,
        );

        if !changed {
            return (strategy, semigroup);
        }
    }
}

fn compute_control_problem_solution(
    dim: usize,
    source: &Ideal,
    final_states: &[usize],
    edges: HashMap<String, Graph>,
    letters: &[&str],
) -> (Strategy, FlowSemigroup) {
    let mut strategy = Strategy::get_maximal_strategy(dim, letters);
    let mut semigroup = FlowSemigroup::new();

    for maximal_finite_value in 1..dim as coef {
        let mut step = 1;
        loop {
            //convert strategy to flows
            info!(
                "Looking for a winning strategy using maximal finite_value {} step {}",
                maximal_finite_value, step
            );
            step += 1;

            let (changed, new_semigroup) = update_strategy(
                dim,
                &mut strategy,
                final_states,
                &edges,
                maximal_finite_value,
            );
            semigroup = new_semigroup;
            let result = strategy.is_defined_on(source);

            if !changed || !result {
                break;
            }
        }
        if strategy.is_defined_on(source) {
            break;
        }
    }
    (strategy, semigroup)
}

fn update_strategy(
    dim: usize,
    strategy: &mut Strategy,
    final_states: &[usize],
    edges: &HashMap<String, Graph>,
    maximal_finite_value: u8,
) -> (bool, FlowSemigroup) {
    let final_ideal = get_omega_ideal(dim, final_states);
    let action_flows = compute_action_flows(strategy, edges);
    debug!("\nAction flows:\n{}", flows_to_string(&action_flows));
    debug!(
        "Computing semigroup with maximal_finite_value {}",
        maximal_finite_value
    );
    let semigroup = semigroup::FlowSemigroup::compute(&action_flows, maximal_finite_value);
    debug!("Semigroup:\n{}", semigroup);
    debug!("Computing winning set");
    let mut winning_downset = semigroup.get_path_problem_solution(final_states);
    winning_downset.insert(&final_ideal);
    winning_downset.round_down(maximal_finite_value, dim);
    winning_downset.minimize();
    debug!("Winning set for the path problem:\n{}", winning_downset);
    debug!("Restricting strategy");
    let changed = strategy.restrict_to(winning_downset, edges, maximal_finite_value);
    debug!("Strategy after restriction:\n{}", strategy);
    (changed, semigroup)
}

fn get_omega_ideal(dim: usize, states: &[usize]) -> Ideal {
    let mut ideal = Ideal::new(dim, C0);
    for state in states {
        ideal.set(*state, OMEGA);
    }
    ideal
}

fn compute_action_flows(
    strategy: &Strategy,
    edges: &HashMap<nfa::Letter, Graph>,
) -> HashSet<flow::Flow> {
    let mut action_flows = HashSet::new();
    for (action, downset) in strategy.iter() {
        let edges_for_action = edges.get(action).unwrap();
        for ideal in downset.ideals() {
            let flows = flow::Flow::from_domain_and_edges(ideal, edges_for_action);
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
    fn test_solve_mono_letter_positive() {
        let mut nfa = Nfa::from_size(2);
        nfa.add_initial_by_index(0);
        nfa.add_final_by_index(1);
        nfa.add_transition_by_index1(0, 0, 'a');
        nfa.add_transition_by_index1(0, 1, 'a');
        nfa.add_transition_by_index1(1, 1, 'a');
        let solution = solve(&nfa, &SolverOutput::Strategy);
        assert_eq!(
            solution.winning_strategy,
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
        let solution = solve(&nfa, &SolverOutput::Strategy);
        assert!(!solution.is_controllable);
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
        let solution = solve(&nfa, &SolverOutput::Strategy);
        print!("{}", solution);
        assert!(!solution.is_controllable);
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

        let solution = solve(&nfa, &SolverOutput::YesNo);
        print!("{}", solution);
        assert!(solution.is_controllable);
    }
}
