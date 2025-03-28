use crate::coef::Coef;
use crate::flow::Flow;
use crate::ideal;
use crate::nfa;
use cached::proc_macro::cached;
use itertools::Itertools;
use log::debug;
use rayon::prelude::*;
use std::collections::HashSet; // for distinct method
use std::collections::VecDeque;
use std::fmt;

pub struct FlowSemigroup {
    //invariant: all flows have the same dimension
    flows: HashSet<Flow>,
}

impl FlowSemigroup {
    pub fn new() -> Self {
        FlowSemigroup {
            flows: HashSet::new(),
        }
    }

    pub fn compute(flows: &HashSet<Flow>, maximal_finite_coordinate: u16) -> Self {
        let mut semigroup = FlowSemigroup::new();
        for flow in flows.iter() {
            semigroup.flows.insert(flow.clone());
        }
        semigroup.close_by_product_and_iteration(maximal_finite_coordinate);
        semigroup
    }

    #[allow(dead_code)]
    pub fn contains(&self, flow: &Flow) -> bool {
        Self::is_covered(flow, &self.flows)
    }

    pub fn get_path_problem_solution(&self, target: &[nfa::State]) -> ideal::Ideal {
        ideal::Ideal::from_vec(
            &self
                .flows
                .iter()
                .map(|flow| flow.pre_image(target))
                .collect::<Vec<_>>(),
        )
    }

    ///non-deterministic product
    fn get_products(
        left: &Flow,
        right: &Flow,
        _maximal_finite_coordinate: u16,
    ) -> rayon::iter::Once<Flow> {
        rayon::iter::once(left.clone() * right.clone())
    }
    /*
    fn get_products(left: &Flow, right: &Flow, maximal_finite_coordinate: u16) -> Vec<Flow> {
        debug_assert_eq!(left.nb_rows , right.nb_rows);
        let dim = left.nb_rows;
        let transports = (0..dim)
            .map(|k| {
                let left_edges = left.edges_to(k);
                let right_edges = right.edges_from(k);
                let left_coefs = left_edges.iter().map(|&(_, c)| c).collect::<Vec<_>>();
                let right_coefs = right_edges.iter().map(|&(_, c)| c).collect::<Vec<_>>();
                let left_indices = left_edges.iter().map(|&(i, _)| i).collect::<Vec<_>>();
                let right_indices = right_edges.iter().map(|&(j, _)| j).collect::<Vec<_>>();
                //cached
                get_transports(left_coefs, right_coefs, maximal_finite_coordinate)
                    .into_iter()
                    .map(|transport| (left_indices.clone(), transport, right_indices.clone()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        transports
            .iter()
            .multi_cartesian_product()
            .map(|transports| Flow::compose(left, transports, right))
            .collect::<Vec<_>>()
        //
    }*/

    fn close_by_product_and_iteration(&mut self, maximal_finite_coordinate: u16) {
        let mut to_process: VecDeque<Flow> = self.flows.iter().cloned().collect();
        let mut processed = HashSet::<Flow>::new();
        while !to_process.is_empty() {
            let flow = to_process.pop_front().unwrap();
            debug!(
                "\nClose_by_product_and_iteration processing flow\n{}\n",
                flow
            );
            if Self::is_covered(&flow, &processed) {
                //debug!("Skipped inqueue\n{}", flow);
                continue;
            }
            processed.insert(flow.clone());

            let iteration = flow.iteration();
            if !Self::is_covered(&iteration, &self.flows) {
                debug!("\n\nAdded iteration\n{}", iteration);
                self.flows.insert(iteration.clone());
                to_process.push_back(iteration);
            } else {
                //debug!("\n\nSkipped iteration\n{}", iteration);
            }
            {
                let right_products = self
                    .flows
                    .par_iter()
                    .flat_map(|other| Self::get_products(&flow, other, maximal_finite_coordinate));
                let left_products = self
                    .flows
                    .par_iter()
                    .flat_map(|other| Self::get_products(other, &flow, maximal_finite_coordinate));
                let products: HashSet<Flow> = left_products.chain(right_products).collect();
                for product in products {
                    if !Self::is_covered(&product, &self.flows) {
                        debug!("\n\nAdded product\n{}", product);
                        self.flows.insert(product.clone());
                        to_process.push_back(product);
                    } else {
                        //debug!("\n\nSkipped product\n{}", product);
                    }
                }
            }
            self.minimize();
        }
    }

    fn is_covered(flow: &Flow, others: &HashSet<Flow>) -> bool {
        /*debug!(
            "Checking whether\n{} is covered by\n{}\n",
            flow,
            others
                .iter()
                .map(Flow::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        );*/
        others.iter().any(|other| flow <= other)
    }

    fn minimize(&mut self) {
        debug!("Minimizing semigroup");
        let before = self.flows.len();
        //debug!("Before minimization\n{}", self);
        let mut to_remove = HashSet::<Flow>::new();
        for flow in self.flows.iter() {
            if to_remove.contains(flow) {
                continue;
            }
            if self.flows.iter().any(|other| flow < other) {
                to_remove.insert(flow.clone());
            }
        }
        for flow in to_remove.iter() {
            self.flows.remove(flow);
        }
        //debug!("After minimization\n{}", self);
        let after = self.flows.len();
        debug!(
            "Minimized semigroup from {} flows to {} flows",
            before, after
        );
    }
}

#[cached]
fn get_transports(
    left_edges: Vec<Coef>,
    right_edges: Vec<Coef>,
    maximal_finite_coordinate: u16,
) -> Vec<Flow> {
    //C = min(dim, sum ni, sum mi)
    let nb_rows = left_edges.len();
    let nb_cols = right_edges.len();
    let omega_left = left_edges
        .iter()
        .enumerate()
        .filter_map(|(i, x)| if *x == Coef::Omega { Some(i) } else { None })
        .collect::<Vec<_>>();
    let omega_right = right_edges
        .iter()
        .enumerate()
        .filter_map(|(j, x)| if *x == Coef::Omega { Some(j) } else { None })
        .collect::<Vec<_>>();
    let base_flow = Flow::from_entries(
        nb_rows,
        nb_cols,
        &(0..nb_rows * nb_cols)
            .map(|k| {
                let (i, j) = (k / nb_rows, k % nb_cols);
                if omega_left.contains(&i) && omega_right.contains(&j) {
                    Coef::Omega
                } else {
                    Coef::Value(0)
                }
            })
            .collect::<Vec<_>>(),
    );

    let nb_strays_left = left_edges
        .iter()
        .map(|&x| match x {
            Coef::Value(y) => y,
            Coef::Omega => maximal_finite_coordinate,
        })
        .collect::<Vec<_>>();

    let nb_strays_right = right_edges
        .iter()
        .map(|&x| match x {
            Coef::Value(y) => y,
            Coef::Omega => maximal_finite_coordinate,
        })
        .collect::<Vec<_>>();

    //extract non omega non zero edges
    let stray_edges = (0..nb_rows * nb_cols)
        .filter_map(|k| {
            let (i, j) = (k / nb_rows, k % nb_cols);
            match base_flow.get(&i, &j) {
                Coef::Omega => None,
                Coef::Value(0) => None,
                _ => Some((i, j)),
            }
        })
        .collect::<Vec<_>>();

    let mut flow_accumulator = Vec::<Flow>::new();
    get_transports_rec(
        &base_flow,
        &stray_edges,
        0,
        &mut nb_strays_left.clone(),
        &mut nb_strays_right.clone(),
        &mut flow_accumulator,
    );
    flow_accumulator
}

fn get_transports_rec(
    current_flow: &Flow,
    edges: &Vec<(usize, usize)>,
    current_edge: usize,
    nb_strays_left: &mut Vec<u16>,
    nb_strays_right: &mut Vec<u16>,
    flow_accumulator: &mut Vec<Flow>,
) {
    debug_assert!(current_edge < edges.len());
    debug_assert!(edges
        .iter()
        .skip(current_edge)
        .all(|(i, j)| { current_flow.get(i, j) == Coef::Value(0) }));
    let (left, right) = edges[current_edge];
    let strays_left = nb_strays_left[left];
    let strays_right = nb_strays_right[right];
    match (strays_left, strays_right) {
        (0, _) | (_, 0) => {
            flow_accumulator.push(current_flow.clone());
        }
        (nbl, nbr) => {
            let nb_max = std::cmp::min(nbl, nbr);
            if current_edge == edges.len() - 1 {
                //no other choice that put remainibng budget into current edge
                let mut new_flow = current_flow.clone();
                new_flow.set(&left, &right, Coef::Value(nb_max));
                flow_accumulator.push(new_flow);
            } else {
                for nb_here in 0..nb_max + 1 {
                    nb_strays_left[left] = nbl - nb_here;
                    nb_strays_right[right] = nbr - nb_here;
                    get_transports_rec(
                        current_flow,
                        edges,
                        current_edge + 1,
                        nb_strays_left,
                        nb_strays_right,
                        flow_accumulator,
                    );
                }
                nb_strays_left[left] = nbl;
                nb_strays_right[right] = nbr;
            }
        }
    }
}

impl fmt::Display for FlowSemigroup {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut flows = self
            .flows
            .iter()
            .map(|flow| flow.to_string())
            .collect::<Vec<_>>();
        flows.sort();
        write!(f, "{}", flows.join("\r\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coef::{C0, C1, OMEGA};
    use crate::sheep::Sheep;
    use Flow;

    #[test]
    fn test_flow_semigroup_compute1() {
        let dim = 2 as usize;
        let flowa = Flow::from_lines(&[&[OMEGA, C1], &[C0, OMEGA]]);
        let flows: HashSet<Flow> = [flowa].into();
        let semigroup = FlowSemigroup::compute(&flows, dim as u16);
        let flow_omega = Flow::from_entries(dim, dim, &[OMEGA, OMEGA, C0, OMEGA]);
        print!("\nsemigroup\n\n{}", semigroup);
        assert!(semigroup.flows.contains(&flow_omega));
    }

    #[test]
    fn test_flow_semigroup_compute2() {
        let dim = 3;
        let flowa = Flow::from_lines(&[&[OMEGA, OMEGA, C0], &[OMEGA, OMEGA, C1], &[C0, C0, OMEGA]]);
        let flowb = Flow::from_lines(&[&[OMEGA, C0, C0], &[C0, C1, C0], &[C0, C0, OMEGA]]);
        let flows: HashSet<Flow> = [flowa.clone(), flowb.clone()].into();
        let semigroup = FlowSemigroup::compute(&flows, dim);
        print!("\nsemigroup\n\n{}", semigroup);
        assert!(semigroup.contains(&flowa));
        assert!(semigroup.contains(&flowb));
    }

    #[test]
    fn test_flow_semigroup_compute3() {
        let dim = 3;
        let flowa = Flow::from_lines(&[&[OMEGA, C1, C0], &[OMEGA, C0, C1], &[C0, C0, OMEGA]]);
        let flowb = Flow::from_lines(&[&[OMEGA, C0, C0], &[C0, C1, C0], &[C0, C0, OMEGA]]);
        let flows: HashSet<Flow> = [flowa.clone(), flowb.clone()].into();
        let semigroup = FlowSemigroup::compute(&flows, dim);
        print!("\nsemigroup\n\n{}", semigroup);
        assert!(semigroup.contains(&flowa));
        assert!(semigroup.contains(&flowb));
    }

    #[test]
    fn test_path_problem() {
        let flow = Flow::from_lines(&[&[C0, C1, C1], &[C0, C0, C0], &[C0, C0, C0]]);
        let flows: HashSet<Flow> = [flow].into();
        let semigroup = FlowSemigroup::compute(&flows, 3);
        println!("semigroup\n\n{}", semigroup);
        let path_problem_solution = semigroup.get_path_problem_solution(&[1, 2]);
        println!("path_problem_solution\n{}", path_problem_solution);
        let expected = &Sheep::from_vec(vec![Coef::Value(2), C0, C0]);
        assert!(path_problem_solution.contains(expected));
    }

    #[test]
    fn test_path_problem2() {
        let dim = 5;
        let c2 = Coef::Value(2);
        let flow = Flow::from_lines(&[
            &[C0, C1, C1, C0, C0],
            &[C0, C0, C0, C1, C0],
            &[C0, C0, C0, C1, C0],
            &[C0, C0, C0, C0, c2],
            &[C0, C0, C0, C0, C0],
        ]);
        let flows: HashSet<Flow> = [flow].into();
        let semigroup = FlowSemigroup::compute(&flows, dim);
        println!("semigroup\n\n{}", semigroup);
        let path_problem_solution = semigroup.get_path_problem_solution(&[4]);
        println!("path_problem_solution\n{}", path_problem_solution);
        let expected = &Sheep::from_vec(vec![c2, C0, C0, C0, C0]);
        assert!(path_problem_solution.contains(expected));
    }
}
