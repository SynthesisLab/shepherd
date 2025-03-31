use crate::coef::Coef;
use crate::coef::OMEGA;
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
use std::io::{self, Write};

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

    ///deterministic product, fails for constants >= 2
    #[allow(dead_code)]
    fn get_products_buggy(
        left: &Flow,
        right: &Flow,
        _maximal_finite_coordinate: u16,
    ) -> rayon::iter::Once<Flow> {
        rayon::iter::once(left.clone() * right.clone())
    }

    ///non-deterministic product
    fn get_products(left: &Flow, right: &Flow, maximal_finite_coordinate: u16) -> HashSet<Flow> {
        debug_assert_eq!(left.nb_rows, right.nb_rows);
        //debug!("get_products\nleft\n{}\nright\n{}", left, right);
        let dim = left.nb_rows;
        let omega_part = Flow::get_omega_entries(left, right);
        //debug!("omega part\n{}\n", omega_part);
        let left = &mut left.clone();
        let right = &mut right.clone();

        let mut result = HashSet::<Flow>::new();
        Self::get_products_rec(
            dim,
            left,
            right,
            maximal_finite_coordinate,
            0,
            omega_part,
            &mut result,
        );
        result
    }

    fn get_products_rec(
        dim: usize,
        left: &Flow,
        right: &Flow,
        maximal_finite_coordinate: u16,
        k: usize,
        current_flow: Flow,
        flow_accumulator: &mut HashSet<Flow>,
    ) {
        debug_assert!(k < dim);
        /*debug!(
            "k={}\nleft\n{}\nright\n{}\ncurrent_flow\n{}\n\n",
            k, left, right, current_flow
        );*/
        let left_edges = left.edges_to(k);
        let right_edges = right.edges_from(k);
        debug_assert!(k < dim);
        if left_edges.is_empty() || right_edges.is_empty() {
            if k + 1 < dim {
                Self::get_products_rec(
                    dim,
                    &left,
                    &right,
                    maximal_finite_coordinate,
                    k + 1,
                    current_flow,
                    flow_accumulator,
                );
            } else {
                flow_accumulator.insert(current_flow);
            }
            return;
        }
        /*
        println!(
            "left_edges to {}\n\t{:?}\nright_edges from {}\n\t{:?}\n",
            k, left_edges, k, right_edges
        );*/

        //todo compute left stuff once at a time with a single into_iter
        let left_coefs = left_edges.iter().map(|&(_, c)| c).collect::<Vec<_>>();
        let right_coefs = right_edges.iter().map(|&(_, c)| c).collect::<Vec<_>>();
        /*
        debug!(
            "left_coefs\n\t{:?}\nright_coefs\n\t{:?}\n",
            left_coefs, right_coefs
        );*/
        let left_indices = left_edges.into_iter().map(|(i, _)| i).collect::<Vec<_>>();
        //todo compute right stuff once at a time with a single into_iter
        let right_indices = right_edges.into_iter().map(|(j, _)| j).collect::<Vec<_>>();
        /*
        debug!(
            "left_indices\n\t{:?}\nright_indices\n\t{:?}\n",
            left_indices, right_indices
        );*/
        let all_indices = left_indices
            .iter()
            .enumerate()
            .cartesian_product(right_indices.iter().enumerate())
            .collect::<Vec<_>>();
        //
        debug_assert!(!left_coefs.is_empty());
        debug_assert!(!right_coefs.is_empty());

        let transports = get_transports(left_coefs, right_coefs, maximal_finite_coordinate);
        for t in transports {
            let mut left = left.clone();
            let mut right = right.clone();
            let mut current_flow = current_flow.clone();
            //debug!("k={}\ntransport\n{}\n", k, t);
            //debug!("current_flow before\n{}\n", current_flow);
            for ((subi, reali), (subj, realj)) in &all_indices {
                let cf = current_flow.get(reali, realj);
                let tij: Coef = t.get(&subi, &subj);
                let newcf = cf + tij;
                if newcf > Coef::Value(maximal_finite_coordinate) {
                    continue;
                } else {
                    current_flow.set(reali, realj, newcf);
                    let cl = left.get(reali, &k);
                    let cr = right.get(&k, realj);
                    debug_assert!(cl >= tij, "{} {}", cl, tij);
                    debug_assert!(cr >= tij, "{} {}", cr, tij);
                    left.set(reali, &k, cl - tij);
                    right.set(&k, realj, cr - tij);
                }
            }
            //debug!("current_flow after\n{}\n", current_flow);
            let k1 = k + 1;
            if k1 >= dim {
                flow_accumulator.insert(current_flow);
            } else {
                Self::get_products_rec(
                    dim,
                    &left,
                    &right,
                    maximal_finite_coordinate,
                    k1,
                    current_flow,
                    flow_accumulator,
                );
            }
        }
        /*
        println!(
            "transports for index {} {}",
            k,
            transports.iter().map(|t| t.to_string()).join("\n")
        );*/
    }

    fn close_by_product_and_iteration(&mut self, maximal_finite_coordinate: u16) {
        let mut to_process: VecDeque<Flow> = self.flows.iter().cloned().collect();
        let mut processed = HashSet::<Flow>::new();
        while !to_process.is_empty() {
            let flow = to_process.pop_front().unwrap();
            print!(".");
            io::stdout().flush().unwrap();
            println!(
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
                debug!("Products {:?}\n", products);
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
) -> HashSet<Flow> {
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
    let omega_flow = Flow::from_entries(
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
    let stray_edges = left_edges
        .iter()
        .enumerate()
        .cartesian_product(right_edges.iter().enumerate())
        .filter_map(|((i, &ci), (j, &cj))| match (ci, cj) {
            (OMEGA, OMEGA) | (Coef::Value(0), _) | (_, Coef::Value(0)) => None,
            _ => Some((i, j)),
        })
        .collect::<Vec<_>>();

    if stray_edges.is_empty() {
        HashSet::from([omega_flow])
    } else {
        let mut flow_accumulator = HashSet::<Flow>::new();
        get_transports_rec(
            &mut omega_flow.clone(),
            &stray_edges,
            0,
            &mut nb_strays_left.clone(),
            &mut nb_strays_right.clone(),
            &mut flow_accumulator,
        );
        flow_accumulator
    }
}

fn get_transports_rec(
    current_flow: &mut Flow,
    edges: &Vec<(usize, usize)>,
    current_edge: usize,
    nb_strays_left: &mut Vec<u16>,
    nb_strays_right: &mut Vec<u16>,
    flow_accumulator: &mut HashSet<Flow>,
) {
    debug_assert!(current_edge < edges.len());
    debug_assert!(
        edges.iter().skip(current_edge).all(|(i, j)| {
            current_flow.get(i, j) == Coef::Value(0) || current_flow.get(i, j) == OMEGA
        }),
        "current_flow\n{}",
        current_flow
    );
    let (left, right) = edges[current_edge];
    /*println!(
        "\n\ncurrent_flow\n{}nb_strays ({:?},{:?}) edge ({},{}) among {:?}",
        current_flow, &nb_strays_left, &nb_strays_right, left, right, edges
    );*/
    let strays_left = nb_strays_left[left];
    let strays_right = nb_strays_right[right];
    let nb_max = std::cmp::min(strays_left, strays_right);
    if current_edge == edges.len() - 1 {
        //no other choice that put remainibng budget into current edge
        let mut new_flow = current_flow.clone();
        new_flow.set(&left, &right, Coef::Value(nb_max));
        //println!("flow\n{} ", new_flow);
        flow_accumulator.insert(new_flow);
    } else {
        let (nbl, nbr) = (strays_left, strays_right);
        /*
        let is_left_over = 1 + right == nb_strays_right.len();
        let is_right_over = 1 + left == nb_strays_left.len();
        let nb_min = if is_left_over || is_right_over {
            nb_max //use all the capacity available
        } else {
            0
        };*/
        for nb_here in 0..nb_max + 1 {
            nb_strays_left[left] = nbl - nb_here;
            nb_strays_right[right] = nbr - nb_here;
            current_flow.set(&left, &right, Coef::Value(nb_here));
            get_transports_rec(
                current_flow,
                edges,
                current_edge + 1,
                nb_strays_left,
                nb_strays_right,
                flow_accumulator,
            );
        }
        //RAZ
        current_flow.set(&left, &right, Coef::Value(0));
        nb_strays_left[left] = nbl;
        nb_strays_right[right] = nbr;
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
            &[C0, C1, C1, C0, C0], //0 -- 1 --> {1,2}
            &[C0, C0, C0, C1, C0], //1 -- 1 --> 3
            &[C0, C0, C0, C1, C0], //2 -- 1 --> 3
            &[C0, C0, C0, C0, c2], //3 -- 2 --> 4
            &[C0, C0, C0, C0, C0], //
        ]);
        let flows: HashSet<Flow> = [flow].into();
        let semigroup = FlowSemigroup::compute(&flows, dim);
        println!("semigroup\n\n{}", semigroup);
        let path_problem_solution = semigroup.get_path_problem_solution(&[4]);
        println!("path_problem_solution\n{}", path_problem_solution);
        let expected = &Sheep::from_vec(vec![c2, C0, C0, C0, C0]);
        assert!(path_problem_solution.contains(expected));
    }

    //get_transports(left_coefs, right_coefs, maximal_finite_coordinate)
    #[test]
    fn get_transports_test1() {
        let dim = 2;
        let left = vec![C1, C0];
        let right = vec![C0, C1];
        let transports = get_transports(left, right, dim as u16);
        //println!("transports {:?}", transports);
        assert_eq!(transports.len(), 1);
        assert_eq!(
            transports.iter().next().unwrap(),
            &Flow::from_lines(&[&[C0, C1], &[C0, C0]])
        );
    }

    #[test]
    fn get_transports_test2() {
        let dim = 2;
        let c2 = Coef::Value(2);
        let c4 = Coef::Value(4);
        let left = vec![c2, c2];
        let right = vec![c4, c4];
        let transports = get_transports(left, right, dim as u16);
        /*println!(
            "transports\n{}",
            transports.iter().map(|t| t.to_string()).join("\n\n")
        );*/
        assert!(transports.contains(&Flow::from_lines(&[&[c2, C0], &[c2, C0]])));
        assert!(transports.contains(&Flow::from_lines(&[&[C0, c2], &[c2, C0]])));
        assert!(transports.contains(&Flow::from_lines(&[&[c2, C0], &[C0, c2]])));
        assert!(transports.contains(&Flow::from_lines(&[&[C0, c2], &[C0, c2]])));
        assert!(transports.contains(&Flow::from_lines(&[&[C1, C1], &[c2, C0]])));
        assert!(transports.contains(&Flow::from_lines(&[&[C1, C1], &[c2, C0]])));
        assert!(transports.contains(&Flow::from_lines(&[&[C1, C1], &[C0, c2]])));
        assert!(transports.contains(&Flow::from_lines(&[&[C1, C1], &[C0, c2]])));
        assert!(transports.contains(&Flow::from_lines(&[&[c2, C0], &[C1, C1]])));
        assert!(transports.contains(&Flow::from_lines(&[&[C0, c2], &[C1, C1]])));
        assert!(transports.contains(&Flow::from_lines(&[&[c2, C0], &[C1, C1]])));
        assert!(transports.contains(&Flow::from_lines(&[&[C0, c2], &[C1, C1]])));
        assert!(transports.contains(&Flow::from_lines(&[&[C1, C1], &[C1, C1]])));
        //assert_eq!(transports.iter().next(), None);
    }

    #[test]
    fn get_transports_test3() {
        let dim = 2;
        let left = vec![];
        let right = vec![];
        let transports = get_transports(left, right, dim as u16);
        assert!(!transports.is_empty());
        let t = transports.iter().next().unwrap();
        assert!(t.nb_rows == 0);
        assert!(t.nb_cols == 0);
    }

    #[test]
    fn get_products_test1() {
        let dim = 2;
        let left = Flow::from_lines(&[&[C1, C0], &[C0, C1]]);
        let right = Flow::from_lines(&[&[C1, C0], &[C0, C1]]);
        let products = FlowSemigroup::get_products(&left, &right, dim as u16);
        assert_eq!(products.len(), 1);
        assert_eq!(
            products.iter().next().unwrap(),
            &Flow::from_lines(&[&[C1, C0], &[C0, C1]])
        );
    }

    #[test]
    fn get_products_test2() {
        let dim = 4;
        let c2 = Coef::Value(2);
        let c4 = Coef::Value(4);
        let left = Flow::from_lines(&[
            &[c2, c2, C0, C0],
            &[c2, c2, C0, C0],
            &[C0, C0, C0, C0],
            &[C0, C0, C0, C0],
        ]);
        let right = Flow::from_lines(&[
            &[c2, c2, C0, C0],
            &[c2, c2, C0, C0],
            &[C0, C0, C0, C0],
            &[C0, C0, C0, C0],
        ]);
        let products = FlowSemigroup::get_products(&left, &right, dim as u16);
        println!(
            "products\n{}",
            products.iter().map(|t| t.to_string()).join("\n\n"),
        );
        assert!(products.contains(&Flow::from_lines(&[
            &[c2, c2, C0, C0],
            &[c2, c2, C0, C0],
            &[C0, C0, C0, C0],
            &[C0, C0, C0, C0],
        ])));
        assert!(products.contains(&Flow::from_lines(&[
            &[c4, C0, C0, C0],
            &[C0, c4, C0, C0],
            &[C0, C0, C0, C0],
            &[C0, C0, C0, C0],
        ])));
    }
    /*left
    ( _ , 1 , 1 , _ , _ )
    ( _ , _ , _ , 1 , _ )
    ( _ , _ , _ , 1 , _ )
    ( _ , _ , _ , _ , 2 )
    ( _ , _ , _ , _ , _ )

    right
    ( _ , 1 , 1 , _ , _ )
    ( _ , _ , _ , 1 , _ )
    ( _ , _ , _ , 1 , _ )
    ( _ , _ , _ , _ , 2 )
    ( _ , _ , _ , _ , _ ) */
    #[test]
    fn get_products_test3() {
        let dim = 5;
        let c2 = Coef::Value(2);
        let left = Flow::from_lines(&[
            &[C0, C1, C1, C0, C0], //0 -- 1 --> {1,2}
            &[C0, C0, C0, C1, C0], //1 -- 1 --> 3
            &[C0, C0, C0, C1, C0], //2 -- 1 --> 3
            &[C0, C0, C0, C0, c2], //3 -- 2 --> 4
            &[C0, C0, C0, C0, C0], //
        ]);
        let right = left.clone();
        let products = FlowSemigroup::get_products(&left, &right, dim as u16);
        println!(
            "products\n{}",
            products.iter().map(|t| t.to_string()).join("\n\n"),
        );
        assert!(products.contains(&Flow::from_lines(&[
            &[C0, C0, C0, c2, C0], //0 -- 1 --> {1,2}
            &[C0, C0, C0, C0, C1], //1 -- 1 --> 3
            &[C0, C0, C0, C0, C1], //2 -- 1 --> 3
            &[C0, C0, C0, C0, C0], //3 -- 2 --> 4
            &[C0, C0, C0, C0, C0], //
        ])))
    }

    #[test]
    fn get_products_test4() {
        let dim = 5;
        let c2 = Coef::Value(2);
        let left = Flow::from_lines(&[
            &[C0, C1, C1, C0, C0],    //0 -- 1 --> {1,2}
            &[C0, C0, C0, C1, C0],    //1 -- 1 --> 3
            &[C0, C0, C0, C1, C0],    //2 -- 1 --> 3
            &[C0, C0, C0, C0, c2],    //3 -- 2 --> 4
            &[C0, C0, C0, C0, OMEGA], //
        ]);
        let right = left.clone();
        let products = FlowSemigroup::get_products(&left, &right, dim as u16);
        println!(
            "products\n{}",
            products.iter().map(|t| t.to_string()).join("\n\n"),
        );
        assert!(products.contains(&Flow::from_lines(&[
            &[C0, C0, C0, c2, C0],    //0 -- 1 --> {1,2}
            &[C0, C0, C0, C0, C1],    //1 -- 1 --> 3
            &[C0, C0, C0, C0, C1],    //2 -- 1 --> 3
            &[C0, C0, C0, C0, c2],    //3 -- 2 --> 4
            &[C0, C0, C0, C0, OMEGA], //
        ])))
    }
}
