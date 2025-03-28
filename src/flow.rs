use crate::coef::{Coef, C0, C1, OMEGA};
use crate::graph::Graph;
use crate::nfa;
use crate::partitions;
use crate::sheep;
use crate::sheep::Sheep;
use itertools::Itertools;
use std::fmt;
use std::ops::Mul;
use std::{collections::HashSet, vec::Vec}; // Import the itertools crate for multi_cartesian_product

pub type Domain = Vec<Coef>;

#[derive(Eq, PartialEq, Hash, Clone, Debug)]
pub struct Flow {
    pub nb_rows: usize,
    pub nb_cols: usize,
    //size is nb_rows * nb_cols
    entries: Vec<Coef>,
}

impl PartialOrd for Flow {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let self_is_smaller_than_other =
            (0..self.nb_rows * self.nb_rows).all(|i| self.entries[i] <= other.entries[i]);
        let self_is_greater_than_other =
            (0..self.nb_rows * self.nb_rows).all(|i| self.entries[i] >= other.entries[i]);
        match (self_is_smaller_than_other, self_is_greater_than_other) {
            (true, true) => Some(std::cmp::Ordering::Equal),
            (true, false) => Some(std::cmp::Ordering::Less),
            (false, true) => Some(std::cmp::Ordering::Greater),
            (false, false) => None,
        }
    }
}

impl Mul for &Flow {
    type Output = Flow;
    fn mul(self, other: &Flow) -> Flow {
        self.product(other)
    }
}

impl Mul for Flow {
    type Output = Flow;
    fn mul(self, other: Flow) -> Flow {
        &self * &other
    }
}

impl Flow {
    pub fn from_entries(nb_rows: usize, nb_cols: usize, entries: &[Coef]) -> Flow {
        debug_assert_eq!(
            entries.len(),
            nb_rows * nb_cols,
            "Invalid number of entries"
        );
        Flow {
            nb_rows,
            nb_cols,
            entries: entries.into(),
        }
    }

    pub fn is_square(&self) -> bool {
        self.nb_rows == self.nb_cols
    }

    #[allow(dead_code)]
    pub fn from_lines(lines: &[&[Coef]]) -> Flow {
        let dim = lines.len();
        if lines.iter().any(|x| x.len() != dim) {
            panic!("Invalid line size ");
        }
        Flow::from_entries(
            dim,
            dim,
            &lines
                .iter()
                .flat_map(|x| x.iter())
                .cloned()
                .collect::<Vec<Coef>>(),
        )
    }

    pub fn product(&self, other: &Flow) -> Flow {
        let entries = &self.entries;
        let other_entries = &other.entries;
        let dim = self.nb_rows;
        let mut result: Vec<Coef> = vec![C0; dim * dim];
        //not idiomatic but fast
        let mut k = 0;
        for i in 0..dim {
            let i0 = i * dim;
            for j in 0..dim {
                //invariant k = i * dim + j
                let mut resultk: u16 = 0;
                let mut li = i0;
                let mut lj = j;
                let mut is_omega = false;
                //more effcicient than the idiomatic stream
                for _l in 0..dim {
                    let c = std::cmp::min(entries[li], other_entries[lj]);
                    match c {
                        Coef::Value(x) => {
                            resultk = std::cmp::max(resultk, x);
                            li += 1;
                            lj += dim;
                        }
                        OMEGA => {
                            is_omega = true;
                            break;
                        }
                    }
                }
                result[k] = if is_omega {
                    OMEGA
                } else {
                    Coef::Value(resultk)
                };
                k += 1;
            }
        }
        Flow {
            nb_rows: dim,
            nb_cols: dim,
            entries: result,
        }
    }

    //deterministic product
    pub fn compose(
        left: &Flow,
        transports: Vec<&(Vec<usize>, Flow, Vec<usize>)>,
        right: &Flow,
    ) -> Flow {
        debug_assert!(left.is_square());
        debug_assert!(right.is_square());
        debug_assert_eq!(left.nb_rows, right.nb_rows);
        let dim = left.nb_rows;
        debug_assert!(transports.iter().all(|(is, t, js)| is.len() == t.nb_rows
            && js.len() == t.nb_cols
            && is.iter().all(|&i| i < dim)
            && js.iter().all(|&j| j < dim)));
        let mut entries: Vec<Coef> = vec![C0; dim * dim];
        let mut k = 0;

        for i in 0..dim {
            let i0 = i * dim;
            for j in 0..dim {
                let mut li = i0;
                let mut lj = j;
                //more effcicient than the idiomatic stream
                for _l in 0..dim {
                    if left.entries[li] == OMEGA && right.entries[lj] == OMEGA {
                        entries[k] = OMEGA;
                        break;
                    } else {
                        li += 1;
                        lj += dim;
                    }
                }
                k += 1;
            }
        }
        /*
        for i in 0..left {
            let i0 = i * dim;
            for j in 0..dim {}
        } */
        Flow {
            nb_rows: dim,
            nb_cols: dim,
            entries,
        }
    }

    pub fn iteration(&self) -> Flow {
        let dim = self.nb_rows;
        let mut result: Flow = self.idempotent();
        for s0 in 0..dim {
            for t0 in 0..dim {
                if self.is_1(&s0, &t0) {
                    //debug!("processing ? -- {} -- {} -- ?", s0, t0);
                    for s in 0..dim {
                        if self.is_omega(&s, &s0) {
                            for t in 0..dim {
                                if self.is_omega(&t0, &t) {
                                    //debug!("found {} -- {} -- {} -- {}", s, s0, t0, t);
                                    result.entries[s * dim + t] = OMEGA;
                                }
                            }
                        }
                    }
                }
            }
        }
        result
    }

    ///computes the preimage of a target set of states
    /// that is the maximal ideal from which there exists a path to the target states
    /// finite coordinates are summed up...
    pub fn pre_image(&self, target: &[nfa::State]) -> Sheep {
        Sheep::from_vec(
            (0..self.nb_rows)
                .map(|i| target.iter().map(|&j| self.get(&i, &j)).sum::<Coef>())
                .collect(),
        )
    }

    //compute all possible flows compatible with this domain and edges
    //there might be choice from small constants: a 5 distributed on 3 edges might lead to (5 + 0 + 0) or (1 + 1+ 3)
    //this is annoyingly exponential and should be stored in the future as a compact representation,
    //compatible with product and start in the monoid
    //but for now we will just compute it exhasutively
    /*
    WIP
    */
    pub(crate) fn from_domain_and_edges(domain: &sheep::Sheep, edges: &Graph) -> HashSet<Flow> {
        //debug!("Creating flow from domain and edges");
        //debug!("domain\n{}", domain);
        //debug!("edges{}", edges);

        let dim = domain.len();
        if edges.iter().any(|f| f.0 >= dim || f.1 >= dim) {
            panic!("Edge out of domain");
        }
        let lines = Self::get_lines_vec(domain, edges);
        lines
            .iter()
            .multi_cartesian_product()
            .map(|x| Flow {
                nb_rows: dim,
                nb_cols: dim,
                entries: x.into_iter().flatten().cloned().collect(),
            })
            .collect()
    }

    //iteration of a fl
    fn idempotent(&self) -> Flow {
        let mut result = self.clone();
        loop {
            let result_squared = &result * &result;
            if result == result_squared {
                break;
            }
            result = result_squared;
        }
        result
    }

    pub fn get(&self, i: &usize, j: &usize) -> Coef {
        self.entries[i * self.nb_rows + j]
    }

    pub(crate) fn set(&mut self, i: &usize, j: &usize, c: Coef) {
        self.entries[i * self.nb_rows + j] = c;
    }

    fn is_1(&self, i: &usize, j: &usize) -> bool {
        self.entries[i * self.nb_rows + j] == C1
    }

    fn is_omega(&self, i: &usize, j: &usize) -> bool {
        self.entries[i * self.nb_rows + j] == OMEGA
    }

    fn get_lines_vec(domain: &sheep::Sheep, edges: &Graph) -> Vec<Vec<Domain>> {
        let dim = domain.len();
        domain
            .iter()
            .enumerate()
            .map(move |(i, &coef)| {
                let out = edges.get_successors(i);
                Self::get_lines(&out, &coef, dim)
            })
            .collect::<Vec<_>>()
    }

    //todo cache results
    fn get_lines(out: &[usize], coef: &Coef, dim: usize) -> Vec<Domain> {
        match *coef {
            C0 => vec![vec![C0; dim]],
            OMEGA => vec![(0..dim)
                .map(|i| if out.contains(&i) { OMEGA } else { C0 })
                .collect()],
            Coef::Value(x) => partitions::get_partitions(x, out.len())
                .iter()
                .map(|p| {
                    let mut result = vec![C0; dim];
                    for (i, j) in out.iter().zip(p.iter()) {
                        result[*i] = Coef::Value(*j);
                    }
                    result
                })
                .collect(),
        }
    }

    //todo: store in object if heavy use
    pub(crate) fn edges_to(&self, j: usize) -> Vec<(usize, Coef)> {
        (0..self.nb_rows)
            .map(|i| (i % self.nb_rows, self.get(&i, &j)))
            .collect()
    }

    //todo: store in object if heavy use
    pub(crate) fn edges_from(&self, i: usize) -> Vec<(usize, Coef)> {
        (0..self.nb_rows)
            .map(|j| (i % self.nb_rows, self.get(&i, &j)))
            .collect()
    }
}

impl fmt::Display for Flow {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut result = String::new();
        for i in 0..self.nb_rows {
            let sheep = sheep::Sheep::from_vec(
                self.entries[i * self.nb_rows..(i + 1) * self.nb_rows].to_vec(),
            );
            result.push_str(&sheep.to_string());
            result.push('\n');
        }
        write!(f, "{}", result)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::coef::{C0, C1, C2, C3};

    #[test]
    #[should_panic]
    fn from_domain_and_edges_panic_case() {
        let domain = sheep::Sheep::from_vec(vec![C1, C2, C3]);
        let edges = Graph::from_vec(2, vec![(0, 1), (1, 3)]);
        Flow::from_domain_and_edges(&domain, &edges);
    }

    #[test]
    fn get_lines_test() {
        let out = vec![0, 1];
        let dim = 3;
        assert_eq!(
            Flow::get_lines(&out, &C3, dim),
            vec![
                vec![C3, C0, C0],
                vec![C2, C1, C0],
                vec![C1, C2, C0],
                vec![C0, C3, C0],
            ]
        );
    }

    #[test]
    fn get_lines_omega_test() {
        let out = vec![0, 1];
        let coef = Coef::Omega;
        let dim = 3;
        let expected = vec![vec![OMEGA, OMEGA, C0]];
        assert_eq!(Flow::get_lines(&out, &coef, dim), expected);
    }

    //test _get_lines_vec on an example with domain=[1,3,omega] and edges=[(0,1),(1,0),(1,1),(2,1),(2,2)]
    #[test]
    fn get_lines_vec_test() {
        let domain = sheep::Sheep::from_vec(vec![C1, C3, OMEGA]);
        let edges = Graph::from_vec(3, vec![(0, 1), (1, 0), (1, 1), (2, 1), (2, 2)]);
        let expected = [
            vec![vec![C0, C1, C0]],
            vec![
                vec![C3, C0, C0],
                vec![C2, C1, C0],
                vec![C1, C2, C0],
                vec![C0, C3, C0],
            ],
            vec![vec![C0, OMEGA, OMEGA]],
        ];
        let computed = Flow::get_lines_vec(&domain, &edges);
        //check computed and expected are equal, up to order of elements
        assert_eq!(computed.len(), expected.len());
        assert_eq!(computed[0], expected[0]);
        assert_eq!(computed[2], expected[2]);
        assert_eq!(computed[1].len(), expected[1].len());
        for x in &computed[1] {
            assert!(expected[1].contains(&x));
        }
    }

    //tests from_domain_and_edges on an example with domain=[1,3,omega] and edges=[(0,1),(1,0),(1,1),(2,1),(2,2)]
    #[test]
    fn from_domain_and_edges_test() {
        let domain = sheep::Sheep::from_vec(vec![C1, C3, OMEGA]);
        let edges = Graph::from_vec(3, vec![(0, 1), (1, 0), (1, 1), (2, 1), (2, 2)]);
        let flows = Flow::from_domain_and_edges(&domain, &edges);
        let expected = vec![
            Flow {
                nb_rows: 3,
                nb_cols: 3,
                entries: vec![C0, C1, C0, C0, C3, C0, C0, OMEGA, OMEGA],
            },
            Flow {
                nb_rows: 3,
                nb_cols: 3,
                entries: vec![C0, C1, C0, C1, C2, C0, C0, OMEGA, OMEGA],
            },
            Flow {
                nb_rows: 3,
                nb_cols: 3,
                entries: vec![C0, C1, C0, C2, C1, C0, C0, OMEGA, OMEGA],
            },
            Flow {
                nb_rows: 3,
                nb_cols: 3,
                entries: vec![C0, C1, C0, C3, C0, C0, C0, OMEGA, OMEGA],
            },
        ];
        assert_eq!(flows, expected.into_iter().collect());
    }

    #[test]
    fn idempotent_test1() {
        let flow = Flow::from_lines(&[
            &[OMEGA, OMEGA, C0, C0],
            &[C0, C0, C1, C0],
            &[C0, C0, C0, OMEGA],
            &[C0, C0, C0, C0],
        ]);
        let expected = Flow::from_lines(&[
            &[OMEGA, OMEGA, C1, C1],
            &[C0, C0, C0, C0],
            &[C0, C0, C0, C0],
            &[C0, C0, C0, C0],
        ]);
        assert_eq!(flow.idempotent(), expected);
    }

    #[test]
    fn idempotent_test2() {
        let flow = Flow::from_lines(&[
            &[OMEGA, OMEGA, C0, C0],
            &[C0, C0, C1, C0],
            &[C0, C0, C0, OMEGA],
            &[C0, C0, C0, OMEGA],
        ]);
        let expected = Flow::from_lines(&[
            &[OMEGA, OMEGA, C1, C1],
            &[C0, C0, C0, C1],
            &[C0, C0, C0, OMEGA],
            &[C0, C0, C0, OMEGA],
        ]);
        assert_eq!(flow.idempotent(), expected);
    }

    //test iteration on the flow OMEGA 1 0 OMEGA
    #[test]
    fn iteration_test() {
        let flow = Flow {
            nb_rows: 2,
            nb_cols: 2,
            entries: vec![OMEGA, C1, C0, OMEGA],
        };
        let expected = Flow {
            nb_rows: 2,
            nb_cols: 2,
            entries: vec![OMEGA, OMEGA, C0, OMEGA],
        };
        assert_eq!(flow.iteration(), expected);
    }

    #[test]
    fn iteration_test2() {
        let flow = Flow::from_lines(&[
            &[OMEGA, OMEGA, C0, C0],
            &[C0, C0, C1, C0],
            &[C0, C0, C0, OMEGA],
            &[C0, C0, C0, OMEGA],
        ]);
        let expected = Flow::from_lines(&[
            &[OMEGA, OMEGA, C1, OMEGA],
            &[C0, C0, C0, C1],
            &[C0, C0, C0, OMEGA],
            &[C0, C0, C0, OMEGA],
        ]);
        assert_eq!(flow.iteration(), expected);
    }

    #[test]
    fn iteration_test3() {
        let flow = Flow::from_lines(&[
            &[OMEGA, OMEGA, C0, C0],
            &[C0, OMEGA, C1, C0],
            &[C0, C0, C0, OMEGA],
            &[C0, C0, C0, OMEGA],
        ]);
        let expected = Flow::from_lines(&[
            &[OMEGA, OMEGA, C1, OMEGA],
            &[C0, OMEGA, C1, OMEGA],
            &[C0, C0, C0, OMEGA],
            &[C0, C0, C0, OMEGA],
        ]);
        assert_eq!(flow.iteration(), expected);
    }

    #[test]
    fn iteration_test4() {
        let flow = Flow::from_lines(&[
            &[OMEGA, OMEGA, C0, C0],
            &[C0, OMEGA, C1, C0],
            &[C0, C0, OMEGA, OMEGA],
            &[C0, C0, C0, OMEGA],
        ]);
        let expected = Flow::from_lines(&[
            &[OMEGA, OMEGA, OMEGA, OMEGA],
            &[C0, OMEGA, OMEGA, OMEGA],
            &[C0, C0, OMEGA, OMEGA],
            &[C0, C0, C0, OMEGA],
        ]);
        assert_eq!(flow.iteration(), expected);
    }

    //tests preimage
    #[test]
    fn pre_image() {
        let flow = Flow::from_lines(&[
            &[OMEGA, OMEGA, C0, C0],
            &[C0, OMEGA, C1, C2],
            &[C0, C0, OMEGA, OMEGA],
            &[C0, C0, C0, OMEGA],
        ]);
        assert_eq!(
            flow.pre_image(&[0]),
            Sheep::from_vec(vec![OMEGA, C0, C0, C0])
        );
        assert_eq!(
            flow.pre_image(&[2, 3]),
            Sheep::from_vec(vec![C0, C3, OMEGA, OMEGA])
        );
        assert_eq!(
            flow.pre_image(&[1, 2]),
            Sheep::from_vec(vec![OMEGA, OMEGA, OMEGA, C0])
        );
    }
}
