use std::{collections::HashSet, vec::Vec};

use crate::sheep;

pub const INFTY: usize = usize::MAX;
pub const OMEGA: usize = INFTY - 1;

pub type Domain = Vec<usize>;
pub type Image = Vec<usize>;

#[derive(Eq, PartialEq, Hash, Clone)]
pub struct Flow {
    pub dim: usize,
    //size is dim * dim
    entries: Vec<usize>,
}

pub trait FlowTrait {
    fn dom(&self) -> Domain;
    fn im(&self) -> Image;
    fn product(&self, other: &Flow) -> Flow;
    fn iteration(&self) -> HashSet<Flow>;
}

impl FlowTrait for Flow {
    fn dom(&self) -> Domain {
        Self::_dom(self.dim, &self.entries)
    }

    fn im(&self) -> Image {
        Self::_im(self.dim, &self.entries)
    }

    //product of two flows
    fn product(&self, other: &Flow) -> Flow {
        let dim = self.dim;
        let entries = Self::_product(&self.entries, &other.entries, dim);
        Flow { dim, entries }
    }

    fn iteration(&self) -> HashSet<Flow> {
        //!todo!("generates all possible sharp results");
        HashSet::from([Self::_iteration(&self.dom(), self.dim)])
    }
}

impl Flow {
    fn _dom(dim: usize, entries: &Vec<usize>) -> Domain {
        if entries.len() != dim * dim {
            panic!("Invalid number of entries");
        }
        let mut result = vec![0; dim];
        if dim == 0 {
            return result;
        }
        for i in 0..dim {
            let max = entries[i * dim..(i + 1) * dim].iter().max().unwrap();

            result[i] = match *max {
                0 => 0,
                OMEGA => OMEGA,
                INFTY => INFTY,
                _ => entries[i * dim..(i + 1) * dim].iter().sum(),
            };
        }
        result
    }

    fn _im(dim: usize, entries: &Vec<usize>) -> Image {
        if entries.len() != dim * dim {
            panic!("Invalid number of entries");
        }
        let mut result = vec![0; dim];
        if dim == 0 {
            return result;
        }
        for j in 0..dim {
            let max = (0..dim).map(|i| entries[i * dim + j]).max().unwrap();
            result[j] = match max {
                0 => 0,
                OMEGA => OMEGA,
                INFTY => INFTY,
                _ => (0..dim).map(|i| entries[i * dim + j]).sum(),
            };
        }
        result
    }

    fn _product(entries: &Vec<usize>, other_entries: &Vec<usize>, dim: usize) -> Vec<usize> {
        let mut result = vec![0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                result[i * dim + j] = (0..dim)
                    .map(|k| std::cmp::min(entries[i * dim + k], other_entries[k * dim + j]))
                    .max()
                    .unwrap();
            }
        }
        result
    }

    //iteration of a flow

    pub fn _iteration(entries: &Vec<usize>, dim: usize) -> Flow {
        let mut result = entries.clone();
        loop {
            let result_squared = Self::_product(&result, &result, dim);
            if result == result_squared {
                return Flow {
                    dim,
                    entries: result,
                };
            }
            result = result_squared;
        }
    }

    pub(crate) fn from_domain_and_edges(
        domain: &sheep::Sheep,
        edges: &HashSet<(usize, usize)>,
    ) -> Flow {
        let dim = domain.len();
        if edges.iter().any(|f| f.0 >= dim || f.1 >= dim) {
            panic!("Edge out of domain");
        }
        let mut entries = vec![0; dim * dim];
        for (i, j) in edges.iter() {
            entries[i * dim + j] = domain[*i];
        }
        Flow { dim, entries }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn from_domain_and_edges() {
        let domain = sheep::Sheep::from(vec![1, 2, 3]);
        let mut edges = HashSet::new();
        edges.insert((0, 1));
        edges.insert((1, 2));
        let flow = Flow::from_domain_and_edges(&domain, &edges);
        assert_eq!(flow.entries, vec![0, 1, 0, 0, 0, 2, 0, 0, 0]);
    }

    #[test]
    #[should_panic]
    fn from_domain_and_edges_panic_case() {
        let domain = sheep::Sheep::from(vec![1, 2, 3]);
        let mut edges = HashSet::new();
        edges.insert((0, 1));
        edges.insert((1, 3));

        let default_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        Flow::from_domain_and_edges(&domain, &edges);

        std::panic::set_hook(default_hook);
    }
}
