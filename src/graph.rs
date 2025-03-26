use crate::partitions;
use std::{collections::HashSet, fmt};

#[derive(Clone, Debug, PartialEq)]
pub struct SubGraph(Vec<Option<usize>>);
impl SubGraph {
    pub(crate) fn iter(&self) -> std::slice::Iter<Option<usize>> {
        self.0.iter()
    }
}

/// A directed graph is a set of edges.
#[derive(Clone, Debug)]
pub struct Graph {
    edges: HashSet<(usize, usize)>,
    subgraphs: Vec<SubGraph>,
}

impl Graph {
    /// Create a new graph from a list of edges.
    pub fn new(dim: usize, edges: &[(usize, usize)]) -> Self {
        let edges: HashSet<(usize, usize)> = edges.iter().cloned().collect();
        let subgraphs = Self::get_deterministic_subgraphs(dim, &edges);
        Graph { edges, subgraphs }
    }

    /// Create a new graph from a list of edges.
    #[allow(dead_code)]
    pub(crate) fn from_vec(dim: usize, vec: Vec<(usize, usize)>) -> Graph {
        let edges: HashSet<(usize, usize)> = vec.into_iter().collect();
        let subgraphs = Self::get_deterministic_subgraphs(dim, &edges);
        Graph { edges, subgraphs }
    }

    /// Return an iterator over the edges of the graph.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &(usize, usize)> {
        self.edges.iter()
    }

    /// Return the successors of a node.
    pub(crate) fn get_successors(&self, i: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter_map(|&(i0, j0)| (i == i0).then_some(j0))
            .collect()
    }

    /// Returns the maximal deterministic subgraphs of the graph.
    /// This is a selection of edges such that every vertex has a single outgoing edge,
    /// or no outgoing edge at all if this is already the case in the original graph.
    pub(crate) fn get_maximal_deterministic_subgraphs(&self) -> &Vec<SubGraph> {
        &self.subgraphs
    }

    fn get_deterministic_subgraphs(dim: usize, edges: &HashSet<(usize, usize)>) -> Vec<SubGraph> {
        let successors = (0..dim)
            .map(|i| {
                let succ: Vec<usize> = edges
                    .iter()
                    .filter_map(|&(i0, j0)| (i == i0).then_some(j0))
                    .collect();
                if succ.is_empty() {
                    vec![None]
                } else {
                    succ.iter().map(|&x| Some(x)).collect()
                }
            })
            .collect::<Vec<_>>();
        partitions::cartesian_product(&successors)
            .into_iter()
            .map(|x| SubGraph(x))
            .collect()
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut vec: Vec<String> = self.edges.iter().map(|x| format!("{:?}", x)).collect();
        vec.sort();
        write!(f, "\n\t{}", vec.join("\n\t"))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_get_choices() {
        let edges = vec![(0, 1), (0, 2)];
        let graph = Graph::new(3, &edges);
        let choices = graph.get_maximal_deterministic_subgraphs();
        assert_eq!(choices.len(), 2);
        print!("{:?}", choices);
        assert!(choices.contains(&SubGraph(vec![Some(1), None, None])));
        assert!(choices.contains(&SubGraph(vec![Some(2), None, None])));
    }
}
