use std::{collections::HashSet, fmt};

#[derive(Clone, Debug, PartialEq)]
pub struct SubGraph(Vec<Option<usize>>);

/// A directed graph is a set of edges.
#[derive(Clone, Debug)]
pub struct Graph {
    dim: usize,
    edges: HashSet<(usize, usize)>,
}

impl Graph {
    /// Create a new graph from a list of edges.
    pub fn new(dim: usize, edges: &[(usize, usize)]) -> Self {
        let edges: HashSet<(usize, usize)> = edges.iter().cloned().collect();
        Graph { dim, edges }
    }

    /// Create a new graph from a list of edges.
    #[allow(dead_code)]
    pub fn from_vec(dim: usize, vec: Vec<(usize, usize)>) -> Graph {
        let edges: HashSet<(usize, usize)> = vec.into_iter().collect();
        Graph { dim, edges }
    }

    /// Return an iterator over the edges of the graph.
    pub fn iter(&self) -> impl Iterator<Item = &(usize, usize)> {
        self.edges.iter()
    }

    /// Return the successors of a node.
    pub fn get_successors(&self, i: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter_map(|&(i0, j0)| (i == i0).then_some(j0))
            .collect()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut vec: Vec<String> = self.edges.iter().map(|x| format!("{:?}", x)).collect();
        vec.sort();
        write!(f, "\n\t{}", vec.join("\n\t"))
    }
}
