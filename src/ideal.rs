use log::debug;

use crate::coef::{Coef, OMEGA};
use crate::sheep::Sheep;
use crate::{coef, partitions};
use std::fmt;
use std::{collections::HashSet, vec::Vec};

/*
An ideal is mathmatically a downward closed set of vectors in N^S.
It is represented as a set of sheep, all have the same dimension,
and the ideal is the union of downard-closure of those sheep.

Several heuristics are used in order to keep the size of the set small:
* a call to 'insertion' of a sheep which is already contained in the ideal has no effect
* a call to 'minimize' removes configurations which are covered by others

The method 'restrict_to' computes the intersection of the ideal with another ideal.
The method 'pre_image' computes the pre-image of an ideal by a graph.
The method 'is_safe' checks whether it is safe to play a configuration w.r. to the graph, in the sense that it ensures the next configuration belongs to the ideal.

 */
#[derive(Clone, Eq, Debug)]
pub struct Ideal(HashSet<Sheep>);

impl PartialEq for Ideal {
    fn eq(&self, other: &Self) -> bool {
        self.is_contained_in(other) && other.is_contained_in(self)
    }
}

impl Ideal {
    /// Create an empty ideal.
    fn new() -> Self {
        Ideal(HashSet::new())
    }

    /// Create an ideal from a vector of sheeps.
    pub(crate) fn from_vec(w: &[Sheep]) -> Self {
        Ideal(w.iter().cloned().collect())
    }

    //returns the dimension of the ideal
    pub(crate) fn dim(&self) -> Option<usize> {
        self.0.iter().next().map(|ideal| ideal.len())
    }

    /// Create an ideal from a vector of vectors of coefficients.
    /// The method is used in the tests.
    #[allow(dead_code)]
    pub(crate) fn from_vecs(w: &[&[Coef]]) -> Self {
        Ideal(w.iter().map(|&v| Sheep::from_vec(v.to_vec())).collect())
    }

    /// Check if a sheep belongs to the ideal.
    /// The ideal is by definition the union of downard-closure of the sheeps it contains
    pub(crate) fn contains(&self, source: &Sheep) -> bool {
        self.0.iter().any(|x| source <= x)
    }

    /// Check if the ideal is contained in another ideal.
    pub(crate) fn is_contained_in(&self, other: &Ideal) -> bool {
        self.0.iter().all(|x| other.contains(x))
    }

    /// Insert a sheep in the ideal.
    /// The method returns true if the ideal has changed, and false if the sheep was already in the ideal.
    pub fn insert(&mut self, sheep: &Sheep) -> bool {
        if self.0.contains(sheep) {
            false
        } else {
            self.0.insert(sheep.clone());
            true
        }
    }

    /// Get an iterator over the sheeps in the ideal.
    pub(crate) fn sheeps(&self) -> impl Iterator<Item = &Sheep> {
        self.0.iter()
    }

    /// Compute the intersection of the ideal with another ideal.
    /// The method returns true if the ideal has changed.
    /// The method is used in the solver to restrict the set of possible configurations.
    ///
    /// # Examples
    /// ```
    /// use crate::ideal::Ideal;
    /// use crate::coef::{C0, C1, C2, OMEGA};
    /// let mut ideal0 = Ideal::from_vecs(&[&[C0, C1, C2, OMEGA], &[OMEGA, C2, C1, C0]]);
    /// let mut ideal1 = Ideal::from_vecs(&[&[OMEGA, C1, C2, OMEGA], &[OMEGA, C2, C1, OMEGA]]);
    /// let ideal2 = Ideal::from_vecs(&[&[C1, OMEGA, C1, C2], &[C2, OMEGA, C1, C1]]);
    /// let ideal0original = ideal0.clone();
    /// let changed0 = ideal0.restrict_to(&ideal1);
    /// assert!(!changed0);
    /// assert_eq!(ideal0, ideal0original);
    ///
    /// let ideal1original = ideal1.clone();
    /// let changed1 = ideal1.restrict_to(&ideal2);
    /// assert!(changed1);
    /// assert!(ideal1 != ideal1original);
    /// assert_eq!(ideal1, Ideal::from_vecs(&[&[C2, C2, C1, C1], &[C1, C2, C1, C2]]));
    /// ```
    pub(crate) fn restrict_to(&mut self, other: &Ideal) -> bool {
        let mut changed = false;
        let mut new_sheeps = Ideal::new();
        for sheep in self.0.iter() {
            if other.contains(sheep) {
                new_sheeps.insert(sheep);
            } else {
                for other_sheep in &other.0 {
                    changed |= new_sheeps.insert(&Sheep::intersection(sheep, other_sheep));
                }
            }
        }
        new_sheeps.minimize();
        self.0 = new_sheeps.0;
        changed
    }

    /// Compute the pre-image of the ideal by a graph.
    /// The method is used in the solver to compute the set of configurations from which it is safe to play an action.
    /// The method returns the set of configurations which are safe to play.
    ///
    /// # Examples
    /// ```
    /// let edges = crate::graph::Graph::from_vec(vec![(0, 0), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]);
    /// let ideal1 = Ideal::from_vecs(&[&[OMEGA, C1, C2, OMEGA], &[OMEGA, C2, C1, OMEGA]]);
    /// let pre_image1 = ideal1.pre_image(&edges);
    /// assert_eq!(
    ///    pre_image1,
    ///    Ideal::from_vecs(&[
    ///        &[OMEGA, C2, C0, OMEGA],
    ///        &[OMEGA, C0, C2, OMEGA],
    ///        &[OMEGA, C1, C1, OMEGA]
    ///    ])
    /// );
    /// ```
    ///
    /// ```
    /// use crate::ideal::Ideal;
    /// use crate::coef::{C0, C1, C2, OMEGA};
    /// let edges = crate::graph::Graph::from_vec(vec![(0, 0), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]);
    /// let ideal0 = Ideal::from_vecs(&[&[C0, C1, C2, OMEGA]]);
    /// let pre_image0 = ideal0.pre_image(&edges);
    /// assert_eq!(
    ///     pre_image0,
    ///        Ideal::from_vecs(&[&[C0, C1, C1, OMEGA], &[C0, C0, C2, OMEGA]]),
    /// );
    /// ```
    pub(crate) fn pre_image(&self, edges: &crate::graph::Graph) -> Ideal {
        let dim = self.0.iter().next().map_or(0, |x| x.len());
        if dim == 0 {
            return Ideal::new();
        }

        //compute for every j the maximal finite coef appearing at index j, if exists
        let max_finite_coordsj = (0..dim)
            .map(|j| {
                self.0
                    .iter()
                    .filter_map(|sheep| match sheep.get(j) {
                        Coef::Omega => None,
                        Coef::Value(c) => Some(c),
                    })
                    .max()
            })
            .collect::<Vec<_>>();

        let max_finite_coordsi = (0..dim)
            .map(|i| {
                {
                    edges
                        .get_successors(i)
                        .iter()
                        .filter_map(|&j| max_finite_coordsj[j])
                        .max()
                }
            })
            .collect::<Vec<_>>();

        //compute for every i whether omega is possible at i, which happens iff there exists a sheep in the ideal such that all successors lead to omega
        let is_omega_possible = (0..dim)
            .map(|i| {
                let succ = edges.get_successors(i);
                return self.0.iter().any(|sheep| sheep.all_omega(&succ));
            })
            .collect::<Vec<_>>();

        //print max_finite_coords and is_omega_possible
        debug!("preimage of {}\n by\n{}\n", self, edges);
        debug!(
            "\nmax_finite_coordsi {:?}\nis_omega_possible {:?}\n",
            max_finite_coordsi, is_omega_possible
        );

        let possible_coefs = (0..dim)
            .map(|i| {
                match (
                    max_finite_coordsi.get(i).unwrap(),
                    is_omega_possible.get(i).unwrap(),
                ) {
                    (Some(c), false) => (0..(std::cmp::max(*c, 1) + 1))
                        .map(coef::Coef::Value)
                        .rev()
                        .collect::<Vec<_>>(),
                    (Some(c), true) => (0..(std::cmp::max(*c, 1) + 1))
                        .map(coef::Coef::Value)
                        .chain(std::iter::once(OMEGA))
                        .rev()
                        .collect::<Vec<_>>(),
                    (None, true) => vec![OMEGA],
                    (None, false) => panic!("logically inconsistent case"),
                }
            })
            .collect::<Vec<_>>();
        //debug!("max_finite_coords: {:?}\n", max_finite_coordsi);
        //debug!("is_omega_possible: {:?}\n", is_omega_possible);
        //debug!("possible_coefs: {:?}\n", possible_coefs);

        let mut result = Ideal::new();
        for candidate in partitions::cartesian_product(&possible_coefs) {
            //debug!("candidate: {:?}\n", candidate);
            if self.is_safe(&candidate, edges) {
                //debug!("\t...ok\n");
                result.insert(&Sheep::from_vec(candidate));
            }
        }
        result.minimize();
        debug!("result {}\n", result);
        result
    }

    /// Check whether it is safe to play the graph in  candidate, in the sense that it ensures
    /// the next configuration belongs to the ideal.
    ///
    /// # Examples
    /// ```
    /// use crate::ideal::Ideal;
    /// use crate::coef::{C0, C1, C2, OMEGA};
    /// let edges = crate::graph::Graph::from_vec(vec![(0, 1), (0, 2)]);
    /// let ideal = Ideal::from_vecs(&[&[C0, C1, C0], &[C0, C0, C1]]);
    /// let candidate = vec![C1, C0, C0];
    /// assert!(ideal.is_safe(&candidate, &edges));
    /// ```
    ///
    /// ```
    /// use crate::ideal::Ideal;
    /// use crate::coef::C0;
    /// let c3 = Coef::Value(3);
    /// let c4 = Coef::Value(4);
    /// let edges = crate::graph::Graph::from_vec(vec![(0, 1), (0, 2)]);
    /// let ideal = Ideal::from_vecs(&[&[C0, c3, C0], &[C0, C0, c3]]);
    /// let candidate = vec![c4, C0, C0];
    /// assert!(!ideal.is_safe(&candidate, &edges));
    /// ```
    fn is_safe(&self, candidate: &[Coef], edges: &crate::graph::Graph) -> bool {
        let dim = candidate.len();
        let choices = edges.get_maximal_deterministic_subgraphs(dim);
        //debug!("edges:\n{:?}\n", edges);
        //debug!("choices:\n{:?}\n", choices);
        choices.iter().all(|choice| {
            //debug!("choice:\n{:?}\n", choice);
            let image = (0..dim)
                .map(|j: usize| {
                    choice
                        .iter()
                        .enumerate()
                        .filter(|(_, &j0)| j0 == Some(j))
                        .filter_map(|(i0, _)| candidate.get(i0))
                        .sum()
                })
                .collect();
            //debug!("image:\n{:?}\n", image);
            self.contains(&Sheep::from_vec(image))
        })
    }

    /// Remove from the ideal any element strictly smaller than another.
    /// The method is used in the solver to keep the size of the representation small.
    fn minimize(&mut self) -> bool {
        //remove from self.0 any element strictly smaller than another
        let mut changed = false;
        for sheep in self
            .0
            .iter()
            .filter(|&x| self.0.iter().any(|y| x < y))
            .cloned()
            .collect::<Vec<_>>()
        {
            changed |= self.0.remove(&sheep);
        }
        changed
    }
}

impl fmt::Display for Ideal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut vec: Vec<String> = self.0.iter().map(|x| x.to_string()).collect();
        vec.sort();
        write!(f, "{{\n\n{}\n\n}}\n", vec.join(" ,\n"))
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::coef::{C0, C1, C2, OMEGA};

    #[test]
    fn is_in_ideal() {
        let master_sheep = Sheep::from_vec(vec![OMEGA, OMEGA]);
        let medium_sheep = Sheep::from_vec(vec![C1, C1]);
        let ini_sheep = Sheep::from_vec(vec![C1, C0]);
        let final_sheep = Sheep::from_vec(vec![C0, C1 + C1]);

        let ideal = Ideal([ini_sheep.clone(), final_sheep.clone()].into());
        assert!(ideal.contains(&ini_sheep));
        assert!(ideal.contains(&final_sheep));
        assert!(!ideal.contains(&master_sheep));
        assert!(!ideal.contains(&medium_sheep));

        let ideal2 = Ideal([medium_sheep.clone()].into());
        assert!(ideal2.contains(&ini_sheep));
        assert!(!ideal2.contains(&final_sheep));
        assert!(!ideal2.contains(&master_sheep));
        assert!(ideal2.contains(&medium_sheep));
    }

    //test equality
    #[test]
    fn order() {
        let ideal0 = Ideal::from_vecs(&[&[C0, C1, C2, OMEGA], &[OMEGA, C2, C1, C0]]);
        let ideal1 = Ideal::from_vecs(&[&[OMEGA, C1, C2, OMEGA], &[OMEGA, C2, C1, OMEGA]]);
        let ideal2 = Ideal::from_vecs(&[&[OMEGA, C2, C2, OMEGA]]);

        assert!(ideal0.is_contained_in(&ideal1));
        assert!(ideal1.is_contained_in(&ideal2));
        assert!(ideal0.is_contained_in(&ideal2));
    }

    #[test]
    fn restrict_to() {
        let mut ideal0 = Ideal::from_vecs(&[&[C0, C1, C2, OMEGA], &[OMEGA, C2, C1, C0]]);
        let mut ideal1 = Ideal::from_vecs(&[&[OMEGA, C1, C2, OMEGA], &[OMEGA, C2, C1, OMEGA]]);
        let ideal2 = Ideal::from_vecs(&[&[C1, OMEGA, C1, C2], &[C2, OMEGA, C1, C1]]);

        let ideal0original = ideal0.clone();
        let changed0 = ideal0.restrict_to(&ideal1);
        assert!(!changed0);
        assert_eq!(ideal0, ideal0original);

        let ideal1original = ideal1.clone();
        let changed1 = ideal1.restrict_to(&ideal2);
        assert!(changed1);
        assert!(ideal1 != ideal1original);
        assert_eq!(
            ideal1,
            Ideal::from_vecs(&[&[C2, C2, C1, C1], &[C1, C2, C1, C2]])
        );
        assert_eq!(
            ideal1,
            Ideal::from_vecs(&[&[C2, C2, C1, C1], &[C1, C1, C1, C2], &[C1, C2, C1, C2]])
        );
        assert_eq!(
            ideal1,
            Ideal::from_vecs(&[
                &[C1, C2, C1, C2],
                &[C2, C2, C1, C1],
                &[C1, C1, C1, C2],
                &[C2, C1, C1, C1],
            ])
        );
    }

    //test issafe
    #[test]
    fn is_safe() {
        let edges = crate::graph::Graph::from_vec(vec![(0, 1), (0, 2)]);
        let ideal = Ideal::from_vecs(&[&[C0, C1, C0], &[C0, C0, C1]]);
        let candidate = vec![C1, C0, C0];
        assert!(ideal.is_safe(&candidate, &edges));
    }

    #[test]
    fn is_safe2() {
        let c4 = Coef::Value(4);
        let edges = crate::graph::Graph::from_vec(vec![(0, 1), (0, 2)]);
        let ideal = Ideal::from_vecs(&[&[C0, c4, C0], &[C0, C0, c4]]);
        let candidate = vec![c4, C0, C0];
        assert!(ideal.is_safe(&candidate, &edges));
    }

    #[test]
    fn is_not_safe() {
        let c3 = Coef::Value(3);
        let c4 = Coef::Value(4);
        let edges = crate::graph::Graph::from_vec(vec![(0, 1), (0, 2)]);
        let ideal = Ideal::from_vecs(&[&[C0, c3, C0], &[C0, C0, c3]]);
        let candidate = vec![c4, C0, C0];
        assert!(!ideal.is_safe(&candidate, &edges));
    }

    #[test]
    fn pre_image1() {
        let edges =
            crate::graph::Graph::from_vec(vec![(0, 0), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]);
        let ideal0 = Ideal::from_vecs(&[&[C0, C1, C2, OMEGA]]);

        let pre_image0 = ideal0.pre_image(&edges);
        assert_eq!(
            pre_image0,
            Ideal::from_vecs(&[&[C0, C1, C1, OMEGA], &[C0, C0, C2, OMEGA]]),
        );
    }

    #[test]
    fn pre_image1bis() {
        let edges =
            crate::graph::Graph::from_vec(vec![(0, 0), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]);
        let ideal1 = Ideal::from_vecs(&[&[OMEGA, C1, C2, OMEGA], &[OMEGA, C2, C1, OMEGA]]);
        let pre_image1 = ideal1.pre_image(&edges);
        assert_eq!(
            pre_image1,
            Ideal::from_vecs(&[
                &[OMEGA, C2, C0, OMEGA],
                &[OMEGA, C0, C2, OMEGA],
                &[OMEGA, C1, C1, OMEGA]
            ])
        );
    }

    #[test]
    fn pre_image2() {
        let edges = crate::graph::Graph::from_vec(vec![(0, 1), (0, 2)]);
        let ideal0 = Ideal::from_vecs(&[&[C0, C0, OMEGA], &[C0, OMEGA, C0]]);
        let pre_image0 = ideal0.pre_image(&edges);
        assert_eq!(pre_image0, Ideal::from_vecs(&[&[C1, OMEGA, OMEGA]]));
    }
}
