use crate::coef::{Coef, C0, OMEGA};
use crate::memoizer::Memoizer;
use crate::sheep::Sheep;
use crate::{coef, partitions};
use cached::proc_macro::cached;
use itertools::Itertools;
use log::debug;
use once_cell::sync::Lazy;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::fmt;
use std::sync::Mutex;
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

type CoefsCollection = Vec<Vec<Coef>>;
type Herd = Vec<Sheep>;
type CoefsCollectionMemoizer = Memoizer<CoefsCollection, Herd, fn(&CoefsCollection) -> Herd>;
static POSSIBLE_COEFS_CACHE: Lazy<Mutex<CoefsCollectionMemoizer>> = Lazy::new(|| {
    Mutex::new(Memoizer::new(|possible_coefs| {
        compute_possible_coefs(possible_coefs)
            .map(|v| Sheep::from_vec(v))
            .collect()
    }))
});

fn compute_possible_coefs(possible_coefs: &CoefsCollection) -> impl Iterator<Item = Vec<Coef>> {
    possible_coefs
        .iter()
        .map(|v| {
            let coef = v
                .iter()
                .filter_map(|&x| match x {
                    OMEGA => None,
                    Coef::Value(c) => Some(c),
                })
                .next();
            let is_omega = v.contains(&OMEGA);
            match (is_omega, coef) {
                (false, None) => vec![C0],
                (true, None) => vec![OMEGA],
                (false, Some(c)) => (0..c + 1).map(Coef::Value).collect(),
                (true, Some(c)) => std::iter::once(OMEGA)
                    .chain((0..c + 1).map(Coef::Value))
                    .collect(),
            }
        })
        .multi_cartesian_product()
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
    /// let mut ideal0 = Ideal::&[&[C0, C1, C2, OMEGA], &[OMEGA, C2, C1, C0]]);
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
                changed = true;
                for other_sheep in &other.0 {
                    new_sheeps.insert(&Sheep::intersection(sheep, other_sheep));
                }
            }
        }
        if changed {
            new_sheeps.minimize();
            self.0 = new_sheeps.0;
        }
        changed
    }

    pub(crate) fn restrict_to_preimage_of(
        &mut self,
        safe_target: &Ideal,
        edges: &crate::graph::Graph,
        dim: usize,
        max_finite_value: u16,
    ) -> bool {
        let mut changed = false;
        let mut new_sheeps = Ideal::new();
        debug!(
            "restrict_to_preimage_of\ndim: {}\nmax_finite_value: {}\nself\n{}\nsafe_target\n{}\nedges\n{}\n",
            dim, max_finite_value, self, safe_target, edges
        );
        for sheep in self.0.iter() {
            debug!("checking safety of\n{}", sheep);
            if Self::is_safe(sheep, edges, safe_target, dim, max_finite_value) {
                debug!("safe");
                new_sheeps.insert(sheep);
            } else {
                changed = true;
                let safe = Self::get_intersection_with_safe_ideal(
                    sheep,
                    edges,
                    safe_target,
                    max_finite_value,
                );
                debug!("restricted to\n{}", safe);
                for other_sheep in safe.sheeps() {
                    new_sheeps.insert(other_sheep);
                }
            }
        }
        if changed {
            new_sheeps.minimize();
            self.0 = new_sheeps.0;
            debug!("new ideal\n{}", self);
        }
        changed
    }

    /// Compute the safe-pre-image of the ideal by a graph.
    /// Unsafe is:
    /// - either putting some weight on a node with no successor
    /// - or taking the risk that the successor configuration is not in the ideal
    ///
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
    pub(crate) fn safe_pre_image(
        &self,
        edges: &crate::graph::Graph,
        maximal_finite_coordinate: u16,
    ) -> Ideal {
        debug!("safe_pre_image\nself\n{}\nedges\n{}", self, edges);
        let dim = edges.dim();
        if dim == 0 || self.is_empty() {
            return Ideal::new();
        }
        //compute for every i whether omega should be allowed at i,
        //this is the case iff there exists a sheep in the ideal such that
        //on that coordinate the non-empty set of successors all lead to omega
        let is_omega_possible = (0..dim)
            .map(|i| {
                let succ = edges.get_successors(i);
                !succ.is_empty() && self.0.iter().any(|sheep| sheep.all_omega(&succ))
            })
            .collect::<Vec<_>>();

        //compute for every j the maximal finite coef appearing at index j, if exists
        //omega are turned to 1
        let max_finite_coordsj: Vec<u16> = (0..dim)
            .map(|j: usize| {
                self.0
                    .iter()
                    .map(|sheep| match sheep.get(j) {
                        Coef::Omega => maximal_finite_coordinate,
                        //if we can really send omega, this will be managed by is_omega_possible
                        Coef::Value(c) => c,
                    })
                    .max()
                    .unwrap() //non-empty
            })
            .collect::<Vec<_>>();

        let max_finite_coordsi = (0..dim)
            .map(|i| {
                {
                    edges
                        .get_successors(i)
                        .iter()
                        .map(|&j| std::cmp::min(maximal_finite_coordinate, max_finite_coordsj[j]))
                        .max()
                        .unwrap_or(0)
                }
            })
            .collect::<Vec<_>>();

        //println!("preimage of\n{}\n by\n{}\n", self, edges);

        let possible_coefs = (0..dim)
            .map(|i| {
                match (
                    max_finite_coordsi.get(i).unwrap(),
                    is_omega_possible.get(i).unwrap(),
                ) {
                    (0, false) => vec![coef::Coef::Value(0)],
                    (0, true) => vec![OMEGA],
                    (&c, false) => vec![coef::Coef::Value(c)],
                    (&c, true) => vec![OMEGA, coef::Coef::Value(c)],
                }
            })
            .collect::<Vec<_>>();
        //println!("max_finite_coords: {:?}\n", max_finite_coordsi);
        //println!("is_omega_possible: {:?}\n", is_omega_possible);
        //println!("possible_coefs: {:?}\n", possible_coefs);

        let mut result = Ideal::new();
        let candidates = POSSIBLE_COEFS_CACHE.lock().unwrap().get(possible_coefs);
        candidates
            .par_iter()
            .filter(|&candidate| {
                self.is_safe_with_roundup(candidate, edges, maximal_finite_coordinate)
            })
            .collect::<HashSet<_>>()
            .iter()
            .for_each(|c| {
                result.insert(c);
            });
        result.minimize();
        //println!("result {}\n", result);
        result
    }

    /* naive exponential impl of  get_intersection_with_safe_ideal*/
    fn get_intersection_with_safe_ideal(
        sheep: &Sheep,
        edges: &crate::graph::Graph,
        safe_target: &Ideal,
        maximal_finite_value: u16,
    ) -> Ideal {
        /*
        println!(
            "get_intersection_with_safe_ideal\nsheep: {}\nsafe_target\n{}\nedges\n{}",
            sheep, safe_target, edges
        ); */
        let mut result = Ideal::new();
        let mut to_process: VecDeque<Sheep> = vec![sheep.clone()].into_iter().collect();
        let mut processed = HashSet::<Sheep>::new();
        while !to_process.is_empty() {
            let flow = to_process.pop_front().unwrap();
            //print!("Processing {}...", flow);
            if result.contains(&flow) {
                //println!("...already included");
                continue;
            }
            if processed.contains(&flow) {
                //println!("...already processed");
                continue;
            }
            processed.insert(flow.clone());
            if Self::is_safe(sheep, edges, safe_target, sheep.len(), maximal_finite_value) {
                //println!("...safe");
                result.insert(sheep);
            } else {
                //println!("...unsafe");
                flow.iter().enumerate().for_each(|(i, &ci)| {
                    if ci != C0 {
                        let smaller = flow.clone_and_decrease(i, maximal_finite_value);
                        if !processed.contains(&smaller) {
                            //println!("adding smaller {} to queue", smaller);
                            to_process.push_back(smaller);
                        }
                    }
                });
            }
        }
        result.minimize();
        result
    }

    #[allow(dead_code)]
    //below is a sad story: an optimized version of safe_pre_image which is extremely slow
    fn safe_pre_image_from(
        &self,
        candidate: &Sheep,
        edges: &crate::graph::Graph,
        accumulator: &mut Ideal,
        maximal_finite_coordinate: u16,
    ) {
        if accumulator.contains(candidate) {
            //println!("{} already in ideal", candidate);
            return;
        }
        if self.is_safe_with_roundup(candidate, edges, maximal_finite_coordinate) {
            //println!("{} inserted", candidate);
            accumulator.insert(candidate);
            return;
        }
        //println!("{} refined", candidate);
        let mut candidate_copy = candidate.clone();
        for i in 0..candidate.len() {
            let ci = candidate.get(i);
            if ci == C0 || ci == OMEGA {
                continue;
            }
            if let Coef::Value(c) = ci {
                let mut c = c - 1;
                loop {
                    if c <= 2 {
                        candidate_copy.set(i, Coef::Value(c));
                        self.safe_pre_image_from(
                            &candidate_copy,
                            edges,
                            accumulator,
                            maximal_finite_coordinate,
                        );
                        candidate_copy.set(i, ci);
                        break;
                    } else {
                        candidate_copy.set(i, Coef::Value(c / 2));
                        if !self.is_safe_with_roundup(
                            &candidate_copy,
                            edges,
                            maximal_finite_coordinate,
                        ) {
                            c /= 2;
                        } else {
                            accumulator.insert(&candidate_copy);
                            candidate_copy.set(i, Coef::Value(c));
                            self.safe_pre_image_from(
                                &candidate_copy,
                                edges,
                                accumulator,
                                maximal_finite_coordinate,
                            );
                            candidate_copy.set(i, ci);
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Check whether it is safe to play the graph in  candidate, in the sense that it ensures
    /// the next configuration belongs to the ideal.
    /// Unsafe is:
    /// - either putting some weight on a node with no successor
    /// - or taking the risk that the successor configuration is not in the ideal.
    ///
    /// There is a roundup operation: any constant larger than the dimension appearing in a successor configuration
    /// is considered as omega.
    ///
    fn is_safe_with_roundup(
        &self,
        candidate: &Sheep,
        edges: &crate::graph::Graph,
        maximal_finite_coordinate: u16,
    ) -> bool {
        let dim = edges.dim() as usize;

        //if we lose some sheep, forget about it
        let lose_sheep =
            (0..dim).any(|i| candidate.get(i) != C0 && edges.get_successors(i).is_empty());
        if lose_sheep {
            return false;
        }

        let image: Ideal = Self::get_image(dim, candidate, edges, maximal_finite_coordinate);
        //println!("image\n{}", &image);
        let answer = image.sheeps().all(|x| self.contains(x));
        answer
    }

    /// Remove from the ideal any element strictly smaller than another.
    /// The method is used in the solver to keep the size of the representation small.
    pub fn minimize(&mut self) -> bool {
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

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn get_image(
        dim: usize,
        dom: &Sheep,
        edges: &crate::graph::Graph,
        max_finite_value: u16,
    ) -> Ideal {
        let mut ideal = Ideal::new();
        let choices = (0..dom.len())
            .map(|index| get_choices(dim, dom.get(index), edges.get_successors(index)))
            .collect::<Vec<_>>();
        for im in choices
            .iter()
            .multi_cartesian_product()
            .map(|x| {
                let mut result = Sheep::new(dim, C0);
                for s in x {
                    result.add_other(s);
                }
                /*
                less efficient
                  x.into_iter()
                      .fold(Sheep::new(dim, C0), |sum, x| &sum + x)
                      .sum::<&Sheep>().round_up(max_finite_value)
                      */
                result.round_up(max_finite_value)
            })
            .collect::<Vec<_>>()
        {
            ideal.insert(&im);
        }
        ideal
    }

    pub(crate) fn round_down(&mut self, upper_bound: u16, dim: usize) {
        let to_remove: Vec<Sheep> = self
            .0
            .iter()
            .filter(|s| s.some_finite_coordinate_is_larger_than(upper_bound))
            .cloned()
            .collect();
        for mut sheep in to_remove {
            self.0.remove(&sheep);
            sheep.round_down(upper_bound, dim);
            self.0.insert(sheep);
        }
    }

    // create a CSV representation of this ideal
    pub fn as_csv(&self) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        for s in &self.0 {
            lines.push(s.as_csv());
        }
        lines
    }

    fn is_safe(
        sheep: &Sheep,
        edges: &crate::graph::Graph,
        safe_target: &Ideal,
        dim: usize,
        max_finite_value: u16,
    ) -> bool {
        let image: Ideal = Self::get_image(dim, sheep, edges, max_finite_value);
        let result = image.sheeps().all(|other| safe_target.contains(other));
        result
    }
}

#[cached]
fn get_choices(dim: usize, value: Coef, successors: Vec<usize>) -> Vec<Sheep> {
    //println!("get_choices({}, {:?}, {:?})", dim, value, successors);
    //assert!(value == OMEGA || value <= Coef::Value(dim as u16));
    match value {
        C0 => vec![Sheep::new(dim, C0)],
        OMEGA => {
            let mut base: Vec<Coef> = vec![C0; dim];
            for succ in successors {
                base[succ] = OMEGA;
            }
            vec![Sheep::from_vec(base)]
        }
        Coef::Value(c) => {
            let transports: Vec<Vec<u16>> = partitions::get_transports(c, successors.len());
            let mut result: Vec<Sheep> = Vec::new();
            for transport in transports {
                let mut vec = vec![C0; dim];
                for succ_index in 0..successors.len() {
                    vec[successors[succ_index]] = Coef::Value(transport[succ_index]);
                }
                result.push(Sheep::from_vec(vec));
            }
            result
        }
    }
}

impl fmt::Display for Ideal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            writeln!(f, "empty ideal")
        } else {
            let mut vec: Vec<String> = self.0.iter().map(|x| x.to_string()).collect();
            vec.sort();
            writeln!(f, "\t{}", vec.join("\n\t"))
        }
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

    #[test]
    fn restrict_to2() {
        let mut ideal0 = Ideal::from_vecs(&[&[C0, C1, C2, OMEGA], &[OMEGA, C2, C1, C0]]);
        let empty = Ideal::from_vecs(&[]);

        assert!(empty.is_empty());
        let changed0 = ideal0.restrict_to(&empty);
        assert!(changed0);
        assert!(ideal0.is_empty());
    }

    //test issafe
    #[test]
    fn is_safe() {
        let dim = 3;
        let edges = crate::graph::Graph::from_vec(dim, vec![(0, 1), (0, 2)]);
        let ideal = Ideal::from_vecs(&[&[C0, C1, C0], &[C0, C0, C1]]);
        let candidate = Sheep::from_vec(vec![C1, C0, C0]);
        assert!(ideal.is_safe_with_roundup(&candidate, &edges, dim as u16));
    }

    #[test]
    fn is_safe2() {
        let dim = 3;
        let c4 = Coef::Value(4);
        let edges = crate::graph::Graph::from_vec(dim, vec![(0, 1), (0, 2)]);
        let ideal = Ideal::from_vecs(&[&[C0, c4, C0], &[C0, C0, c4]]);
        let candidate = Sheep::from_vec(vec![c4, C0, C0]);
        assert!(!ideal.is_safe_with_roundup(&candidate, &edges, dim as u16));
    }

    #[test]
    fn is_safe3() {
        let dim = 3;
        let c3 = Coef::Value(3);
        let edges = crate::graph::Graph::from_vec(dim, vec![(0, 1), (0, 2)]);
        let ideal = Ideal::from_vecs(&[&[C0, c3, C0], &[C0, C2, C1], &[C0, C1, C2], &[C0, C0, c3]]);
        let candidate = Sheep::from_vec(vec![c3, C0, C0]);
        assert!(ideal.is_safe_with_roundup(&candidate, &edges, dim as u16));
    }

    #[test]
    fn is_not_safe() {
        let dim = 3;
        let c3 = Coef::Value(3);
        let c4 = Coef::Value(4);
        let edges = crate::graph::Graph::from_vec(3, vec![(0, 1), (0, 2)]);
        let ideal = Ideal::from_vecs(&[&[C0, c3, C0], &[C0, C0, c3]]);
        let candidate = Sheep::from_vec(vec![c4, C0, C0]);
        assert!(!ideal.is_safe_with_roundup(&candidate, &edges, dim as u16));
    }

    #[test]
    fn pre_image1() {
        let dim = 4;
        let edges = crate::graph::Graph::from_vec(
            dim,
            vec![(0, 0), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)],
        );
        let ideal0 = Ideal::from_vecs(&[&[C0, C1, C2, OMEGA]]);

        let pre_image0 = ideal0.safe_pre_image(&edges, dim as u16);
        assert_eq!(
            pre_image0,
            Ideal::from_vecs(&[&[C0, C1, C1, OMEGA], &[C0, C0, C2, OMEGA]]),
        );
    }

    #[test]
    fn pre_image1bis() {
        let dim = 4;
        let edges = crate::graph::Graph::from_vec(
            dim,
            vec![(0, 0), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)],
        );
        let ideal1 = Ideal::from_vecs(&[&[OMEGA, C1, C2, OMEGA], &[OMEGA, C2, C1, OMEGA]]);
        let pre_image1 = ideal1.safe_pre_image(&edges, dim as u16);
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
        let edges = crate::graph::Graph::from_vec(3, vec![(0, 1), (0, 2)]);
        let ideal0 = Ideal::from_vecs(&[&[C0, C0, OMEGA], &[C0, OMEGA, C0]]);
        let pre_image0 = ideal0.safe_pre_image(&edges, 3);
        assert_eq!(pre_image0, Ideal::from_vecs(&[&[C1, C0, C0]]));
    }

    #[test]
    fn pre_image3() {
        let dim = 4;
        let edges = crate::graph::Graph::from_vec(dim, vec![(2, 3)]);
        let ideal0 = Ideal::from_vecs(&[
            &[C0, C0, C0, OMEGA],
            &[C0, C0, OMEGA, C0],
            &[C0, OMEGA, C0, C0],
            &[OMEGA, C0, C0, C0],
        ]);
        let pre_image0 = ideal0.safe_pre_image(&edges, dim as u16);
        assert_eq!(pre_image0, Ideal::from_vecs(&[&[C0, C0, OMEGA, C0]]));
    }

    #[test]
    fn pre_image4() {
        let dim = 6;
        let ideal0 = Ideal::from_vecs(&[
            &[OMEGA, OMEGA, C0, OMEGA, OMEGA, C0],
            &[OMEGA, OMEGA, OMEGA, C0, OMEGA, C0],
        ]);
        let edges = crate::graph::Graph::from_vec(
            dim,
            vec![
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (2, 4),
                (3, 5),
                (4, 4),
                (5, 5),
            ],
        );
        let pre_image0 = ideal0.safe_pre_image(&edges, dim as u16);
        assert_eq!(
            pre_image0,
            Ideal::from_vecs(&[&[OMEGA, OMEGA, OMEGA, C0, OMEGA, C0]])
        );
    }

    #[test]
    fn pre_image5() {
        let dim = 6;

        /*preimage of
               ( ω , ω , _ , ω , ω , _ )
               ( ω , ω , ω , _ , ω , _ )
        by
                (0, 0)
                (1, 2)
                (1, 3)
                (3, 4)
                (2, 5)
                (4, 4)
                (5, 5)
        */

        let ideal0 = Ideal::from_vecs(&[
            &[OMEGA, OMEGA, C0, OMEGA, OMEGA, C0],
            &[OMEGA, OMEGA, OMEGA, C0, OMEGA, C0],
        ]);
        let edges = crate::graph::Graph::from_vec(
            6,
            vec![(0, 0), (1, 2), (1, 3), (3, 4), (2, 5), (4, 4), (5, 5)],
        );
        let pre_image0 = ideal0.safe_pre_image(&edges, dim as u16);
        assert_eq!(
            pre_image0,
            Ideal::from_vecs(&[&[OMEGA, C1, C0, OMEGA, OMEGA, C0]])
        );
    }

    #[test]
    fn is_safe6() {
        let dim = 5;
        let c5 = Coef::Value(5);
        let edges = crate::graph::Graph::from_vec(dim, vec![(0, 1), (0, 2), (0, 3)]);
        let ideal = Ideal::from_vecs(&[
            &[C0, OMEGA, OMEGA, C0, OMEGA],
            &[C0, C0, OMEGA, OMEGA, OMEGA],
            &[C0, OMEGA, C0, OMEGA, OMEGA],
        ]);
        let candidate = Sheep::from_vec(vec![c5, C0, C0, C0, C0]);
        assert!(!ideal.is_safe_with_roundup(&candidate, &edges, dim as u16));
    }

    #[test]
    fn pre_image6() {
        let dim = 5;
        /*preimage of
               ( _ , _ , _ , ω , _ )
               ( _ , _ , ω , _ , ω )
               ( _ , ω , _ , _ , ω )
               ( _ , ω , ω , _ , _ )
               ( ω , _ , _ , _ , _ )
        by
               (0, 1)
               (0, 2)
               (0, 4)
        */
        let ideal0 = Ideal::from_vecs(&[
            &[C0, C0, C0, OMEGA, C0],
            &[C0, C0, OMEGA, C0, OMEGA],
            &[C0, OMEGA, C0, C0, OMEGA],
            &[C0, OMEGA, OMEGA, C0, C0],
            &[OMEGA, C0, C0, C0, C0],
        ]);
        let edges = crate::graph::Graph::from_vec(dim, vec![(0, 1), (0, 2), (0, 4)]);
        let pre_image0 = ideal0.safe_pre_image(&edges, dim as u16);
        assert_eq!(pre_image0, Ideal::from_vecs(&[&[C2, C0, C0, C0, C0]]));
    }
}
