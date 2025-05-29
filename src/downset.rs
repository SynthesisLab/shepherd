use crate::coef::{coef, Coef, C0, OMEGA};
use crate::ideal::Ideal;
use crate::partitions;
use cached::proc_macro::cached;
use itertools::Itertools;
use log::{debug, trace, warn};
use std::collections::VecDeque;
use std::fmt;
use std::{collections::HashSet, vec::Vec};

/*
A downset represents a downward closed set of vectors in N^S.
It is represented as a set of ideal, all have the same dimension,
and the downward-closed set is the union of downard-closure of those ideal.

Several heuristics are used in order to keep the size of the set small:
* a call to 'insertion' of a ideal which is already contained in the downward-closed set has no effect
* a call to 'minimize' removes configurations which are covered by others

The method 'restrict_to' computes the intersection of the downward-closed set with another ideal.
The method 'pre_image' computes the pre-image of an ideal by a graph.
The method 'is_safe' checks whether it is safe to play a configuration w.r. to the graph, in the sense that it ensures the next configuration belongs to the downward-closed set.

 */
#[derive(Clone, Eq, Debug)]
pub struct DownSet(HashSet<Ideal>);

impl PartialEq for DownSet {
    fn eq(&self, other: &Self) -> bool {
        self.is_contained_in(other) && other.is_contained_in(self)
    }
}

type CoefsCollection = Vec<Vec<Coef>>;

/**
 * every vector comes in order omega / 0 / c+1 / 2 / 1
 */
fn expand_finite_downward_closure(
    maximal_finite_coef: &Vec<u8>,
    is_omega_sometimes_possible: &Vec<bool>,
    is_omega_always_possible: &Vec<bool>,
) -> CoefsCollection {
    trace!(
        "expand_finite_downward_closure maximal_finite_coef {:?} is_omega_sometimes_possible {:?} is_omega_always_possible {:?})",
        maximal_finite_coef,
        is_omega_sometimes_possible,
        is_omega_always_possible
    );
    assert!(maximal_finite_coef.len() == is_omega_sometimes_possible.len());
    assert!(maximal_finite_coef.len() == is_omega_always_possible.len());
    maximal_finite_coef
        .iter()
        .enumerate()
        .map(|(i, &coef)| {
            let is_omega_sometimes = is_omega_sometimes_possible[i];
            let is_omega_always = is_omega_always_possible[i];
            match (is_omega_always, is_omega_sometimes, coef) {
                (true, _, _) => vec![OMEGA],
                (false, true, _) => vec![C0, OMEGA],
                (false, false, c) => std::iter::once(C0)
                    .chain((1..c + 1).map(Coef::Value).rev())
                    .collect(),
            }
        })
        .collect()
}

impl DownSet {
    /// Create an empty downset.
    fn new() -> Self {
        DownSet(HashSet::new())
    }

    /// Create a downset from a vector of ideals.
    pub fn from_vec(w: &[Ideal]) -> Self {
        DownSet(w.iter().cloned().collect())
    }

    /// Create a downset from a vector of vectors of coefficients.
    /// The method is used in the tests.
    #[allow(dead_code)]
    pub fn from_vecs(w: &[&[Coef]]) -> Self {
        DownSet(w.iter().map(|&v| Ideal::from_vec(v.to_vec())).collect())
    }

    /// Check if an ideal is included in the downward-closed set.
    pub fn contains(&self, source: &Ideal) -> bool {
        self.0.iter().any(|x| source <= x)
    }

    /// Check if the downset is contained in another downset.
    pub fn is_contained_in(&self, other: &DownSet) -> bool {
        self.0.iter().all(|x| other.contains(x))
    }

    /// Insert an ideal in the downward-closed set.
    /// The method returns true if the downset has changed, and false if the ideal was already in the downset.
    pub fn insert(&mut self, ideal: &Ideal) -> bool {
        if self.0.contains(ideal) {
            false
        } else {
            self.0.insert(ideal.clone());
            true
        }
    }

    /// Get an iterator over the ideals of the downset.
    pub fn ideals(&self) -> impl Iterator<Item = &Ideal> {
        self.0.iter()
    }

    /// Compute the intersection of the downset set with another ideal.
    /// The method returns true if the downward-closed set has changed.
    /// The method is used in the solver to restrict the set of possible configurations.
    ///
    /// # Examples
    /// ```
    /// use shepherd::coef::{C0, C1, C2, OMEGA};
    /// use shepherd::downset::DownSet;
    /// let mut downset0 = DownSet::from_vecs(&[&[C0, C1, C2, OMEGA], &[OMEGA, C2, C1, C0]]);
    /// let mut downset1 = DownSet::from_vecs(&[&[OMEGA, C1, C2, OMEGA], &[OMEGA, C2, C1, OMEGA]]);
    /// let downset2 = DownSet::from_vecs(&[&[C1, OMEGA, C1, C2], &[C2, OMEGA, C1, C1]]);
    /// let downset0original = downset0.clone();
    /// let changed0 = downset0.restrict_to(&downset1);
    /// assert!(!changed0);
    /// assert_eq!(downset0, downset0original);
    ///
    /// let downset1original = downset1.clone();
    /// let changed1 = downset1.restrict_to(&downset2);
    /// assert!(changed1);
    /// assert!(downset1 != downset1original);
    /// assert_eq!(downset1, DownSet::from_vecs(&[&[C2, C2, C1, C1], &[C1, C2, C1, C2]]));
    /// ```
    pub fn restrict_to(&mut self, other: &DownSet) -> bool {
        let mut changed = false;
        let mut new_ideals = DownSet::new();
        for ideal in self.0.iter() {
            if other.contains(ideal) {
                new_ideals.insert(ideal);
            } else {
                changed = true;
                for other_ideal in &other.0 {
                    new_ideals.insert(&Ideal::intersection(ideal, other_ideal));
                }
            }
        }
        if changed {
            new_ideals.minimize();
            self.0 = new_ideals.0;
        }
        changed
    }

    #[allow(dead_code)]
    pub fn restrict_to_preimage_of(
        &mut self,
        safe_target: &DownSet,
        edges: &crate::graph::Graph,
        dim: usize,
        max_finite_value: coef,
    ) -> bool {
        let mut changed = false;
        let mut new_ideals = DownSet::new();
        debug!(
            "restrict_to_preimage_of\ndim: {}\nmax_finite_value: {}\nself\n{}\nsafe_target\n{}\nedges\n{}\n",
            dim, max_finite_value, self, safe_target, edges
        );
        for ideal in self.0.iter() {
            debug!("checking safety of\n{}", ideal);
            if Self::is_safe(ideal, edges, safe_target, dim, max_finite_value) {
                debug!("safe");
                new_ideals.insert(ideal);
            } else {
                changed = true;
                let safe = Self::safe_post(ideal, edges, safe_target, max_finite_value);
                debug!("restricted to\n{}", safe);
                for other_ideal in safe.ideals() {
                    new_ideals.insert(other_ideal);
                }
            }
        }
        if changed {
            new_ideals.minimize();
            self.0 = new_ideals.0;
            debug!("new downset\n{}", self);
        }
        changed
    }

    /// Compute the safe-pre-image of the downward-closed set by a graph.
    /// Unsafe is:
    /// - either putting some weight on a node with no successor
    /// - or taking the risk that the successor configuration is not in the downward-closed set
    ///
    /// The method is used in the solver to compute the set of configurations from which it is safe to play an action.
    /// The method returns the set of configurations which are safe to play.
    ///
    /// # Examples
    /// ```
    /// use shepherd::coef::{coef, C0, C1, C2, OMEGA};
    /// use shepherd::downset::DownSet;
    /// use shepherd::graph::Graph;
    /// let dim = 4;
    /// let edges = Graph::from_vec(dim, vec![(0, 0), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]);
    /// let downset1 = DownSet::from_vecs(&[&[OMEGA, C1, C2, OMEGA], &[OMEGA, C2, C1, OMEGA]]);
    /// let pre_image1 = downset1.safe_pre_image(&edges, dim as coef);
    /// assert_eq!(
    ///    pre_image1,
    ///    DownSet::from_vecs(&[
    ///        &[OMEGA, C2, C0, OMEGA],
    ///        &[OMEGA, C0, C2, OMEGA],
    ///        &[OMEGA, C1, C1, OMEGA]
    ///    ])
    /// );
    /// ```
    ///
    /// ```
    /// use shepherd::downset::DownSet;
    /// use shepherd::coef::{coef, C0, C1, C2, OMEGA};
    /// use shepherd::graph::Graph;
    /// let dim = 4;
    /// let edges = Graph::from_vec(dim, vec![(0, 0), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]);
    /// let downset0 = DownSet::from_vecs(&[&[C0, C1, C2, OMEGA]]);
    /// let pre_image0 = downset0.safe_pre_image(&edges, dim as coef);
    /// assert_eq!(
    ///     pre_image0,
    ///        DownSet::from_vecs(&[&[C0, C1, C1, OMEGA], &[C0, C0, C2, OMEGA]]),
    /// );
    /// ```
    pub fn safe_pre_image(
        &self,
        edges: &crate::graph::Graph,
        maximal_finite_coordinate: coef,
    ) -> DownSet {
        debug!("safe_pre_image\nself\n{}\nedges\n{}", self, edges);
        let dim = edges.dim();
        if dim == 0 || self.is_empty() {
            return DownSet::new();
        }
        //compute for every i whether omega should be allowed at i,
        //this is the case iff there exists a ideal in the downward-closed set such that
        //on that coordinate the non-empty set of successors all lead to omega
        let is_omega_sometimes_possible = (0..dim)
            .map(|i| {
                let succ = edges.get_successors(i);
                !succ.is_empty() && self.0.iter().any(|ideal| ideal.all_omega(&succ))
            })
            .collect::<Vec<_>>();
        let is_omega_always_possible = (0..dim)
            .map(|i| {
                let succ = edges.get_successors(i);
                !succ.is_empty() && self.0.iter().all(|ideal| ideal.all_omega(&succ))
            })
            .collect::<Vec<_>>();

        //compute for every j the maximal finite coef appearing at index j, if exists
        //omega are turned to 1
        let max_finite_coords_post: Vec<coef> = (0..dim)
            .map(|j: usize| {
                self.0
                    .iter()
                    .map(|ideal| match ideal.get(j) {
                        Coef::Omega => maximal_finite_coordinate,
                        //if we can really send omega, this will be managed by is_omega_possible
                        Coef::Value(c) => c,
                    })
                    .max()
                    .unwrap() //non-empty
            })
            .collect::<Vec<_>>();

        let max_finite_coords_pre = (0..dim)
            .map(|i| {
                {
                    edges
                        .get_successors(i)
                        .iter()
                        .map(|&j| {
                            std::cmp::min(maximal_finite_coordinate, max_finite_coords_post[j])
                        })
                        .min()
                        .unwrap_or(0)
                }
            })
            .collect::<Vec<_>>();

        trace!("preimage of\n{}\n by\n{}\n", self, edges);
        trace!("max_finite_coords: {:?}\n", max_finite_coords_pre);
        trace!(
            "is_omega_sometimes_possible: {:?}\n",
            is_omega_sometimes_possible
        );
        trace!("is_omega_always_possible: {:?}\n", is_omega_always_possible);

        let all_possible_coefs: CoefsCollection = expand_finite_downward_closure(
            &max_finite_coords_pre,
            &is_omega_sometimes_possible,
            &is_omega_always_possible,
        );

        let max_iteration_nb = all_possible_coefs.iter().fold(1, |acc, l| acc * l.len());
        trace!("max_iteration_nb: {:?}\n", max_iteration_nb);
        if max_iteration_nb > 10_000_000 {
            warn!("iterating over a potentially very large number of possible ideals (up to {}), will possibly never terminate", max_iteration_nb);
        } else {
            debug!("iterating over up to {} possible ideals", max_iteration_nb);
        }

        let mut result = DownSet::new();
        let mut iterator: Vec<usize> = DownSet::get_initial_iterator(&all_possible_coefs);
        let mut is_not_over = true;
        while is_not_over {
            let coordinates = iterator
                .iter()
                .enumerate()
                .map(|(i, &j)| all_possible_coefs[i][j])
                .collect();
            let candidate = Ideal::from_vec(coordinates);
            trace!("checking candidate {} for safe preimage", candidate);
            let is_safe = self.is_safe_with_roundup(&candidate, edges, maximal_finite_coordinate);
            if is_safe {
                trace!("{} is safe", candidate);
                result.insert(&candidate);
            } else {
                trace!("{} is unsafe", candidate);
            }
            is_not_over = DownSet::get_next_iterator(&mut iterator, &all_possible_coefs, is_safe);
        }

        trace!("minimizing result");
        result.minimize();
        trace!("result {}\n", result);
        result
    }

    fn get_initial_iterator(all_possible_coefs: &CoefsCollection) -> Vec<usize> {
        assert!(all_possible_coefs.iter().all(|l| !l.is_empty()));
        all_possible_coefs.iter().map(|_l| 0).collect()
    }
    fn get_next_iterator(
        iterator: &mut [usize],
        all_possible_coefs: &CoefsCollection,
        all_below_current_are_safe: bool,
    ) -> bool {
        assert!(iterator.len() == all_possible_coefs.len());
        if all_below_current_are_safe {
            //l'idéal actuel est gagant donc tous les idéaux plus petits également
            //on va chercherun itérateur qui n'est aps plus petit
            DownSet::get_next_not_below_current(iterator, all_possible_coefs)
        } else {
            let mut non_zero = false;
            for i in (0..iterator.len()).rev() {
                if all_possible_coefs[i].len() == 1 {
                    continue; //only one coef at this index
                }
                non_zero |= iterator[i] > 0;
                if (iterator[i] == 0 && non_zero)
                    || (0 < iterator[i] && iterator[i] < all_possible_coefs[i].len() - 1)
                {
                    iterator[i] += 1;
                    return true;
                }
                iterator[i] = 0;
            }
            false
        }
    }
    fn get_next_not_below_current(
        iterator: &mut [usize],
        all_possible_coefs: &CoefsCollection,
    ) -> bool {
        assert!(iterator.len() == all_possible_coefs.len());
        for i in (0..iterator.len()).rev() {
            if all_possible_coefs[i].len() == 1 {
                continue; //only one coef at this index
            }
            if iterator[i] == 1 {
                //on a déjà mis le 1, on ne peut pas incrémenter
                continue;
            }
            if iterator[i] == 0 {
                //il suffit de passer à 1 et resetter à droite
                iterator[i] = 1;
                iterator.iter_mut().skip(i + 1).for_each(|x| *x = 0);
                return true;
            }
            iterator[i] -= 1;
            iterator.iter_mut().skip(i + 1).for_each(|x| *x = 0);
            //on va incrémenter à gauche de i, dès que possible
            let mut j = i;
            while j > 0 {
                j -= 1;
                let itj1 = iterator[j];
                let len1 = all_possible_coefs[j].len() - 1;
                if len1 >= 1 && itj1 < len1 {
                    if itj1 == 0 {
                        iterator[j] = 1; //on met celui-là au max
                        iterator.iter_mut().skip(j + 1).for_each(|x| *x = 0);
                        return true;
                    } else {
                        iterator[j] += 1;
                        return true;
                    }
                }
            }
        }
        false
    }
    /* naive exponential impl of  get_intersection_with_safe_ideal*/
    fn safe_post(
        ideal: &Ideal,
        edges: &crate::graph::Graph,
        safe: &DownSet,
        maximal_finite_value: coef,
    ) -> DownSet {
        trace!(
            "get_intersection_with_safe_ideal\nideal: {}\nsafe_target\n{}\nedges\n{}",
            ideal,
            safe,
            edges
        );
        let mut result = DownSet::new();
        let mut to_process: VecDeque<Ideal> = vec![ideal.clone()].into_iter().collect();
        let mut processed = HashSet::<Ideal>::new();
        while !to_process.is_empty() {
            let flow = to_process.pop_front().unwrap();
            trace!("Processing {}...", flow);
            if result.contains(&flow) {
                trace!("...already included");
                continue;
            }
            if processed.contains(&flow) {
                trace!("...already processed");
                continue;
            }
            processed.insert(flow.clone());
            if Self::is_safe(ideal, edges, safe, ideal.dimension(), maximal_finite_value) {
                trace!("...safe");
                result.insert(ideal);
            } else {
                trace!("...unsafe");
                flow.iter().enumerate().for_each(|(i, &ci)| {
                    if ci != C0 {
                        let smaller = flow.clone_and_decrease(i, maximal_finite_value);
                        if !processed.contains(&smaller) {
                            trace!("adding smaller {} to queue", smaller);
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
        candidate: &Ideal,
        edges: &crate::graph::Graph,
        accumulator: &mut DownSet,
        maximal_finite_coordinate: coef,
    ) {
        if accumulator.contains(candidate) {
            trace!("{} already in ideal", candidate);
            return;
        }
        if self.is_safe_with_roundup(candidate, edges, maximal_finite_coordinate) {
            trace!("{} inserted", candidate);
            accumulator.insert(candidate);
            return;
        }
        trace!("{} refined", candidate);
        let mut candidate_copy = candidate.clone();
        for i in 0..candidate.dimension() {
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
    /// the next configuration belongs to the downward-closed set.
    /// Unsafe is:
    /// - either putting some weight on a node with no successor
    /// - or taking the risk that the successor configuration is not in the downward-closed set.
    ///
    /// There is a roundup operation: any constant larger than the dimension appearing in a successor configuration
    /// is considered as omega.
    ///
    fn is_safe_with_roundup(
        &self,
        candidate: &Ideal,
        edges: &crate::graph::Graph,
        maximal_finite_coordinate: coef,
    ) -> bool {
        let dim = edges.dim();

        //if we lose some ideal, forget about it
        let lose_ideal =
            (0..dim).any(|i| candidate.get(i) != C0 && edges.get_successors(i).is_empty());
        if lose_ideal {
            return false;
        }

        let image: DownSet = Self::get_image(dim, candidate, edges, maximal_finite_coordinate);
        trace!("image\n{}", &image);
        let answer = image.ideals().all(|x| self.contains(x));
        answer
    }

    /// Remove from the downward-closed set any element strictly smaller than another.
    /// The method is used in the solver to keep the size of the representation small.
    pub fn minimize(&mut self) -> bool {
        //remove from self.0 any element strictly smaller than another
        let mut changed = false;
        for ideal in self
            .0
            .iter()
            .filter(|&x| self.0.iter().any(|y| x < y))
            .cloned()
            .collect::<Vec<_>>()
        {
            changed |= self.0.remove(&ideal);
        }
        changed
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn get_image(
        dim: usize,
        dom: &Ideal,
        edges: &crate::graph::Graph,
        max_finite_value: coef,
    ) -> DownSet {
        let mut downset = DownSet::new();
        let choices = (0..dom.dimension())
            .map(|index| get_choices(dim, dom.get(index), edges.get_successors(index)))
            .collect::<Vec<_>>();
        for im in choices
            .iter()
            .multi_cartesian_product()
            .map(|x| {
                let mut result = Ideal::new(dim, C0);
                for s in x {
                    result.add_other(s);
                }
                /*
                less efficient
                  x.into_iter()
                      .fold(Ideal::new(dim, C0), |sum, x| &sum + x)
                      .sum::<&Ideal>().round_up(max_finite_value)
                      */
                result.round_up(max_finite_value)
            })
            .collect::<Vec<_>>()
        {
            downset.insert(&im);
        }
        downset
    }

    /// Removes ideal with precision >.
    pub fn round_down(&mut self, maximal_finite_value: coef, dim: usize) {
        let to_remove: Vec<Ideal> = self
            .0
            .iter()
            .filter(|s| s.some_finite_coordinate_is_larger_than(maximal_finite_value))
            .cloned()
            .collect();
        for mut ideal in to_remove {
            self.0.remove(&ideal);
            ideal.round_down(maximal_finite_value, dim);
            self.0.insert(ideal);
        }
    }

    fn is_safe(
        ideal: &Ideal,
        edges: &crate::graph::Graph,
        safe_target: &DownSet,
        dim: usize,
        max_finite_value: coef,
    ) -> bool {
        let image: DownSet = Self::get_image(dim, ideal, edges, max_finite_value);
        let result = image.ideals().all(|other| safe_target.contains(other));
        result
    }

    // create a CSV representation of this downward-closed set
    pub fn as_csv(&self) -> Vec<String> {
        let mut lines: Vec<String> = Vec::new();
        for s in &self.0 {
            lines.push(s.as_csv());
        }
        lines
    }
}

#[cached]
fn get_choices(dim: usize, value: Coef, successors: Vec<usize>) -> Vec<Ideal> {
    trace!("get_choices({}, {:?}, {:?})", dim, value, successors);
    //assert!(value == OMEGA || value <= Coef::Value(dim as coef));
    match value {
        C0 => vec![Ideal::new(dim, C0)],
        OMEGA => {
            let mut base: Vec<Coef> = vec![C0; dim];
            for succ in successors {
                base[succ] = OMEGA;
            }
            vec![Ideal::from_vec(base)]
        }
        Coef::Value(c) => {
            let transports: Vec<Vec<coef>> = partitions::get_transports(c, successors.len());
            let mut result: Vec<Ideal> = Vec::new();
            for transport in transports {
                let mut vec = vec![C0; dim];
                for succ_index in 0..successors.len() {
                    vec[successors[succ_index]] = Coef::Value(transport[succ_index]);
                }
                result.push(Ideal::from_vec(vec));
            }
            result
        }
    }
}

impl fmt::Display for DownSet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            writeln!(f, "empty downward-closed set")
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
    use crate::coef::{C0, C1, C2, C3, OMEGA};

    #[test]
    fn is_in_ideal() {
        let master_ideal = Ideal::from_vec(vec![OMEGA, OMEGA]);
        let medium_ideal = Ideal::from_vec(vec![C1, C1]);
        let ini_ideal = Ideal::from_vec(vec![C1, C0]);
        let final_ideal = Ideal::from_vec(vec![C0, C1 + C1]);

        let downset = DownSet([ini_ideal.clone(), final_ideal.clone()].into());
        assert!(downset.contains(&ini_ideal));
        assert!(downset.contains(&final_ideal));
        assert!(!downset.contains(&master_ideal));
        assert!(!downset.contains(&medium_ideal));

        let downset2 = DownSet([medium_ideal.clone()].into());
        assert!(downset2.contains(&ini_ideal));
        assert!(!downset2.contains(&final_ideal));
        assert!(!downset2.contains(&master_ideal));
        assert!(downset2.contains(&medium_ideal));
    }

    //test equality
    #[test]
    fn order() {
        let downset0 = DownSet::from_vecs(&[&[C0, C1, C2, OMEGA], &[OMEGA, C2, C1, C0]]);
        let downset1 = DownSet::from_vecs(&[&[OMEGA, C1, C2, OMEGA], &[OMEGA, C2, C1, OMEGA]]);
        let downset2 = DownSet::from_vecs(&[&[OMEGA, C2, C2, OMEGA]]);

        assert!(downset0.is_contained_in(&downset1));
        assert!(downset1.is_contained_in(&downset2));
        assert!(downset0.is_contained_in(&downset2));
    }

    #[test]
    fn restrict_to() {
        let mut downset0 = DownSet::from_vecs(&[&[C0, C1, C2, OMEGA], &[OMEGA, C2, C1, C0]]);
        let mut downset1 = DownSet::from_vecs(&[&[OMEGA, C1, C2, OMEGA], &[OMEGA, C2, C1, OMEGA]]);
        let downset2 = DownSet::from_vecs(&[&[C1, OMEGA, C1, C2], &[C2, OMEGA, C1, C1]]);

        let downset0original = downset0.clone();
        let changed0 = downset0.restrict_to(&downset1);
        assert!(!changed0);
        assert_eq!(downset0, downset0original);

        let downset1original = downset1.clone();
        let changed1 = downset1.restrict_to(&downset2);
        assert!(changed1);
        assert!(downset1 != downset1original);
        assert_eq!(
            downset1,
            DownSet::from_vecs(&[&[C2, C2, C1, C1], &[C1, C2, C1, C2]])
        );
        assert_eq!(
            downset1,
            DownSet::from_vecs(&[&[C2, C2, C1, C1], &[C1, C1, C1, C2], &[C1, C2, C1, C2]])
        );
        assert_eq!(
            downset1,
            DownSet::from_vecs(&[
                &[C1, C2, C1, C2],
                &[C2, C2, C1, C1],
                &[C1, C1, C1, C2],
                &[C2, C1, C1, C1],
            ])
        );
    }

    #[test]
    fn restrict_to2() {
        let mut downset0 = DownSet::from_vecs(&[&[C0, C1, C2, OMEGA], &[OMEGA, C2, C1, C0]]);
        let empty = DownSet::from_vecs(&[]);

        assert!(empty.is_empty());
        let changed0 = downset0.restrict_to(&empty);
        assert!(changed0);
        assert!(downset0.is_empty());
    }

    //test issafe
    #[test]
    fn is_safe() {
        let dim = 3;
        let edges = crate::graph::Graph::from_vec(dim, vec![(0, 1), (0, 2)]);
        let downset = DownSet::from_vecs(&[&[C0, C1, C0], &[C0, C0, C1]]);
        let candidate = Ideal::from_vec(vec![C1, C0, C0]);
        assert!(downset.is_safe_with_roundup(&candidate, &edges, dim as coef));
    }

    #[test]
    fn is_safe2() {
        let dim = 3;
        let c4 = Coef::Value(4);
        let edges = crate::graph::Graph::from_vec(dim, vec![(0, 1), (0, 2)]);
        let downset = DownSet::from_vecs(&[&[C0, c4, C0], &[C0, C0, c4]]);
        let candidate = Ideal::from_vec(vec![c4, C0, C0]);
        assert!(!downset.is_safe_with_roundup(&candidate, &edges, dim as coef));
    }

    #[test]
    fn is_safe3() {
        let dim = 3;
        let c3 = Coef::Value(3);
        let edges = crate::graph::Graph::from_vec(dim, vec![(0, 1), (0, 2)]);
        let downset =
            DownSet::from_vecs(&[&[C0, c3, C0], &[C0, C2, C1], &[C0, C1, C2], &[C0, C0, c3]]);
        let candidate = Ideal::from_vec(vec![c3, C0, C0]);
        assert!(downset.is_safe_with_roundup(&candidate, &edges, dim as coef));
    }

    #[test]
    fn is_not_safe() {
        let dim = 3;
        let c3 = Coef::Value(3);
        let c4 = Coef::Value(4);
        let edges = crate::graph::Graph::from_vec(3, vec![(0, 1), (0, 2)]);
        let downset = DownSet::from_vecs(&[&[C0, c3, C0], &[C0, C0, c3]]);
        let candidate = Ideal::from_vec(vec![c4, C0, C0]);
        assert!(!downset.is_safe_with_roundup(&candidate, &edges, dim as coef));
    }

    #[test]
    fn pre_image1() {
        let dim = 4;
        let edges = crate::graph::Graph::from_vec(
            dim,
            vec![(0, 0), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)],
        );
        let downset0 = DownSet::from_vecs(&[&[C0, C1, C2, OMEGA]]);

        let pre_image0 = downset0.safe_pre_image(&edges, dim as coef);
        assert_eq!(
            pre_image0,
            DownSet::from_vecs(&[&[C0, C1, C1, OMEGA], &[C0, C0, C2, OMEGA]]),
        );
    }

    #[test]
    fn pre_image1bis() {
        let dim = 4;
        let edges = crate::graph::Graph::from_vec(
            dim,
            vec![(0, 0), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)],
        );
        let downset1 = DownSet::from_vecs(&[&[OMEGA, C1, C2, OMEGA], &[OMEGA, C2, C1, OMEGA]]);
        let pre_image1 = downset1.safe_pre_image(&edges, dim as coef);
        assert_eq!(
            pre_image1,
            DownSet::from_vecs(&[
                &[OMEGA, C2, C0, OMEGA],
                &[OMEGA, C0, C2, OMEGA],
                &[OMEGA, C1, C1, OMEGA]
            ])
        );
    }

    #[test]
    fn pre_image2() {
        let edges = crate::graph::Graph::from_vec(3, vec![(0, 1), (0, 2)]);
        let downset0 = DownSet::from_vecs(&[&[C0, C0, OMEGA], &[C0, OMEGA, C0]]);
        let pre_image0 = downset0.safe_pre_image(&edges, 3);
        assert_eq!(pre_image0, DownSet::from_vecs(&[&[C1, C0, C0]]));
    }

    #[test]
    fn pre_image3() {
        let dim = 4;
        let edges = crate::graph::Graph::from_vec(dim, vec![(2, 3)]);
        let downset0 = DownSet::from_vecs(&[
            &[C0, C0, C0, OMEGA],
            &[C0, C0, OMEGA, C0],
            &[C0, OMEGA, C0, C0],
            &[OMEGA, C0, C0, C0],
        ]);
        let pre_image0 = downset0.safe_pre_image(&edges, dim as coef);
        assert_eq!(pre_image0, DownSet::from_vecs(&[&[C0, C0, OMEGA, C0]]));
    }

    #[test]
    fn pre_image4() {
        let dim = 6;
        let downset0 = DownSet::from_vecs(&[
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
        let pre_image0 = downset0.safe_pre_image(&edges, dim as coef);
        assert_eq!(
            pre_image0,
            DownSet::from_vecs(&[&[OMEGA, OMEGA, OMEGA, C0, OMEGA, C0]])
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

        let downset0 = DownSet::from_vecs(&[
            &[OMEGA, OMEGA, C0, OMEGA, OMEGA, C0],
            &[OMEGA, OMEGA, OMEGA, C0, OMEGA, C0],
        ]);
        let edges = crate::graph::Graph::from_vec(
            6,
            vec![(0, 0), (1, 2), (1, 3), (3, 4), (2, 5), (4, 4), (5, 5)],
        );
        let pre_image0 = downset0.safe_pre_image(&edges, dim as coef);
        assert_eq!(
            pre_image0,
            DownSet::from_vecs(&[&[OMEGA, C1, C0, OMEGA, OMEGA, C0]])
        );
    }

    #[test]
    fn is_safe6() {
        let dim = 5;
        let c5 = Coef::Value(5);
        let edges = crate::graph::Graph::from_vec(dim, vec![(0, 1), (0, 2), (0, 3)]);
        let downset = DownSet::from_vecs(&[
            &[C0, OMEGA, OMEGA, C0, OMEGA],
            &[C0, C0, OMEGA, OMEGA, OMEGA],
            &[C0, OMEGA, C0, OMEGA, OMEGA],
        ]);
        let candidate = Ideal::from_vec(vec![c5, C0, C0, C0, C0]);
        assert!(!downset.is_safe_with_roundup(&candidate, &edges, dim as coef));
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
        let downset0 = DownSet::from_vecs(&[
            &[C0, C0, C0, OMEGA, C0],
            &[C0, C0, OMEGA, C0, OMEGA],
            &[C0, OMEGA, C0, C0, OMEGA],
            &[C0, OMEGA, OMEGA, C0, C0],
            &[OMEGA, C0, C0, C0, C0],
        ]);
        let edges = crate::graph::Graph::from_vec(dim, vec![(0, 1), (0, 2), (0, 4)]);
        let pre_image0 = downset0.safe_pre_image(&edges, dim as coef);
        assert_eq!(pre_image0, DownSet::from_vecs(&[&[C2, C0, C0, C0, C0]]));
    }

    #[test]
    fn expand_finite_downward_closure() {
        use crate::downset::expand_finite_downward_closure;
        let expanded = expand_finite_downward_closure(
            &vec![0, 3, 1, 0],
            &vec![true, false, false, true],
            &vec![true, false, false, false],
        );
        assert_eq!(
            expanded,
            vec![
                vec![OMEGA],
                vec![C0, C3, C2, C1],
                vec![C0, C1],
                vec![C0, OMEGA],
            ]
        );
    }
}
