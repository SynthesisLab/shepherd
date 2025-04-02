use crate::coef::{coef, Coef, OMEGA};
use std::cmp::min;
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign};
use std::vec::Vec;

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Sheep(Vec<Coef>);

impl PartialOrd for Sheep {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let is_smaller_or_equal = self.0.iter().zip(other.0.iter()).all(|(x, y)| x <= y);
        let is_greater_or_equal = other.0.iter().zip(self.0.iter()).all(|(x, y)| x <= y);
        match (is_smaller_or_equal, is_greater_or_equal) {
            (true, true) => Some(std::cmp::Ordering::Equal),
            (true, false) => Some(std::cmp::Ordering::Less),
            (false, true) => Some(std::cmp::Ordering::Greater),
            (false, false) => None,
        }
    }
}

impl Add for &Sheep {
    type Output = Sheep;

    fn add(self, other: Self) -> Self::Output {
        debug_assert_eq!(self.len(), other.len());
        Sheep(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(&x, &y)| x + y)
                .collect(),
        )
    }
}

impl Add for Sheep {
    type Output = Sheep;
    fn add(self, other: Self) -> Self::Output {
        &self + &other
    }
}

impl AddAssign for Sheep {
    fn add_assign(&mut self, other: Self) {
        debug_assert_eq!(self.len(), other.len());
        for (i, x) in self.0.iter_mut().enumerate() {
            *x += other.0[i];
        }
    }
}

//will crash for empty iterators
impl Sum for Sheep {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut iter = iter;
        match iter.next() {
            None => panic!("Cannot sum up empty sheep iterator"),
            Some(mut sheep) => {
                for x in iter {
                    sheep += x;
                }
                sheep
            }
        }
    }
}

impl<'a> Sum<&'a Sheep> for Sheep {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Sheep>,
    {
        let mut iter = iter;
        match iter.next() {
            None => panic!("Cannot sum up empty sheep iterator"),
            Some(sheep) => {
                let mut result = sheep.clone();
                for x in iter {
                    result.add_other(x);
                }
                result
            }
        }
    }
}

impl Sheep {
    pub fn new(dimension: usize, val: Coef) -> Self {
        Sheep(vec![val; dimension])
    }

    pub(crate) fn from_vec(vec: Vec<Coef>) -> Sheep {
        Sheep(vec)
    }

    pub fn is_below(&self, other: &Self) -> bool {
        self.0.iter().enumerate().all(|(i, &x)| x <= other.0[i])
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn get(&self, i: usize) -> Coef {
        self.0[i]
    }

    pub(crate) fn set(&mut self, state: usize, val: Coef) {
        self.0[state] = val;
    }

    pub(crate) fn intersection(x: &Sheep, sheep: &Sheep) -> Sheep {
        debug_assert_eq!(x.len(), sheep.len());
        Sheep(
            x.0.iter()
                .zip(sheep.0.iter())
                .map(|(x, y)| min(x, y))
                .cloned()
                .collect(),
        )
    }

    #[allow(dead_code)]
    pub(crate) fn from_non_zero_coefs(
        dim: usize,
        partition: &[coef],
        predecessors: &[usize],
    ) -> Sheep {
        let mut result = vec![Coef::Value(0); dim];
        for (i, &x) in predecessors.iter().enumerate() {
            debug_assert!(x < dim);
            result[x] = Coef::Value(partition[i]);
        }
        Sheep(result)
    }

    pub(crate) fn all_omega(&self, succ: &[usize]) -> bool {
        succ.iter().all(|&i| self.get(i) == OMEGA)
    }

    pub(crate) fn round_up(&mut self, max_finite_value: coef) -> Sheep {
        Sheep(
            self.0
                .iter()
                .map(|x| x.round_up(max_finite_value))
                .collect(),
        )
    }

    pub(crate) fn round_down(&mut self, upper_bound: coef, dim: usize) {
        for i in 0..dim {
            if let Coef::Value(x) = self.get(i) {
                if x > upper_bound {
                    self.set(i, Coef::Value(upper_bound));
                }
            }
        }
    }

    pub(crate) fn some_finite_coordinate_is_larger_than(&self, upper_bound: coef) -> bool {
        self.0
            .iter()
            .any(|&x| x < OMEGA && x > Coef::Value(upper_bound))
    }

    // create a CSV representation of this sheep,
    // as comma separated values, one for each state
    pub fn as_csv(&self) -> String {
        let content = self
            .0
            .iter()
            .map(|&x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        content
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &Coef> {
        self.0.iter()
    }

    //why AddAssign does not allow adding a reference !!??
    pub fn add_other(&mut self, x: &Sheep) {
        debug_assert_eq!(self.len(), x.len());
        for i in 0..self.len() {
            self.0[i] += x.0[i];
        }
    }

    pub(crate) fn clone_and_decrease(&self, i: usize, maximal_finite_value: coef) -> Sheep {
        let mut result: Sheep = self.clone();
        let c = result.0[i];
        debug_assert!(c != Coef::Value(0));
        match c {
            Coef::Omega => {
                result.0[i] = Coef::Value(maximal_finite_value);
            }
            Coef::Value(0) => {
                panic!("Cannot decrease zero");
            }
            Coef::Value(x) => {
                result.0[i] = Coef::Value(std::cmp::min(x - 1, maximal_finite_value));
            }
        }
        result
    }
}

impl fmt::Display for Sheep {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let content = self
            .0
            .iter()
            .map(|&x| x.to_string())
            .collect::<Vec<_>>()
            .join(" , ");
        write!(f, "( {} )", content)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::coef::C0;
    use crate::coef::C1;
    use crate::coef::C2;
    use crate::coef::OMEGA;

    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    #[test]
    fn is_below() {
        let master_sheep = Sheep(vec![OMEGA, OMEGA]);
        let medium_sheep = Sheep(vec![Coef::Value(7), Coef::Value(7)]);
        let ini_sheep = Sheep(vec![OMEGA, C0]);
        let final_sheep = Sheep(vec![C0, OMEGA]);

        assert!(master_sheep <= master_sheep);
        assert!(medium_sheep <= master_sheep);
        assert!(ini_sheep <= master_sheep);
        assert!(final_sheep <= master_sheep);

        assert!(!(master_sheep <= medium_sheep));
        assert!(medium_sheep <= medium_sheep);
        assert!(!(ini_sheep <= medium_sheep));
        assert!(!(final_sheep <= medium_sheep));

        assert!(!(master_sheep <= ini_sheep));
        assert!(!(medium_sheep <= ini_sheep));
        assert!(ini_sheep <= ini_sheep);
        assert!(!(final_sheep <= ini_sheep));

        assert!(!(master_sheep <= final_sheep));
        assert!(!(medium_sheep <= final_sheep));
        assert!(!(ini_sheep <= final_sheep));
        assert!(final_sheep <= final_sheep);
    }

    #[test]
    fn min() {
        let sheep0 = Sheep::from_vec(vec![C0, C1, C2, OMEGA]);
        let sheep1 = Sheep::from_vec(vec![OMEGA, C2, C1, C0]);
        let intersect = Sheep::from_vec(vec![C0, C1, C1, C0]);
        assert_eq!(intersect, Sheep::intersection(&sheep0, &sheep1));
    }

    //from_non_zero_coefs
    #[test]
    fn from_non_zero_coefs() {
        let sheep = Sheep::from_non_zero_coefs(4, &[1, 2], &[1, 3]);
        assert_eq!(sheep, Sheep::from_vec(vec![C0, C1, C0, C2]));
    }
}
