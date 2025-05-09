use crate::coef::{coef, Coef, OMEGA};
use std::cmp::min;
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign};
use std::vec::Vec;

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct Ideal(Vec<Coef>);

impl PartialOrd for Ideal {
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

impl Add for &Ideal {
    type Output = Ideal;

    fn add(self, other: Self) -> Self::Output {
        debug_assert_eq!(self.dimension(), other.dimension());
        Ideal(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(&x, &y)| x + y)
                .collect(),
        )
    }
}

impl Add for Ideal {
    type Output = Ideal;
    fn add(self, other: Self) -> Self::Output {
        &self + &other
    }
}

impl AddAssign for Ideal {
    fn add_assign(&mut self, other: Self) {
        debug_assert_eq!(self.dimension(), other.dimension());
        for (i, x) in self.0.iter_mut().enumerate() {
            *x += other.0[i];
        }
    }
}

//will crash for empty iterators
impl Sum for Ideal {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut iter = iter;
        match iter.next() {
            None => panic!("Cannot sum up empty ideal iterator"),
            Some(mut ideal) => {
                for x in iter {
                    ideal += x;
                }
                ideal
            }
        }
    }
}

impl<'a> Sum<&'a Ideal> for Ideal {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Ideal>,
    {
        let mut iter = iter;
        match iter.next() {
            None => panic!("Cannot sum up empty ideal iterator"),
            Some(ideal) => {
                let mut result = ideal.clone();
                for x in iter {
                    result.add_other(x);
                }
                result
            }
        }
    }
}

impl Ideal {
    pub fn new(dimension: usize, val: Coef) -> Self {
        Ideal(vec![val; dimension])
    }

    pub fn from_vec(vec: Vec<Coef>) -> Ideal {
        Ideal(vec)
    }

    pub fn is_below(&self, other: &Self) -> bool {
        self.0.iter().enumerate().all(|(i, &x)| x <= other.0[i])
    }

    /// Returns the dimension of this ideal,
    /// which for us is the number of states in the NFA
    pub fn dimension(&self) -> usize {
        self.0.len()
    }

    pub fn get(&self, i: usize) -> Coef {
        self.0[i]
    }

    pub fn set(&mut self, state: usize, val: Coef) {
        self.0[state] = val;
    }

    pub fn intersection(x: &Ideal, ideal: &Ideal) -> Ideal {
        debug_assert_eq!(x.dimension(), ideal.dimension());
        Ideal(
            x.0.iter()
                .zip(ideal.0.iter())
                .map(|(x, y)| min(x, y))
                .cloned()
                .collect(),
        )
    }

    #[allow(dead_code)]
    pub fn from_non_zero_coefs(
        dim: usize,
        partition: &[coef],
        predecessors: &[usize],
    ) -> Ideal {
        let mut result = vec![Coef::Value(0); dim];
        for (i, &x) in predecessors.iter().enumerate() {
            debug_assert!(x < dim);
            result[x] = Coef::Value(partition[i]);
        }
        Ideal(result)
    }

    pub fn all_omega(&self, succ: &[usize]) -> bool {
        succ.iter().all(|&i| self.get(i) == OMEGA)
    }

    pub fn round_up(&mut self, max_finite_value: coef) -> Ideal {
        Ideal(
            self.0
                .iter()
                .map(|x| x.round_up(max_finite_value))
                .collect(),
        )
    }

    pub fn round_down(&mut self, upper_bound: coef, dim: usize) {
        for i in 0..dim {
            if let Coef::Value(x) = self.get(i) {
                if x > upper_bound {
                    self.set(i, Coef::Value(upper_bound));
                }
            }
        }
    }

    pub fn some_finite_coordinate_is_larger_than(&self, upper_bound: coef) -> bool {
        self.0
            .iter()
            .any(|&x| x < OMEGA && x > Coef::Value(upper_bound))
    }

    // create a CSV representation of this ideal,
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

    pub fn iter(&self) -> impl Iterator<Item = &Coef> {
        self.0.iter()
    }

    //why AddAssign does not allow adding a reference !!??
    pub fn add_other(&mut self, x: &Ideal) {
        debug_assert_eq!(self.dimension(), x.dimension());
        for i in 0..self.dimension() {
            self.0[i] += x.0[i];
        }
    }

    pub fn clone_and_decrease(&self, i: usize, maximal_finite_value: coef) -> Ideal {
        let mut result: Ideal = self.clone();
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

impl fmt::Display for Ideal {
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
        let master_ideal = Ideal(vec![OMEGA, OMEGA]);
        let medium_ideal = Ideal(vec![Coef::Value(7), Coef::Value(7)]);
        let ini_ideal = Ideal(vec![OMEGA, C0]);
        let final_ideal = Ideal(vec![C0, OMEGA]);

        assert!(master_ideal <= master_ideal);
        assert!(medium_ideal <= master_ideal);
        assert!(ini_ideal <= master_ideal);
        assert!(final_ideal <= master_ideal);

        assert!(!(master_ideal <= medium_ideal));
        assert!(medium_ideal <= medium_ideal);
        assert!(!(ini_ideal <= medium_ideal));
        assert!(!(final_ideal <= medium_ideal));

        assert!(!(master_ideal <= ini_ideal));
        assert!(!(medium_ideal <= ini_ideal));
        assert!(ini_ideal <= ini_ideal);
        assert!(!(final_ideal <= ini_ideal));

        assert!(!(master_ideal <= final_ideal));
        assert!(!(medium_ideal <= final_ideal));
        assert!(!(ini_ideal <= final_ideal));
        assert!(final_ideal <= final_ideal);
    }

    #[test]
    fn min() {
        let ideal0 = Ideal::from_vec(vec![C0, C1, C2, OMEGA]);
        let ideal1 = Ideal::from_vec(vec![OMEGA, C2, C1, C0]);
        let intersect = Ideal::from_vec(vec![C0, C1, C1, C0]);
        assert_eq!(intersect, Ideal::intersection(&ideal0, &ideal1));
    }

    //from_non_zero_coefs
    #[test]
    fn from_non_zero_coefs() {
        let ideal = Ideal::from_non_zero_coefs(4, &[1, 2], &[1, 3]);
        assert_eq!(ideal, Ideal::from_vec(vec![C0, C1, C0, C2]));
    }
}
