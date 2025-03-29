use crate::coef::{Coef, OMEGA};
use std::cmp::min;
use std::fmt;
use std::iter::Sum;
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

impl Sum for Sheep {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut iter = iter;
        let mut result = iter.next().unwrap().clone();
        for x in iter {
            result = Sheep(
                result
                    .0
                    .iter()
                    .zip(x.0.iter())
                    .map(|(x, y)| x + y)
                    .collect(),
            );
        }
        result
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
        assert!(x.len() == sheep.len());
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
        partition: &[u16],
        predecessors: &[usize],
    ) -> Sheep {
        let mut result = vec![Coef::Value(0); dim];
        for (i, &x) in predecessors.iter().enumerate() {
            assert!(x < dim);
            result[x] = Coef::Value(partition[i]);
        }
        Sheep(result)
    }

    pub(crate) fn all_omega(&self, succ: &[usize]) -> bool {
        succ.iter().all(|&i| self.get(i) == OMEGA)
    }

    pub(crate) fn round_up(&mut self, max_finite_value: u16) -> Sheep {
        Sheep(
            self.0
                .iter()
                .map(|x| x.round_up(max_finite_value))
                .collect(),
        )
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
        let sheep = Sheep::from_non_zero_coefs(4, &vec![1, 2], &vec![1, 3]);
        assert_eq!(sheep, Sheep::from_vec(vec![C0, C1, C0, C2]));
    }
}
