use std::fmt;
use std::iter::Sum;
use std::ops::Add;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
pub enum Coef {
    Value(u16),
    Omega,
}
impl Coef {
    pub(crate) fn round_up(&self, max_finite_value: u16) -> Coef {
        match self {
            Coef::Value(x) if *x > max_finite_value => Coef::Omega,
            _ => *self,
        }
    }

    pub(crate) fn is_finite(&self) -> bool {
        return self != &Coef::Omega;
    }
}

pub const C0: Coef = Coef::Value(0);
#[allow(dead_code)]
pub const C1: Coef = Coef::Value(1);
#[allow(dead_code)]
pub const C2: Coef = Coef::Value(2);
#[allow(dead_code)]
pub const C3: Coef = Coef::Value(3);
pub const OMEGA: Coef = Coef::Omega;

impl Add for &Coef {
    type Output = Coef;

    fn add(self, other: Self) -> Self::Output {
        match (self, other) {
            (Coef::Omega, _) | (_, Coef::Omega) => OMEGA,
            (Coef::Value(x), Coef::Value(y)) => Coef::Value(x + y),
        }
    }
}

#[allow(clippy::op_ref)]
impl Add for Coef {
    type Output = Coef;
    fn add(self, other: Self) -> Self::Output {
        &self + &other
    }
}

impl<'a> Sum<&'a Coef> for Coef {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Coef>,
    {
        let mut iter = iter;
        iter.try_fold(0, |sum, &x| match x {
            Coef::Omega => Err(Coef::Omega),
            Coef::Value(v) => Ok(sum + v),
        })
        .map_or(Coef::Omega, Coef::Value)
    }
}

impl Sum for Coef {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        let mut iter = iter;
        iter.try_fold(0, |sum, x| match x {
            Coef::Omega => Err(Coef::Omega),
            Coef::Value(v) => Ok(sum + v),
        })
        .map_or(Coef::Omega, Coef::Value)
    }
}

impl fmt::Display for Coef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Coef::Omega => write!(f, "ω"),
            Coef::Value(0) => write!(f, "_"),
            Coef::Value(x) => write!(f, "{}", x),
        }
    }
}

//tests
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn add() {
        assert_eq!(C1 + C1, Coef::Value(2));
        assert_eq!(OMEGA + C1, OMEGA);
        assert_eq!(OMEGA + OMEGA, OMEGA);
    }

    #[test]
    fn sum() {
        let vec = [C1, C1, C1];
        assert_eq!(vec.iter().sum::<Coef>(), Coef::Value(3));
        assert_eq!(vec.iter().copied().sum::<Coef>(), Coef::Value(3));
        let vec = [C1, OMEGA, C1];
        assert_eq!(vec.iter().sum::<Coef>(), OMEGA);
        assert_eq!(vec.iter().copied().sum::<Coef>(), OMEGA);
    }

    #[test]
    fn cmp() {
        assert!(C1 < OMEGA);
        assert!(C0 < C1);
        assert!(C0 < OMEGA);
        assert!(C1 < OMEGA);
        assert!(C1 < Coef::Value(2));
    }
}
