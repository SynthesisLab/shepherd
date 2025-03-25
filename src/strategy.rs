use crate::coef::OMEGA;
use crate::graph::Graph;
use crate::ideal::Ideal;
use crate::nfa;
use crate::sheep::Sheep;

use std::collections::HashMap;
use std::fmt;

/// A strategy is a map from letters to ideals, possibly empty.
/// All non-empty ideals have the same dimension, this is the number of states of the (complete) nfa.
/// The ideal associated to a letter represents the set of configurations where the strategy can non-deterministically play the letter.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Strategy(HashMap<nfa::Letter, Ideal>);

impl Strategy {
    pub fn get_maximal_strategy(dim: usize, letters: &[&str]) -> Self {
        let maximal_ideal = Ideal::from_vecs(&[&vec![OMEGA; dim]]);
        Strategy(
            letters
                .iter()
                .map(|&l| (l.to_string(), maximal_ideal.clone()))
                .collect(),
        )
    }

    pub fn is_defined_on(&self, source: &Sheep) -> bool {
        self.0.values().any(|ideal| ideal.contains(source))
    }

    pub(crate) fn restrict_to(
        &mut self,
        safe: Ideal,
        edges_per_letter: &HashMap<nfa::Letter, Graph>,
    ) -> bool {
        let mut result = false;
        for (a, ideal) in self.0.iter_mut() {
            let edges = edges_per_letter.get(a).unwrap();
            //compute the subset of safe which gurantees to stay in safe when playing the letter
            let very_safe = safe.pre_image(edges);
            result |= ideal.restrict_to(&very_safe);
        }
        result
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&nfa::Letter, &Ideal)> {
        self.0.iter()
    }
}

impl fmt::Display for Strategy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let vec: Vec<String> = self
            .0
            .iter()
            .map(|x| {
                format!(
                    "Play action '{}' in the downward-closure of\n{}\n",
                    x.0, x.1
                )
            })
            .collect();
        write!(f, "{}", vec.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sheep::Sheep;

    #[test]
    fn test_strategy() {
        let dim = 2;
        let letters = ["a", "b"];
        let strategy = Strategy::get_maximal_strategy(dim, &letters);
        let source = Sheep::new(dim, OMEGA);
        assert!(strategy.is_defined_on(&source));
        assert_eq!(
            strategy.0,
            HashMap::from([
                ('a'.to_string(), Ideal::from_vecs(&[&[OMEGA, OMEGA]])),
                ('b'.to_string(), Ideal::from_vecs(&[&[OMEGA, OMEGA]]))
            ])
        );
    }

    //test restrict to_ideal
}
