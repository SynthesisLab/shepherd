use crate::coef::{coef, OMEGA};
use crate::downset::DownSet;
use crate::graph::Graph;
use crate::ideal::Ideal;
use crate::nfa;

use std::collections::HashMap;
use std::fmt;

/// A strategy is a map from letters to downsets, possibly empty.
/// All non-empty downsets have the same dimension, this is the number of states of the (complete) nfa.
/// The downset associated to a letter represents the set of configurations where the strategy
/// can non-deterministically play the letter.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Strategy(HashMap<nfa::Letter, DownSet>);

impl Strategy {
    pub fn get_maximal_strategy(dim: usize, letters: &[&str]) -> Self {
        let maximal_downset = DownSet::from_vecs(&[&vec![OMEGA; dim]]);
        Strategy(
            letters
                .iter()
                .map(|&l| (l.to_string(), maximal_downset.clone()))
                .collect(),
        )
    }

    pub fn is_defined_on(&self, source: &Ideal) -> bool {
        self.0.values().any(|downset| downset.contains(source))
    }

    pub fn restrict_to(
        &mut self,
        safe: DownSet,
        edges_per_letter: &HashMap<nfa::Letter, Graph>,
        maximal_finite_value: coef,
    ) -> bool {
        let mut result = false;
        for (a, downset) in self.0.iter_mut() {
            // print!(".");
            //io::stdout().flush().unwrap();
            let edges = edges_per_letter.get(a).unwrap();
            let safe_pre_image = safe.safe_pre_image(edges, maximal_finite_value);
            result |= downset.restrict_to(&safe_pre_image);
        }
        result
    }

    pub fn iter(&self) -> impl Iterator<Item = (&nfa::Letter, &DownSet)> {
        self.0.iter()
    }

    // create a CSV representation of this strategy.
    pub fn as_csv(&self) -> String {
        let mut lines: Vec<nfa::Letter> = Vec::new();

        for (a, i) in &self.0 {
            for s in i.as_csv() {
                let l = format!("{a},{s}");
                lines.push(l);
            }
        }
        lines.join("\n")
    }
}

impl fmt::Display for Strategy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut letters = self.0.keys().collect::<Vec<_>>();
        letters.sort();
        let strategy = letters
            .iter()
            .map(|a| {
                let downset = self.0.get(*a).unwrap();
                if downset.is_empty() {
                    format!("Never play action '{}'", a)
                } else {
                    format!(
                        "Play action '{}' in the downward-closure of\n{}\n",
                        a, downset
                    )
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        write!(f, "{}", strategy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ideal::Ideal;

    #[test]
    fn test_strategy() {
        let dim = 2;
        let letters = ["a", "b"];
        let strategy = Strategy::get_maximal_strategy(dim, &letters);
        let source = Ideal::new(dim, OMEGA);
        assert!(strategy.is_defined_on(&source));
        assert_eq!(
            strategy.0,
            HashMap::from([
                ('a'.to_string(), DownSet::from_vecs(&[&[OMEGA, OMEGA]])),
                ('b'.to_string(), DownSet::from_vecs(&[&[OMEGA, OMEGA]]))
            ])
        );
    }
}
