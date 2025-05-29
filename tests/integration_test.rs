use shepherd::coef::{C0, C1, C2, OMEGA};
use shepherd::downset::DownSet;
use shepherd::ideal::Ideal;
use shepherd::nfa;
use shepherd::solver;

const EXAMPLE1: &str = include_str!("../examples/bottleneck-1-ab.tikz");
const EXAMPLE1_COMPLETE: &str = include_str!("../examples/bottleneck-1-ab-complete.tikz");
const EXAMPLE2: &str = include_str!("../examples/bottleneck-2.tikz");
const EXAMPLE_BUG12: &str = include_str!("../examples/bug12.tikz");

#[test]
fn test_example_1() {
    let nfa = nfa::Nfa::from_tikz(EXAMPLE1);
    let solution = solver::solve(&nfa, &solver::SolverOutput::YesNo);
    print!("{}", solution);
    assert!(!solution.is_controllable);
    assert_eq!(solution.winning_strategy.iter().count(), 2);
    let downseta = solution
        .winning_strategy
        .iter()
        .filter(|x| x.0 == "a")
        .map(|x| x.1)
        .next()
        .unwrap();
    let downsetb = solution
        .winning_strategy
        .iter()
        .filter(|x| x.0 == "b")
        .map(|x| x.1)
        .next()
        .unwrap();

    assert_eq!(
        *downseta,
        DownSet::from_vecs(&[&[C1, C0, C0, C0, C0], &[C0, OMEGA, C0, C0, C0]])
    );
    assert_eq!(*downsetb, DownSet::from_vecs(&[&[C0, C0, OMEGA, C0, C0]]));
}

#[test]
fn test_example_1bis() {
    let nfa = nfa::Nfa::from_tikz(EXAMPLE1_COMPLETE);
    let solution = solver::solve(&nfa, &solver::SolverOutput::YesNo);
    print!("{}", solution);
    assert!(!solution.is_controllable);
    assert_eq!(solution.winning_strategy.iter().count(), 2);
    let downseta = solution
        .winning_strategy
        .iter()
        .filter(|x| x.0 == "a")
        .map(|x| x.1)
        .next()
        .unwrap();
    let downsetb = solution
        .winning_strategy
        .iter()
        .filter(|x| x.0 == "b")
        .map(|x| x.1)
        .next()
        .unwrap();

    assert_eq!(
        *downseta,
        DownSet::from_vecs(&[&[C1, OMEGA, C0, OMEGA, C0]])
    );
    assert_eq!(
        *downsetb,
        DownSet::from_vecs(&[&[C0, C0, OMEGA, OMEGA, C0]])
    );
}

#[test]
fn test_example_2() {
    let nfa = nfa::Nfa::from_tikz(EXAMPLE2);
    let solution = solver::solve(&nfa, &solver::SolverOutput::Strategy);
    print!("{}", solution);
    assert!(!solution.is_controllable);
    assert_eq!(solution.winning_strategy.iter().count(), 4);
    let downseta = solution
        .winning_strategy
        .iter()
        .filter(|x| x.0 == "a")
        .map(|x| x.1)
        .next()
        .unwrap();

    assert_eq!(*downseta, DownSet::from_vecs(&[&[C2, C0, C0, C0, C0]]));
}

#[test]
fn test_example_2_sorted_alpha() {
    let mut nfa = nfa::Nfa::from_tikz(EXAMPLE2);
    nfa.sort(&nfa::StateOrdering::Alphabetical);
    let solution = solver::solve(&nfa, &solver::SolverOutput::Strategy);
    assert!(!solution.is_controllable);
    assert_eq!(solution.winning_strategy.iter().count(), 4);
    let downseta = solution
        .winning_strategy
        .iter()
        .filter(|x| x.0 == "a")
        .map(|x| x.1)
        .next()
        .unwrap();

    assert_eq!(*downseta, DownSet::from_vecs(&[&[C0, C0, C0, C0, C2]]));
}

#[test]
fn test_example_2_sorted_topo() {
    let mut nfa = nfa::Nfa::from_tikz(EXAMPLE2);
    nfa.sort(&nfa::StateOrdering::Topological);
    let solution = solver::solve(&nfa, &solver::SolverOutput::Strategy);
    assert!(!solution.is_controllable);
    assert_eq!(solution.winning_strategy.iter().count(), 4);
    let downseta = solution
        .winning_strategy
        .iter()
        .filter(|x| x.0 == "a")
        .map(|x| x.1)
        .next()
        .unwrap();

    assert_eq!(*downseta, DownSet::from_vecs(&[&[C2, C0, C0, C0, C0]]));
}

#[test]
fn test_bug12() {
    let mut nfa = nfa::Nfa::from_tikz(EXAMPLE_BUG12);
    nfa.sort(&nfa::StateOrdering::Topological);
    let solution = solver::solve(&nfa, &solver::SolverOutput::Strategy);
    let downsetb = solution
        .winning_strategy
        .iter()
        .filter(|x| x.0 == "b")
        .map(|x| x.1)
        .next()
        .unwrap();
    println!("{}", downsetb);
    assert!(downsetb.contains(&Ideal::from_vec(vec![C2, C0, C0, C0, C0, C0, C0, C0])));
    assert!(downsetb.contains(&Ideal::from_vec(vec![C0, C1, C1, C0, C0, C0, C0, C0])));
    assert!(downsetb.contains(&Ideal::from_vec(vec![C0, C0, C0, C0, C0, C0, C0, OMEGA])));
}
