# shepherd

A Rust implementation of an algorithm solving the random population control problem,
presented in https://arxiv.org/abs/2411.15181 .

Starting from an nfa with states Q, the algorithm performs a fixpoint computation of the 
maximal winning strategy for the random population control problem.
The strategy is finitely represented as a symbolic set of pairs of letters and configurations,
whose finite coordinates are below |Q|,
and other coordinates are OMEGA (see Algorithm 1 in the paper for further details).

The fixpoint computation makes use of a subprocedure call to an algorithm solving the so-called `path problem`.
Given a symbolic arena and a (finite) set of abstract configurations F in the arena,
this problem asks whether, from every finite configuration of the arena,
there exists a path within the arena that reaches a configuration in (the ideal generated by) F.
This is solved by computing a semigroup called the symbolic flow semigroup (see Theorem 26 in the paper for further details).

#First run on examples
Install rust and cargo from https://www.rust-lang.org/tools/install

In the root folder launch
```cargo run -- -f examples/example1.tikz```

#Using shepherd
Create an automaton using https://finsm.io
Copy paste the export (in Tikz format) in some local file 

```cargo add clap --features derive```

