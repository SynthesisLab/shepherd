# shepherd

A Rust implementation of an algorithm solving the random population control problem,
presented in https://arxiv.org/abs/2411.15181 .

Starting from an nfa with states Q, the algorithm performs a fixpoint computation of the largest
arena containing all abstract winning configurations whose finite coordinates are below |Q|,
and other coordinates are omega (see Algorithm 1 in the paper for further details).

The fixpoint computation makes use of a subprocedure call to an algorithm solving the so-called path problem.
Given a symbolic arena and a (finite) set of abstract configurations F in the arena,
this problem asks whether, from every finite configuration of the arena,
there exists a path within the arena that reaches a configuration in (the ideal generated by) F.
This is solved by computing a semigroup called the symbolic flow semigroup (see Theorem 26 in the paper for further details).
