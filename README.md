# Shepherd

`Shepherd` is a Rust implementation of an algorithm solving the random population control problem,
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

## First run on examples
Install rust and cargo from https://www.rust-lang.org/tools/install

In the root folder launch
```cargo run -- -f examples/example1.tikz```

That will load an automaton from the file ```examples/example1.tikz```,
compute the maximal winning strategy for the random population control problem
and display the details in the terminal.

The file ```examples.pdf```at the root give details about examples.

## Generate input files in tikz format

- Create an automaton using https://finsm.io
- Copy paste the export (in Tikz format) in some local file and give it as input to shepherd, using the `-f` option.

## Get the solution in tex and pdf format

By default, a tex file and a pdf file are generated, assuming `pdflatex`is installed on the machine.
The output is formatted using the template `latex/solution.template.tex`.
The tex conversion and pdf compilations can be optionally disabled, and the tex processor can also be modified,
run `cargo run -- -help` for more details.

## DOT-files as input

You can give the input NFA in [graphviz DOT](https://graphviz.org/docs/layouts/dot/) format 
by setting the input-format as "dot" and give a path to a dot-file as input file:

```cargo run -- -i dot -f examples/bottleneck-1-ab.dot```

The input graphs are interpret as NFA using the following convention.

- All nodes except for one special node with id "init" are states of the NFA;
- Initial states are those which have an unlabeled edge from "init" into it;
- Accepting states are those with attribute "shape:doublecircle";
- Every edge with "label" attribute results in a transition over the value of that label

See `examples/bottleneck-1-ab.dot` for a dot-representation equivalent to the simple bottleneck in `examples/bottleneck-1-ab.tikz`.
>>>>>>> 01f84c0 (README: add docs for dot input)
