# Shepherd <img src="https://raw.githubusercontent.com/SynthesisLab/shepherd/main/.graphics/logo.png" align="right" height="140" />

<!-- badges: start -->
[![made-with-rust](https://img.shields.io/badge/Made%20with-Rust-1f425f.svg)](https://www.rust-lang.org/)
[![Build](https://github.com/SynthesisLab/shepherd/actions/workflows/build.yml/badge.svg?branch=main&event=push)](https://github.com/SynthesisLab/shepherd/actions/workflows/build.yml)
[![Docs](https://github.com/SynthesisLab/shepherd/actions/workflows/docs.yml/badge.svg?branch=main&event=push)](https://github.com/SynthesisLab/shepherd/actions/workflows/docs.yml)
[![Tests](https://github.com/SynthesisLab/shepherd/actions/workflows/test.yml/badge.svg?branch=main&event=push)](https://github.com/SynthesisLab/shepherd/actions/workflows/test.yml)
<!-- badges: end -->


`Shepherd` is an implementation of an algorithm solving the "random population control problem",
as presented in <https://arxiv.org/abs/2411.15181>.

Starting from a non-deterministic finite automaton (nfa),
the algorithm performs a fixpoint computation of the 
maximal winning strategy for the random population control problem.
The strategy is finitely represented as a symbolic set of pairs of letters and configurations,
whose finite coordinates are below $|Q|$, where $Q$ is the set of states of the nfa,
and other coordinates are $\omega$ (see Algorithm 1 in the paper for further details).

The fixpoint computation makes use of a subprocedure to solve the so-called "path problem".
Given a symbolic arena and a (finite) set of abstract configurations $F$ in the arena,
this problem asks whether, from every finite configuration of the arena,
there exists a path within the arena that reaches a configuration in (the downward-closed set generated by) $F$.
This is solved by computing a semigroup called the *symbolic flow semigroup* (see Theorem 26 in the paper for further details).

## Quick start

Assuming you have [rust and cargo installed](https://www.rust-lang.org/tools/install), compile and run shepherd as follows.

```
cargo run -- examples/example1.tikz
```

That will load an automaton from the file `examples/example1.tikz`,
compute the maximal winning strategy for the random population control problem,
and displays the answer in the terminal, including a winning strategy for positive instances.

Check the file ```examples.pdf``` at the root  which gives an overview of all available examples.

## Installation

You can build an optimized binary (will be placed in `target/release/shepherd`) using the following command.

```
cargo build --release
```

To install the binary for later use this:
```
cargo install --path .  # to install the `shepherd` binary to your $PATH
```
This will move the binary into to your cargo path, usually `~/.cargo/bin`, make sure to include this in your `$PATH`.

To run tests:
```
cargo test
```

To generate html docs to `target/doc/shepherd/index.html`
```
cargo doc
```

## Command-line Usage

```
Usage: shepherd [OPTIONS] <AUTOMATON_FILE>

Arguments:
  <AUTOMATON_FILE>  Path to the input

Options:
  -f, --from <INPUT_FORMAT>
          The input format [default: tikz] [possible values: dot, tikz]
  -v, --verbose...
          Increase verbosity level
  -l, --log-output <LOG_FILE>
          Optional path to the log file. Defaults to stdout if not specified.
  -t, --to <OUTPUT_FORMAT>
          The output format [default: plain] [possible values: plain, tex, csv]
  -o, --output <OUTPUT_FILE>
          Where to write the strategy; defaults to stdout.
  -s, --state-ordering <STATE_ORDERING>
          The state reordering type. [default: input] [possible values: input, alphabetical, topological]
      --solver-output <SOLVER_OUTPUT>
          Solver output specification. [default: strategy] [possible values: yes-no, strategy]
  -h, --help
          Print help
  -V, --version
          Print version
```

## Input

Two kind of input files can be processed by `shepherd`.

### Tikz files (as produced by finsm.io)

- Create an automaton using <https://finsm.io>
- Copy and paste the generated tikz code into some file and feed it to shepherd.

### DOT files

You can give the input NFA in [graphviz DOT](https://graphviz.org/docs/layouts/dot/) format 
by setting the input-format as "dot" and give a path to a dot-file as input file:

```
shepherd -i dot -f examples/bottleneck-1-ab.dot
```

The input graphs are interpret as NFA using the following convention.

- All nodes except for one special node with id "init" are states of the NFA;
- Initial states are those which have an unlabeled edge from "init" into it;
- Accepting states are those with attribute "shape:doublecircle";
- Every edge with "label" attribute results in a transition over the value of that label

See `examples/bottleneck-1-ab.dot` for a dot-representation equivalent to the simple bottleneck in `examples/bottleneck-1-ab.tikz`.

## Output

Each computation produces and prints whether the given autonmaton is controllable or not.
For positive instances, it will also give a representation of the winning strategy.
You can optionally select which format this is given via the `-t` argument (either "tex" or "plain", defaults to "tex"),
and give the path to where the solution is written via the `-o` argument (defaults to stdout).

For a pretty latex report use

```
shepherd -f examples/example1.tikz -o report.tex && pdflatex report.tex
```

The states of the NFA can be automatically reordered in order to make the generated reports more readable.
Either topologically (`-s topological`) or alphabetically (`-s alphabetical`).


