#!/usr/bin/env python3
"""
generate_models.py

This script generates DOT files representing a family of NFA (nondeterministic finite automaton) models.
For each integer i from 1 to n (inclusive), it creates a file named 'nfa-i.dot' describing an NFA
with a specific structure:
- States 0 through i, plus a target state T.
- Initial transitions from state 0 to each state 1..i+1 labeled 'a'.
- For each state j in 1..i+1, transitions from j to T labeled 'a{k}' for every k in 1..i+1 where k != j.

Usage:
    python generate_models.py <n>
Where <n> is a positive integer.


Note: All of these models are not congrollable for arbitrary number of processes, but the n-fold product of the size-m model is controllable by playing action "a", then "aj" if node "j" if none of the n processes reside in node j.
"""
import sys
def generate_model_file(n):
    filename = f"nfa-{n}.dot"
    with open(filename, "w") as f:
        f.write('digraph NFA {\n')
        f.write('    rankdir=TB;\n')
        f.write('    init [label=" ",shape=none,height=0,width=0];\n')
        # States 0..N
        for i in range(n + 2):
            f.write(f'    {i} [label="{i}", shape=circle];\n')
        # Target state T
        f.write('    T [label="T", shape=doublecircle];\n\n')
        # Initial arrow
        f.write('    init -> 0;\n')
        # Transitions from 0 to 1..N
        for i in range(1, n + 2):
            f.write(f'    0 -> {i} [label="a"];\n')
        # For every two nodes 1 <= i, j <= N with i != j, add j -> T [label="i"]
        for j in range(1, n + 2):
            for i in range(1, n + 2):
                if i != j:
                    f.write(f'    {j} -> T [label="a{i}"];\n')
        f.write('}\n')

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_models.py <n>")
        sys.exit(1)
    try:
        n = int(sys.argv[1])
        if n < 1:
            raise ValueError
    except ValueError:
        print("n must be a positive integer")
        sys.exit(1)
    for i in range(1, n + 1):
        generate_model_file(i)

if __name__ == "__main__":
    main()
