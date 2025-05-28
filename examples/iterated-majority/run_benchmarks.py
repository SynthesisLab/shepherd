#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import time
import csv
import matplotlib.pyplot as plt

# Constants
COMMANDS = [
    "schaeppert -vv -f dot nfa-{n}.dot iterate tmp/",
    # "schaeppert -f dot nfa-{n}.dot iterate tmp/",
    #"shepherd -vv -f dot nfa-{n}.dot",
    "shepherd -f dot nfa-{n}.dot",
]
TIMEOUT = 60  # seconds
TMP_DIR = "tmp"
LOGFILE_TEMPLATE = "log_{cmd_idx}_n{n}.txt"

def ensure_empty_tmp():
    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR)

def run_command(cmd, logfile, timeout):
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        elapsed = time.time() - start
        with open(logfile, 'wb') as f:
            f.write(b'=== STDOUT ===\n')
            f.write(proc.stdout)
            f.write(b'\n=== STDERR ===\n')
            f.write(proc.stderr)
        return elapsed
    except subprocess.TimeoutExpired as e:
        with open(logfile, 'wb') as f:
            f.write(b'=== STDOUT ===\n')
            f.write((e.stdout or b''))
            f.write(b'\n=== STDERR ===\n')
            f.write((e.stderr or b''))
            f.write(b'\n=== TIMEOUT ===\n')
        return None

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} N", file=sys.stderr)
        sys.exit(1)
    try:
        N = int(sys.argv[1])
    except ValueError:
        print("N must be an integer", file=sys.stderr)
        sys.exit(1)

    ensure_empty_tmp()
    results = []
    for n in range(1, N + 1):
        row = {'n': n}
        for cmd_idx, cmd_template in enumerate(COMMANDS):
            cmd = cmd_template.format(n=n)
            logfile = LOGFILE_TEMPLATE.format(cmd_idx=cmd_idx, n=n)
            print(f"Running command {cmd_idx+1} for n={n}: {cmd_template}")
            t = run_command(cmd, logfile, TIMEOUT)
            if t is None:
                break
            row[f'cmd{cmd_idx+1}_time'] = t
        else:
            results.append(row)
            continue
        break

    # Write CSV to stdout
    fieldnames = ['n'] + [f'cmd{i+1}_time' for i in range(len(COMMANDS))]
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

    # Plotting
    ns = [row['n'] for row in results]
    for cmd_idx, cmd_template in enumerate(COMMANDS):
        times = [row.get(f'cmd{cmd_idx+1}_time') for row in results]
        plt.plot(ns, times, label=cmd_template)
    plt.xlabel('n')
    plt.ylabel('Wallclock time (s)')
    plt.legend()
    plt.title('Command Running Times')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    plt.show()

if __name__ == '__main__':
    main()
