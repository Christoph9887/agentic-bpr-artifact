# Agentic business process redesign from redesign heuristics and simulation models

This repository contains the framework, the inputs of the three event logs the
paper evaluates, and the complete records of the nine runs behind every table,
figure and number in the evaluation.

The system takes an event log, a process model and a simulation model fitted to
that log, detects candidate redesigns with fifteen redesign heuristics, and runs
a closed loop in which a compliance gate rules on what may be built, three
agents propose and judge plans, and every plan is **measured by simulation**
before anything is believed. A fourth agent picks the deliverable among the
measurements. The loop is a graph of 10 nodes and 15 edges; a deterministic
rule, not an agent, decides whether a round improved anything and when to stop.

Two ways to reproduce the paper, one command each.

## Setting up

    pip install -r requirements.txt

`requirements.txt` pins the ten packages the code imports at the versions the
nine reported runs were produced with, on **Python 3.9**. Two of them decide
whether a run is reproducible at all: `pm4py` reads the log and the model, and
`prosimos` is the simulator every measurement runs.

**No key is shipped.** Copy `.env.example` to `.env` and put your own
OpenAI-compatible key in it, or set `OPENAI_API_KEY` in the environment. Only
the closed loop calls a model; everything else here runs without a key.

## 1. Reproduce what the paper reports, from the records

    python reproduce_results.py

A few seconds, no model call, no simulation, no key. It

* prints the nine runs with all five key performance indicators, which of them
  are significant, the verdict of the pre-specified improvement rule, and the
  candidates each run delivered;
* prints the three result tables exactly as the paper carries them, and ends
  with the record file behind every other figure in the prose.

The medians it prints are these, and the spread of three runs is itself a
reported result:

| log | run 1 | run 2 | run 3 | spread |
|---|---:|---:|---:|---:|
| synthetic service process | -57.52 | -57.42 | -72.48 | 15.06 pp |
| BPI Challenge 2012 | -36.37 | -36.74 | -36.78 | 0.41 pp |
| BPI Challenge 2017 | -1.52 | -2.23 | -1.67 | 0.71 pp |

Percentages are the median case cycle time against that log's own simulated
baseline. All nine are significant and all nine satisfy the improvement rule:
*significantly better on at least one directional measure and significantly
worse on none*. Mean resource utilisation has no agreed direction, so it is
reported every round and can neither create nor block that verdict.

## 2. Reproduce a run from scratch, detection included

    python reproduce_run.py --dataset bpo

The inputs are three files and nothing else:

    data/<name>.xes[.gz]       the event log
    inputs/<tag>/<name>.bpmn   the mined process model, read by the detectors
                               and by the simulator
    inputs/<tag>/<name>.json   the simulation model fitted to that log:
                               resources, calendars, branching probabilities,
                               fitted durations and extraneous delays

All three are in this repository for all three datasets, so every command below
runs as it stands.

`reproduce_run.py` prints those three with their sizes before it starts, and
names any that is missing. From them it runs three stages, in order:

1. **detection** -- every heuristic is detected from the log and the model, and
   the candidates are written to `inputs/<tag>/detected_candidates/`,
   overwriting the ones shipped here. Re-detection on the synthetic log
   reproduces the shipped candidates exactly, up to the order of a few
   auxiliary lists;
2. **the baseline** -- the unchanged simulation model is simulated 20 times.
   That aggregate is the one fixed basis every later measurement is scored
   against, so it is measured before any redesign exists;
3. **the closed loop** -- the gate, the agents, apply, simulate and measure,
   round after round. It stops after two consecutive rounds that fail to improve
   on the incumbent, with a cap of 12 rounds, and writes
   `mabpr_out/<tag>/loop/<run>/` in the same shape as the records under
   `results/`.

Options: `--dataset bpo | 2012_full | 2017_full`, `--tag NAME` to name the run
directory, `--max-rounds N`, `--workers N` for how many Prosimos processes run
at once (default 8), and `--skip-detection` / `--skip-baseline` to reuse what an
earlier attempt left on disk.

**What it costs.** Each measurement is 20 Prosimos replicates, and a round takes
one or more measurements; the nine recorded runs made 8 to 18 model calls per
round, 439 in all. Expect tens of minutes for the synthetic log and hours for a
BPI log. What binds on a large log is committed memory rather than cores,
because every Prosimos worker is a process that imports pandas and numpy --
lower `--workers` before suspecting anything else.

**Three runs will not agree exactly**, and the paper reports three per log for
that reason. Three steps depend on a model call: the compliance ruling, which
open question the Critic spends a confirmation simulation on, and which
measurement is delivered. Everything between those calls -- the detectors, the
build step, the simulations, the five indicators and every z -- is deterministic
given the same inputs and the same plan.

### The logs

All three event logs the paper evaluates are here, in `data/`, and they are the
files the nine runs read. The two BPI Challenge logs are stored gzipped, which
is how they are published:

| log | cases | events | here | uncompressed |
|---|---:|---:|---|---:|
| synthetic service process | 3,000 | 12,464 | `.xes` | 4.4 MiB |
| BPI Challenge 2012 | 13,087 | 262,200 | `.xes.gz` | 58.7 MiB |
| BPI Challenge 2017 | 31,509 | 1,202,269 | `.xes.gz` | 445.6 MiB |

Nothing needs unpacking: the loader reads either form, and the 2017 log goes
from its 34.8 MiB archive to 475,306 activity instances in 25 seconds. Each BPI
log also has a `.csv.gz` beside it -- the same log in the preprocessed form the
mining step was given, which is what the batching evidence is computed from.

Re-detection reproduces the candidates: all 13 on the synthetic log and all 33
on BPI 2012, with the same evidence numbers. Two kinds of field are not
bit-stable between runs, because a topological order of a mined model is not
unique and the traversal order decides it: the local node names inside each
candidate's `process_fragment`, and, on BPI 2012, which of the two knock-out
checks counts as first -- which flips that candidate's `efficiency_gap` between
0.0 and 0.0511. The plain and the gzipped log give the same answer, so it is not
the compression, and no reported plan depends on it: the knock-out candidate
changed nothing in any of the nine runs.

## Layout

    mabpr/              the framework: the detectors, the four agents, the
                        deterministic core (apply, simulate, measure), the
                        evidence layer and the knowledge layer
    mabpr/foundation/   the reused base library, kept as its own tree: the first
                        seven heuristics' detectors and transformation handlers,
                        the graph of heuristic interactions, and the simulation
                        runner. The eight heuristics this paper adds live beside
                        it in `mabpr/detectors/` and `mabpr/transform/` and
                        register into its dispatch table;
                        `mabpr/transform/__init__.py` is all of that wiring
    inputs/             per log: the mined model, the simulation model, the
                        detected candidates and the mining configuration
    data/               the three event logs, the two BPI ones gzipped, each
                        with the preprocessed CSV the mining step was given
    results/            the nine runs: `result.json`, every round's plan,
                        compliance ruling, explanation, LLM trace and measurement
                        summary, the knowledge overlays, the probes, and each
                        run's delivered model and parameters
    gen_tables.py       rebuilds the paper's three result tables from `results/`
    reproduce_results.py, reproduce_run.py   the two commands above
    requirements.txt, .env.example           the pinned environment, and the one
                        value to fill in

Paths recorded inside the records begin with `mabpr_out/`, the output tree a run
writes; under `results/` the same run sits at `results/<log>/loop/<run>/`.
