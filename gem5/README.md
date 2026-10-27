# Where Does Scratchpad SRAM Saturate? (gem5)

EEL5764 Computer Architecture, University of Florida, Fall 2026, Group 4:
Evan Espino-DeJesus, Tyla Hart, Kasey Hill, James Woelke.

A weight-stationary INT8 systolic array does N² multiply-accumulates per
cycle. Its operands come from a scratchpad SRAM split into BANKS banks,
each delivering 8 bytes per cycle. Compute grows as N², but scratchpad
bandwidth grows only with BANKS. This project finds the array size at which
the scratchpad can no longer keep the array busy, how that depends on data
reuse, and how many banks fix it.

This repository is the gem5 half of the project: everything needed to
build the simulator and regenerate every result, and nothing generated.
It contains:

* a cycle-level gem5 model of the array (`SystolicAccel`);
* a RISC-V host system that drives it;
* the workload generator;
* the definitions of the 772 simulations behind the results;
* the scripts that turn those simulations into data, figures and tables.

The generated results (CSV data, figures, tables) are not kept in git. The
project's documents and presentations carry them, and `make data` plus
`make figures` regenerate them from scratch. Every simulation checks its
output bit-exactly against a reference result.

The RTL half (SystemVerilog, Questa, Vivado) is developed separately. This
repository provides its test data and the gem5 side of the gem5-vs-RTL
comparison.

What the flow produces, in `results/`:

* `data/`: one CSV per experiment suite (one row per simulation, every
  metric of the proposal's Table II) and `key_results.json` (every
  headline number).
* `figures/`: 21 figures, such as the saturation curves (PE utilization vs
  N per bank count), the knee bank by bank, the bank-matching rule, and
  low reuse vs DRAM. Each comes as PNG and PDF.
* `tables/`: 19 tables, from the proposal's Tables I and II to the
  saturation points, as Markdown and CSV.

`analysis/README.txt` describes each one.

## What each folder is for

| Folder | Purpose |
| :--- | :--- |
| [gem5_model/](gem5_model/) | **The simulated hardware.** See below. |
| [workloads/](workloads/) | **The layers being simulated.** See below. |
| [analysis/](analysis/) | **The experiments and the analysis.** See below. |
| [scripts/](scripts/) | **Build and run scripts.** The Makefile calls them, and they can also be run by hand. See below. |

**[gem5_model/](gem5_model/)**, the simulated hardware:

* `src/accel/`: the `SystolicAccel` SimObject in C++ and Python. It holds
  the N×N array's timing and functional model, its registers and its DMA
  engines. gem5 compiles it in through its `EXTRAS` option, so no files
  are copied into gem5.
* `configs/systolic_se.py`: builds the whole system and runs one layer.
  The system is a RISC-V CPU, caches, DDR4, the accelerator and its banked
  scratchpad.
* `host/systolic_host.c`: the RISC-V program the CPU runs. It loads a
  layer, programs the accelerator and checks the result.
* `patches/`: the single 6-line change this project needs in gem5 itself.
* `README.txt`: the detailed reference (memory map, registers, schedule,
  statistics, debugging).

**[workloads/](workloads/)**, the layers being simulated:

* `workloads.yaml` lists the shapes: ResNet-50 conv3_x, BERT-base FFN1 and
  QKᵀ, LLaMA-7B GEMV, and a synthetic reuse sweep.
* `gen_workloads.py` writes every operand and golden result from a fixed
  seed.
* `README.txt` cites where each shape comes from.

**[analysis/](analysis/)**, the experiments and the analysis:

* `experiments.py` defines the 18 experiment suites (772 simulations).
* `collect.py` runs them in parallel.
* `parse_stats.py` turns gem5's `stats.txt` into one CSV row.
* `model.py` is the analytical model (eq. 1–2).
* `plot_curves.py` and `style.py` draw the figures and tables.
* `compare_results.py` compares two sets of results number by number.
* `templates/` holds the formats for the RTL cycle counts and Vivado
  numbers, which feed two of the figures once they exist.
* `README.txt` describes every suite, figure and table.

**[scripts/](scripts/)**, the build and run scripts:

* `env.sh` sets the paths every script uses.
* `build_gem5.sh` fetches, patches and builds gem5.
* `build_host.sh` cross-compiles the host program.
* `run_workload.sh` runs one layer.
* `run_all_workloads.sh` is the smoke test.

Top-level files:

* `Makefile` runs every step (`make help`).
* `env.local.sh.example` is a template for machine-specific settings (copy
  it to `env.local.sh`).
* `requirements.txt` lists the Python packages.
* `.gitignore` and `.gitattributes` keep generated files out of git and
  force LF line endings.

These folders are created when you build and run, and git ignores them:

| Created folder | By | Contents | Size |
| :--- | :--- | :--- | :--- |
| `third_party/gem5/` | `make gem5` | gem5 checkout and its build | about 6 GB |
| `third_party/zig-venv/` | `make zig` | RISC-V C compiler (optional) | 0.4 GB |
| `.venv/` | `make venv` | Python packages | 0.3 GB |
| `workloads/out/` | `make workloads` | generated operands and golden results | 40 MB |
| `gem5_model/host/systolic_host` | `make host` | the RISC-V host binary | 31 KB |
| `build/runs/` | `make data` | raw gem5 output of every simulation | 0.5 GB |
| `results/` | `make data`, `make figures` | data, figures and tables (or `RESULTS_DIR`) | 5 MB |
| `build/m5out/` | `make run`, `make smoke` | output of single runs | small |

## Running everything in WSL

These steps take a fresh WSL install to finished results. They were run on
Windows 11 with WSL2 and Ubuntu 24.04 (24 threads, 16 GB of memory for
WSL). Every command runs in the Ubuntu terminal.

### 0. WSL basics

* To install Ubuntu, open PowerShell and run `wsl --install -d Ubuntu-24.04`,
  then open "Ubuntu" from the Start menu.
* **Keep this folder in the Linux file system** (under `~`), not under
  `/mnt/c/...`. On `/mnt/c` the gem5 build is many times slower, and
  Windows tools can break the scripts' line endings.
* Windows Explorer can still open it at
  `\\wsl.localhost\Ubuntu-24.04\home\<your-user>\`. Running `code .`
  opens the folder in VS Code (Remote - WSL).
* Check what WSL gives you with `nproc` (threads) and `free -g` (memory).
  WSL gets half of the Windows machine's memory by default.

### 1. Install the build tools (once)

```bash
sudo apt update
sudo apt install build-essential m4 scons python3-dev python3-venv git zlib1g-dev
sudo apt install libgoogle-perftools-dev   # optional: makes gem5 about 12% faster
```

The first four packages are gem5's own minimum for Ubuntu 24.04.
`python3-venv` is needed by `make venv`, and `git` by `make gem5`.
Ubuntu 24.04's GCC 13 is within gem5's supported range (GCC 11–14).

**Without sudo,** use a conda environment that provides Python 3 with its
headers, scons and a GCC 11–14. Point the `gem5_build_env` function in
`env.local.sh` at it (step 3). The published results were built this
way; see `env.local.sh.example`.

### 2. Get this folder

```bash
cd ~
git clone <repository-url> systolic-gem5 && cd systolic-gem5
```

or, from the zip (Ubuntu has no `unzip` by default; Python can extract it):

```bash
cd ~
python3 -m zipfile -e /mnt/c/Users/<you>/Downloads/for_git.zip .
mv for_git systolic-gem5 && cd systolic-gem5
```

Every command from here on runs in this folder.

### 3. Machine-specific settings (only if needed)

```bash
cp env.local.sh.example env.local.sh
```

Then uncomment what applies: an existing gem5 checkout (`GEM5_DIR`), a
different Python, a RISC-V compiler, where results go (`RESULTS_DIR`), the
reference for `make check` (`CHECK_REF`), job counts, or the
`gem5_build_env` function for a conda toolchain. Settings given on the
make command line (`make data SIM_JOBS=8`) override this file. git
ignores `env.local.sh`.

### 4. Python packages

```bash
make venv
```

This installs `requirements.txt` into `.venv/`, which every script then
uses automatically. Ubuntu 24.04 refuses a plain `pip install` outside a
virtual environment. The versions are pinned to the ones the results were
made with, so a rerun reproduces `results/` byte for byte. To use another
Python instead, set `PYTHON` in `env.local.sh`.

### 5. Configure and build gem5 for this folder

The accelerator is not copied into gem5's source tree. gem5 compiles it
from `gem5_model/src` through its `EXTRAS` option into a separate build
directory, `build/RISCV_SYSTOLIC`, so one gem5 checkout can hold a stock
build and this one side by side. Choose the checkout:

* **Default: a private copy.** Change nothing. `make gem5` clones gem5
  v25.1.0.1 into `third_party/gem5/`.
* **An existing checkout** (it must be release v25.1.0.1). Add
  `export GEM5_DIR="${GEM5_DIR:-$HOME/gem5}"` to `env.local.sh`.
  `make gem5` will patch that checkout (6 lines in
  `src/arch/riscv/tlb.cc`) and build there.

Then build:

```bash
make gem5                     # 20-30 min the first time
make gem5 BUILD_JOBS=8        # fewer parallel jobs if memory is short
```

`make gem5` (= `scripts/build_gem5.sh`) does four things:

1. It clones gem5, if `GEM5_DIR` does not exist yet.
2. It applies `gem5_model/patches/riscv_se_uncacheable_tlb.patch`, unless
   it is already applied.
3. On the first build only, it runs
   `scons defconfig build/RISCV_SYSTOLIC build_opts/RISCV`.
4. It runs `scons build/RISCV_SYSTOLIC/gem5.opt EXTRAS=<this folder>/gem5_model/src`.

The result is `$GEM5_DIR/build/RISCV_SYSTOLIC/gem5.opt` (about 950 MB).
After editing the accelerator's C++ or `SystolicAccel.py`, run
`make gem5` again; it only recompiles what changed (1–2 min). Config
changes (`systolic_se.py`) need no rebuild.

To check where the binary is:

```bash
source scripts/env.sh && ls -lh "$GEM5_BIN"
```

### 6. RISC-V compiler and host program

```bash
make zig      # zig 0.16.0 into third_party/zig-venv; skip if you have
              # riscv64-linux-gnu-gcc (sudo apt install gcc-riscv64-linux-gnu)
make host     # -> gem5_model/host/systolic_host
```

The published results were produced with zig 0.16.0. `make zig`
installs exactly that version.

### 7. Workloads

```bash
make workloads    # -> workloads/out/ (a few seconds, about 40 MB)
```

### 8. Smoke test

```bash
make smoke
```

This runs every workload in its reduced size, at N = 8 with 4 banks, once
with the banked scratchpad and once with an ideal one. It takes about
10 seconds. It prints a table and fails unless all 8 runs verified:

```
workload               scratchpad  pe_util  model  verify
bert_attn_qk_rtl       banked        0.783  0.772  ok
...
all 8 runs verified
```

### 9. One layer

```bash
make run WL=resnet50_conv3x N=16 BANKS=8
make run WL=bert_ffn1 N=32 BANKS=21 ARGS='--bank-policy reads_first'
make run WL=bert_ffn1 N=16 BANKS=8 ARGS='--mode sw'     # software baseline
```

* `WL` is `resnet50_conv3x`, `bert_ffn1`, `bert_attn_qk`, `llama7b_gemv`,
  a reduced `*_rtl` variant, or `reuse_M<M>`.
* `N` is a multiple of 8, and `BANKS` is any count from 1 up.
* Each run prints the proposal's Table II metrics. gem5's full output goes
  to `build/m5out/<WL>_N<N>_B<BANKS>/`.
* For every option of the system, run
  `source scripts/env.sh && $GEM5_BIN gem5_model/configs/systolic_se.py --help`.
* To trace the accelerator, add
  `DEBUG=SystolicAccel,SystolicAccelDma` in front of `make run`. Use a
  small `_rtl` layer, because traces get large.

### 10. All experiments

```bash
make list         # the 18 suites and how many runs each has
make data         # all 772 simulations -> results/data/<suite>.csv
make data RESULTS_DIR=~/project-docs/results   # write somewhere else
```

* `make data` takes about an hour on 24 threads, and most of that is one
  long software baseline. It runs 12 simulations at a time (`SIM_JOBS`).
* Finished runs are kept in `build/runs/`. If Windows sleeps or WSL
  restarts, run `make data` again and it continues where it stopped.
* To run only some suites, use `make data SUITES="main knee"`.
* After changing the model, use `make data FORCE=1` so that old runs are
  not reused.

### 11. Figures and tables

```bash
make figures      # -> results/figures, results/tables, results/data/key_results.json
make figures PLOTS="fig1 fig17 tables"   # only some of them
```

### 12. Check the results against a reference

```bash
make check REF=path/to/reference/results
```

This compares `results/` with another results folder, number by number,
for example the copy that went into the project documents. Set
`CHECK_REF` in `env.local.sh` to make that the default.

* The simulation data (`results/data/<suite>.csv`) must match exactly.
* Derived numbers (tables, `key_results.json`) may differ in their last
  bits when numpy or pandas is a different version, because those
  libraries sum in a different order. `make check` accepts differences
  below 10⁻⁹ relative and says how many it found.
* With the pinned versions from `make venv`, a rerun rewrites every file
  byte-for-byte; `diff -r` against the reference then prints nothing.

To do steps 5 to 11 in one go after the setup, run `make all` (about
1.5 hours from scratch).

## Make targets

| Target | What it does |
| :--- | :--- |
| `make help` | list the targets and examples |
| `make venv` | install `requirements.txt` into `.venv` |
| `make all` | `build`, `data`, `figures` |
| `make build` | `gem5`, `host`, `workloads` |
| `make gem5` | clone (if needed), patch and build gem5 with the accelerator |
| `make zig` | install zig 0.16.0 into `third_party/zig-venv` |
| `make host` | cross-compile the RISC-V host program |
| `make workloads` | generate the workload files |
| `make smoke` | 8 short runs that must all verify |
| `make run WL=… N=… BANKS=… ARGS=…` | one layer, with its Table II metrics |
| `make list` | the experiment suites and their sizes |
| `make data [SUITES=…] [FORCE=1]` | run the experiments into `results/data/` |
| `make figures [PLOTS=…]` | redraw figures and tables from `results/data/` |
| `make model` | the analytical model's predictions (Table I) |
| `make check REF=dir` | compare `results/` with a reference results folder |
| `make clean` | delete `build/` (raw runs) |
| `make distclean` | also delete `results/`, workloads, host program, `.venv` and `third_party/` |

| Setting | Default | Meaning |
| :--- | :--- | :--- |
| `GEM5_DIR` | `third_party/gem5` | gem5 checkout to build against |
| `PYTHON` | `.venv/bin/python3`, else `python3` | Python with `requirements.txt` |
| `RESULTS_DIR` | `results/` | where `make data` and `make figures` write |
| `CHECK_REF` | none | reference results folder for `make check` |
| `RISCV_CC` | first one found | RISC-V compiler command |
| `SIM_JOBS` | 12 | simulations in parallel for `make data` |
| `BUILD_JOBS` | nproc − 2 | compile jobs for `make gem5` |

## Reproducibility

The simulations are deterministic, and the results do not depend on where
this folder is installed:

* the simulated program is given file names, not paths, because its
  arguments are copied onto its stack and would change its timing;
* the host binary carries no paths;
* the CSVs store run directories relative to the repository;
* figures and tables carry no dates or version strings.

This was checked on 2026-10-25 by running everything twice, in a fresh
clone and in a second copy at a different path. Both times gem5 was built
from scratch and all 772 runs were repeated. Every result file came out
byte-identical. So did the workload files and the host binary.

The results depend on:

* gem5 v25.1.0.1 with the patch in `gem5_model/patches/`;
* the host program built by zig 0.16.0;
* the seed in `workloads/workloads.yaml`;
* for byte-identical tables and figures, the Python package versions in
  `requirements.txt`.

## Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| scons: `Can't find a working Python installation` | `python3-dev` is missing, or gem5 finds the wrong `python3-config`. Install it (step 1) or set up `gem5_build_env` (step 3). |
| Build stops with `Killed signal terminated program cc1plus` | WSL ran out of memory. Run `make gem5 BUILD_JOBS=8`, or raise WSL's memory in `%UserProfile%\.wslconfig` (`[wsl2]` then `memory=12GB`), then run `wsl --shutdown` in PowerShell. |
| `error: externally-managed-environment` from pip | Use `make venv` (step 4). |
| `ensurepip is not available` | `sudo apt install python3-venv` |
| `No RISC-V compiler found` | `make zig` |
| `$'\r': command not found` or `bash\r` | The files have Windows line endings. Clone or extract inside WSL, not with Windows tools. |
| Everything is very slow | The folder is under `/mnt/c`. Move it to `~`. |
| `make gem5` recompiles everything after the folder moved | gem5 records absolute paths, so a moved checkout rebuilds once (about 20 min). |
| `make check`: `give a reference results folder` | Pass `REF=dir`, or set `CHECK_REF` in `env.local.sh`. |

## More documentation

* [gem5_model/README.txt](gem5_model/README.txt): the system, memory map,
  registers, schedule, arbitration, statistics and debugging.
* [workloads/README.txt](workloads/README.txt): the layers, file formats,
  and where each shape comes from.
* [analysis/README.txt](analysis/README.txt): every experiment suite,
  figure and table, and how to add the RTL and Vivado numbers.

The evaluation of the results against the proposal's hypotheses lives
with the project documents, next to the results it discusses.

## References

* J. Lowe-Power et al., "The gem5 Simulator: Version 20.0+,"
  arXiv:2007.03152, 2020.
* H. T. Kung, "Why systolic architectures?" *IEEE Computer*, vol. 15,
  no. 1, pp. 37–46, 1982.
* The sources of the workload shapes (ResNet-50, BERT-base, LLaMA-7B) are
  cited in [workloads/README.txt](workloads/README.txt).
