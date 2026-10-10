#!/bin/bash
#SBATCH -t 0-04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH -p test
#SBATCH --job-name="PHIBI_reactant_test"
#SBATCH -o /n/home06/dochang/KerrGMRF/reactant/logs/%x_%j.out
#SBATCH -e /n/home06/dochang/KerrGMRF/reactant/logs/%x_%j.err

# Precompile the Reactant project, then run a test script (default test_fast_light.jl).
cd /n/home06/dochang/KerrGMRF/reactant
export JULIA_NUM_PRECOMPILE_TASKS=8
julia --project=. --startup-file=no -e 'using Pkg; Pkg.precompile()'
julia --project=. --startup-file=no --threads=auto ${TEST:-test_fast_light.jl}
