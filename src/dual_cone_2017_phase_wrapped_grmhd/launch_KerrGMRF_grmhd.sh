#!/bin/sh
#SBATCH -t 3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2gb

#SBATCH -p blackhole
#SBATCH --job-name="KerrGMRF_grmhd"
#SBATCH -o /n/home06/dochang/KerrGMRF/src/dual_cone_2017_phase_wrapped_grmhd/out.txt
#SBATCH -e /n/home06/dochang/KerrGMRF/src/dual_cone_2017_phase_wrapped_grmhd/err.txt

cd ~/KerrGMRF
julia --project=. --threads=auto /n/home06/dochang/KerrGMRF/src/dual_cone_2017_phase_wrapped_grmhd/fit_kerrgmrf_dual_cone_to_grmhd_phase_wrapped.jl

