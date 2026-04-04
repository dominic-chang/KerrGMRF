#!/bin/sh
#SBATCH -t 3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3gb

#SBATCH -p blackhole
#SBATCH --job-name="KerrGMRFDualCone2017_phase_wrapped_no_n1_variability"
#SBATCH -o /n/home06/dochang/KerrGMRF/src/dual_cone_2017_phase_wrapped_no_n1_variability/out.txt
#SBATCH -e /n/home06/dochang/KerrGMRF/src/dual_cone_2017_phase_wrapped_no_n1_variability/err.txt

cd KerrGMRF
julia --project=/n/home06/dochang/KerrGMRF --threads=auto /n/home06/dochang/KerrGMRF/src/dual_cone_2017_phase_wrapped_no_n1_variability/fit_kerrgmrf_dual_cone_to_data_phase_wrapped.jl
