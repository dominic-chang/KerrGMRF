#!/bin/sh
#SBATCH -t 3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2gb

#SBATCH -p blackhole
#SBATCH --job-name="KerrGMRFDualCone2017"
#SBATCH -o /n/home06/dochang/KerrGMRF/src/dual_cone_2017/out.txt
#SBATCH -e /n/home06/dochang/KerrGMRF/src/dual_cone_2017/err.txt

cd KerrGMRF
julia --project=. --threads=auto /n/home06/dochang/KerrGMRF/src/dual_cone_2017/fit_kerrgmrf_dual_cone_to_data.jl

