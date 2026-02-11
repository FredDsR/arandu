#!/bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=draco
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=cleanup_%j.out
#SBATCH --error=cleanup_%j.err

# This script cleans up the project directory by removing all files and directories except for the cache directory.

docker run --rm -v /home/users/fdsreckziegel/etno-kgc-preprocessing:/project alpine find /project -mindepth 1 -maxdepth 1 ! -name cache -exec rm -rf {} +
