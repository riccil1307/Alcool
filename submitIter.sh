#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type='FAIL'
#SBATCH --mail-user='laura.ricci@student.uclouvain.be'

#SBATCH --output='/CECI/proj/pilab/PermeableAccess/alcoolique_TnB32xGDr7h/Codes/slurmJob.out'
#SBATCH --error='/CECI/proj/pilab/PermeableAccess/alcoolique_TnB32xGDr7h/Codes/slurmJob.err'
        
#source /home/users/n/d/ndelinte/Studies/envA/bin/activate
#source /CECI/proj/pilab/PermeableAccess/alcoolique_TnB32xGDr7h/Codes/.bash_profile
        
python /CECI/proj/pilab/PermeableAccess/alcoolique_TnB32xGDr7h/Codes/registration.py $1