# Run Gaussian on Hipergator

## Linux
### Common
```bash
cd 
cd .      # current dir
cd ..     # cd up one directory
cd ~      # cd home
ls
pwd       # print work dir
realpath  # absolute path of file
cp        # copy
mv        # move, rename
rm        # delete a file
```
### file reading and writing
```
less
head
tail
vim
```
### SSH
login with password
```bash
ssh username@hpg2.rc.ufl.edu
```
Setup public key login
```bash
# generate public key to login without password for a single machine
# ~/.ssh/id_rsa.pub is the public key
# ~/.ssh/id_rsa is the private key
ssh-keygen
# if remote machine ~/.ssh/authorized_keys has local machine's public key, 
# public key authentication is enabled
ssh-copy-id username@hpg2.rc.ufl.edu
```
then one could login without password
```bash
ssh username@hpg2.rc.ufl.edu
```
### Scp
copy files between local and remote machine
```bash
# copy local file to remote machine
scp ch4.com username@hpg2.rc.ufl.edu:~/project
# copy remote file to local machine
scp username@hpg2.rc.ufl.edu:~/project/ch4.log ./
```
Some apps make it easier to transfer files: ForkLift, search google for sftp app
## Gaussian
```bash
g09 < ch4.com  # run gaussian without saving
g09 < ch4.com > ch4.log  # save output to log file
```
## Slurm
### Login node
Never run job on login node, login node is shared to all the people. The purpose of login node is to submit jobs, not for doing calculation.

### srun (iteractive)
```bash
srun --ntasks=1 --cpus-per-task=2 --time=01:00:00 --mem=20gb  --pty -u bash -i
```
### submit job to queue (big job)
```bash
squeue  # show the queue
```
create a job submission script `ch4-submit.sh`
```bash
#!/bin/bash
#SBATCH --job-name=run_gaussian
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12gb
#SBATCH -t 01:00:00
#SBATCH -o run_gaussian-%j.out

module load gaussian
g09 < ch4.com > ch4.log
```
```
sbatch ch4-submit.sh
```

Ref: 
1. [SLURM Commands - UFRC](https://help.rc.ufl.edu/doc/SLURM_Commands)
2. [UFRC Class_Orientation.pdf](https://caleb-huo.github.io/teaching/2018FALL/lectures/week4_hiperGator/hiperGator/RC_Class_Orientation.pdf)
