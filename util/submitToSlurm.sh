#!/bin/bash
#SBATCH --partition=slurm_me759
#SBATCH --time=0-00:05:00               # maximum run time in days-hh:mm:ss
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=q2
#SBATCH -o outConvolution_%j

cd $SLURM_SUBMIT_DIR

#./convolution.out 32
#./convolution.out 64
#./convolution.out 128
#./convolution.out 256
#./convolution.out 512
#./convolution.out 1024
#./convolution.out 2048

#filename="v5.txt"
#rm -f $filename
#touch $filename
echo \<root\>
for i in `seq 4 12`;
do
    N=$((2**i))
    echo N = 2^$i = $N
    echo \<run ID=$N\>
    ./convolution.out $N
    echo \<\/run\>
    echo -------------
done   
echo \<\/root\>
