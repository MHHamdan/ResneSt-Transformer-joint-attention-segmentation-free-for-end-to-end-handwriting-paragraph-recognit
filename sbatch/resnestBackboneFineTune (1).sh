#!/bin/bash
#SBATCH --account=def-mcheriet
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=40000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=2-00:00:00     # DD-HH:MM:SS
export CUBLAS_WORKSPACE_CONFIG=:4096:2

module restore my_modules
source ~/env/bin/activate
# python train_resnest.py --target_path "saved_models/" --file_path "rimes_experimental.hdf5" --batch_size 32 --charset_base 'rimes_vocab' --finetune "resnestrimes_paragraph_pretrainingbest_loss.pt" --lr "0.00006" --epochs 100

python train_resnest-encoder.py --target_path "saved_models/" --file_path "reads_paragraph_finetune.hdf5" --batch_size 32 --charset_base 'read_vocab' --lr "0.00006" --epochs 200 --name_file "resnest_encoder" --finetune "resnest_encoderreads_paragraph_pretrainingbest_loss.pt" 

# python train_resnest.py --target_path "saved_models/" --file_path "reads_paragraph_pretraining.hdf5" --batch_size 36 --charset_base 'read_vocab' --epochs 100