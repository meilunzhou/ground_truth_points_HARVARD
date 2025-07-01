#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=4:00:00
#SBATCH --mem=30gb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhou.m@ufl.edu
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1



source /blue/azare/zhou.m/anaconda3/bin/activate gpu_commanet



# python neon_image_reconstruction_continuous.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 --cls_weight $6
# python neon_image_reconstruction_without_triplet.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 
# python neon_image_reconstruction_discrete.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 


# python neon_mm_image_reconstruction.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 --cls_weight $6
# python neon_mm_image_reconstruction_ultimate.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 

# python neon_CL_classification.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 --cls_weight $6
python neon_HL_regression.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 --cls_weight $6
# python neon_BL_regression.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 --cls_weight $6
# python neon_PL_regression.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 --cls_weight $6

# python neon_HL_regression_no_triplet.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 --cls_weight $6
# python neon_BL_regression_no_triplet.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 --cls_weight $6
# python neon_CL_classification_no_triplet.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 --cls_weight $6
# python neon_PL_regression_no_triplet.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 --cls_weight $6

# python neon_HL_regression_discrete_triplet.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 --cls_weight $6
# python neon_BL_regression_discrete_triplet.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 --cls_weight $6
# python neon_CL_classification_discrete_triplet.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 --cls_weight $6
# python neon_PL_regression_discrete_triplet.py --margin $1 --embedding_dimension $2 --batch_size $3 --modality $4 --test_size $5 --cls_weight $6