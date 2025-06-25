#!/bin/bash
mkdir -p logs

margins=(0.1)
embedding_dims=(1024)
batch_sizes=(32) 
#cls_weights=(0.5 0.25 0.1)
modalities=('hsi' 'rgb' 'lidar')
test_sizes=(0.5)
box_weights=(0.5)
# betas=(0.2)
betas=(0.7 0.8 0.9 1.5 2.0 3.0)

# Variable to hold the ID of the last submitted job
last_job_id=""

#Single-Modal AWIR Experiments
for margin in ${margins[@]}; do
    for embed_dim in ${embedding_dims[@]}; do
        for batch_size in ${batch_sizes[@]}; do
            for modality in ${modalities[@]}; do
                for test_size in ${test_sizes[@]}; do
                    for beta in ${betas[@]}; do

                        echo "Preparing to run with margin=${margin}, embedding_dimension=${embed_dim}, batch_size=${batch_size}, modality=${modality}"

                        # Check if there's a job to depend on
                        if [[ -z $last_job_id ]]; then
                            # No dependency
                            last_job_id=$(sbatch --parsable --job-name="6_tasks_${modality}" \
                                      --output=logs/6_tasks_${margin}_${embed_dim}_${batch_size}_${modality}_${test_size}_${beta}_%j.out \
                                      run_model.sh $margin $embed_dim $batch_size $modality $test_size $beta)
                        else
                            # Submit with dependency on the completion of the last job
                            last_job_id=$(sbatch --parsable --dependency=afterok:$last_job_id \
                                      --job-name="6_tasks_${modality}" \
                                      --output=logs/6_tasks_${margin}_${embed_dim}_${batch_size}_${modality}_${test_size}_${beta}_%j.out \
                                      run_model.sh $margin $embed_dim $batch_size $modality $test_size $beta)
                        fi
                        echo "Submitted job $last_job_id with margin=${margin}, embedding_dimension=${embed_dim}, batch_size=${batch_size}, modality=${modality}"
                    done
                done
            done
        done
    done
done


# #MultiModal AWIR Experiments
# for margin in ${margins[@]}; do
#     for embed_dim in ${embedding_dims[@]}; do
#         for batch_size in ${batch_sizes[@]}; do
#             for test_size in ${test_sizes[@]}; do

#                 echo "Preparing to run with margin=${margin}, embedding_dimension=${embed_dim}, batch_size=${batch_size}"

#                 # Check if there's a job to depend on
#                 if [[ -z $last_job_id ]]; then
#                     # No dependency
#                     last_job_id=$(sbatch --parsable --job-name="Task_MM_${margin}_${embed_dim}_${batch_size}" \
#                               --output=logs/Task_MM_${margin}_${embed_dim}_${batch_size}_${test_size}_%j.out \
#                               run_model.sh $margin $embed_dim $batch_size $test_size)
#                 else
#                     # Submit with dependency on the completion of the last job
#                     last_job_id=$(sbatch --parsable --dependency=afterok:$last_job_id \
#                               --job-name="Task_MM_${margin}_${embed_dim}_${batch_size}" \
#                               --output=logs/Task_MM_${embed_dim}_${batch_size}_${test_size}_%j.out \
#                               run_model.sh $margin $embed_dim $batch_size $test_size)
#                 fi
#                 echo "Submitted job $last_job_id with margin=${margin}, embedding_dimension=${embed_dim}, batch_size=${batch_size}"
#             done
#         done
#     done
# done


# for margin in ${margins[@]}; do
#     for embed_dim in ${embedding_dims[@]}; do
#         for batch_size in ${batch_sizes[@]}; do
#             for modality in ${modality[@]}; do
#                 for cls_weight in ${cls_weights[@]}; do
#                     echo "Preparing to run with margin=${margin}, embedding_dimension=${embed_dim}, batch_size=${batch_size}"

#                     # Check if there's a job to depend on
#                     if [[ -z $last_job_id ]]; then
#                         # No dependency
#                         last_job_id=$(sbatch --parsable --job-name="${modality}_WM_${cls_weight}" \
#                                   --output=logs/${modality}_WM_${margin}_${embed_dim}_${batch_size}_${cls_weight}_%j.out \
#                                   run_model.sh $margin $embed_dim $batch_size $modality $cls_weight)
#                     else
#                         # Submit with dependency on the completion of the last job
#                         last_job_id=$(sbatch --parsable --dependency=afterok:$last_job_id \
#                                   --job-name="${modality}_WM_${cls_weight}" \
#                                   --output=logs/${modality}_WM_${margin}_${embed_dim}_${batch_size}_${cls_weight}_%j.out \
#                                   run_model.sh $margin $embed_dim $batch_size $modality $cls_weight)
#                     fi
#                     echo "Submitted job $last_job_id with margin=${margin}, embedding_dimension=${embed_dim}, batch_size=${batch_size}"
#                 done
#             done
#         done
#     done
# done

