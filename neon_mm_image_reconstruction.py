# Import libraries
from comet_ml import Experiment
import os, sys, matplotlib.pyplot as plt, numpy as np, time, random
import gc
import psutil
from TripletNetwork_Online import image_reconstruction_multi_modal, image_reconstruction_multi_modalv1, image_reconstruction_multi_modalv2
from sklearn.preprocessing import LabelEncoder
# from triplet_loss import batch_all_triplet_loss
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import classification_report, silhouette_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras import backend as K
import argparse
from collections import Counter
from sklearn.utils import resample
from tensorflow.keras.callbacks import Callback
import umap
import pickle
import datetime
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from neon_utilities import plot_pca_umap, calculate_iou, calculate_average_iou, generate_triplet_box_label_normalized, global_max_normalize, calculate_mask_iou, adjust_and_calculate_iou_mask, print_and_log_average_iou, generate_triplet_box_label_one_hot, resampling, log_one_image_per_class, log_gt_image_per_class, assign_tile_labels, weigh_features_multi, remove_bands, center_crop, log_one_hsi_recon_per_class, log_gt_hsi_per_class, log_gt_hsi_per_class, filter_and_remap_top_classes, center_crop_on_box, filter_npz_in_memory

from neon_custom_losses import keras_batch_all_triplet_continuous_loss_multimodal, ciou_loss, dice_loss, ssim_loss, mean_squared_loss, hybrid_hsi_loss, rgb_ssim_loss, lidar_ssim_loss


class CometCallback(keras.callbacks.Callback):
    def __init__(self, experiment, total_epochs):
        super(CometCallback, self).__init__()
        self.experiment = experiment
        self.global_epoch = 0  # Initialize a global epoch counter
        self.total_epochs = total_epochs  # Total number of epochs to log

    def on_epoch_end(self, epoch, logs=None):
        # Log training and validation loss using the global epoch counter
        self.experiment.log_metric("train_loss", logs["loss"], step=self.global_epoch)
        self.experiment.log_metric("val_loss", logs["val_loss"], step=self.global_epoch)
        
        # Increment the global epoch counter
        self.global_epoch += 1

                        
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--embedding_dimension", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--modality", type=str, default='rgb')
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--cls_weight", type=float, default=0.5)
    args = parser.parse_args()
    
    parameters = {
    "epochs": 200, 
    "iterations": 10,
    "runs": 5,
    "classes": 10,
    "patch_size": 6,
    "embedding_dimension": args.embedding_dimension,
    "margin" : args.margin,
    "batch_size": args.batch_size,
    "test_size": args.test_size,
    "modality": args.modality,
    "cls_weight": args.cls_weight
}

    # Load the merged data
    f = np.load('/blue/azare/zhou.m/commanet/ground_truth_points_HARVARD/NEON_big_dataset_updatedv1.npz')
    
    modality_chosen = args.modality
    print('Modality: ' + modality_chosen, flush=True)
    
    test_imgs, test_indices = center_crop_on_box(f['rgb'], f['box_labels']*10, parameters['patch_size']*10)

    print('test_img.shape: ', test_imgs.shape, ' len(test_indices): ', len(test_indices))
    
    filtered_arrays = filter_npz_in_memory('/blue/azare/zhou.m/commanet/ground_truth_points_HARVARD/NEON_big_dataset_updatedv1.npz', test_indices)
    
    
    # Extract the necessary arrays
    y_pos = filtered_arrays['box_labels']
    y_cls = filtered_arrays['class_labels'] 
    y_box = filtered_arrays['box_info'] 
    y_height = filtered_arrays['height_info']
    
    
    rgb, _ = center_crop_on_box(filtered_arrays['rgb'], filtered_arrays['box_labels']*10, parameters['patch_size']*10)
    hsi, _ = center_crop_on_box(filtered_arrays['hsi'], filtered_arrays['box_labels'], parameters['patch_size'])
    lidar, _ = center_crop_on_box(filtered_arrays['lidar'], filtered_arrays['box_labels'], parameters['patch_size'])
    
    print('rgb.shape post cropping: ', rgb.shape, ' hsi.shape post cropping: ', hsi.shape)
 
    rgb, hsi, lidar, y_cls, y_pos, y_box, y_height, class_map = filter_and_remap_top_classes(rgb, hsi, lidar, y_cls, y_pos, y_box, y_height, top_X=parameters['classes'])
    
    print("New class mapping:", class_map)
    
    
    hsi = remove_bands(hsi)
    print('hsi shape after removing bands: ', hsi.shape)
        
    X_rgb_trainval_prenorm, X_rgb_test, X_lidar_trainval_prenorm, X_lidar_test, X_hsi_trainval_prenorm, X_hsi_test, y_cls_trainval, y_cls_test, pos_feat_trainval, pos_feat_test, box_feat_trainval, box_feat_test, height_feat_trainval, height_feat_test = train_test_split(rgb, lidar, hsi, y_cls, y_pos, y_box, y_height, test_size=parameters['test_size'], stratify=y_cls, random_state=42)
    
    print('X_rgb_trainval_prenorm min max: ', X_rgb_trainval_prenorm.min(), X_rgb_trainval_prenorm.max())
    print('X_hsi_trainval_prenorm min max: ', X_hsi_trainval_prenorm.min(), X_hsi_trainval_prenorm.max())
    print('X_lidar_trainval_prenorm min max: ', X_lidar_trainval_prenorm.min(), X_lidar_trainval_prenorm.max())
    
    X_rgb_trainval = global_max_normalize(X_rgb_trainval_prenorm)
    X_lidar_trainval = global_max_normalize(X_lidar_trainval_prenorm)
    X_hsi_trainval = global_max_normalize(X_hsi_trainval_prenorm)
    
    print('X_rgb_trainval min max after norm: ', X_rgb_trainval.min(), X_rgb_trainval.max())
    print('X_hsi_trainval min max after norm: ', X_hsi_trainval.min(), X_hsi_trainval.max())
    print('X_lidar_trainval min max after norm: ', X_lidar_trainval.min(), X_lidar_trainval.max())

    # print(np.unique(y_cls_trainval, return_counts=True))
    # print(np.unique(y_cls_test, return_counts=True))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Ensure reproducibility

    # Prepare lists to hold the split data
    train_splits = []
    val_splits = []
    
    for train_index, val_index in skf.split(X_rgb_trainval, y_cls_trainval):

        # Split into train and val
        X_rgb_train, X_rgb_val = X_rgb_trainval[train_index], X_rgb_trainval[val_index]
        X_lidar_train, X_lidar_val = X_lidar_trainval[train_index], X_lidar_trainval[val_index]
        X_hsi_train, X_hsi_val = X_hsi_trainval[train_index], X_hsi_trainval[val_index]
        
        y_cls_train, y_cls_val = y_cls_trainval[train_index], y_cls_trainval[val_index]
        pos_feat_train, pos_feat_val = pos_feat_trainval[train_index], pos_feat_trainval[val_index]
        box_feat_train, box_feat_val = box_feat_trainval[train_index], box_feat_trainval[val_index]
        height_feat_train, height_feat_val = height_feat_trainval[train_index], height_feat_trainval[val_index]

        # Store the splits
        train_splits.append((X_rgb_train, X_lidar_train, X_hsi_train, y_cls_train, pos_feat_train, box_feat_train, height_feat_train))
        val_splits.append((X_rgb_val, X_lidar_val, X_hsi_val, y_cls_val, pos_feat_val, box_feat_val, height_feat_val))

    # Define the base save path
    base_save_path = os.getcwd() + "/trained_models/image_reconstruction/continuous_triplet/noL2/"
    base_save_path += f"{modality_chosen}/v1/class_{parameters['classes']}/patch_size_{parameters['patch_size']}/"
    base_save_path += f"_emb{parameters['embedding_dimension']}_batch{parameters['batch_size']}_margin"
    base_save_path += f"{parameters['margin']}_test_size{parameters['test_size']}_cls_weight{parameters['cls_weight']}"
    

    chosen_loss_rgb = rgb_ssim_loss(alpha=0.1)

    chosen_loss_lidar = mean_squared_loss

    chosen_loss_hsi = hybrid_hsi_loss


    for run in range(parameters['runs']):
        experiment = Experiment(
          api_key="xuGNxq43n5AvOfi7zn0JavYDR",
          project_name="neon",
          workspace="zhou-m"
        )
        
        run_number = run + 1  # Increment run number
        checkpoint_save_path = f"{base_save_path}_best_val_model_run_{run_number}.h5"
        
        experiment_name = f"{modality_chosen}_CTL_image_recon_{run_number}"
        experiment.set_name(experiment_name)
        
        experiment.log_parameters(parameters)
        
        X_rgb_train, X_lidar_train, X_hsi_train, y_cls_train, pos_feat_train, box_feat_train, height_feat_train = train_splits[run]
        X_rgb_val, X_lidar_val, X_hsi_val, y_cls_val, pos_feat_val, box_feat_val, height_feat_val = val_splits[run]

        print('y_cls_train unique before resampling: ', np.unique(y_cls_train, return_counts=True))
        print('box_feat_train shape before resampling: ', box_feat_train.shape)
        print('height_feat_train shape before resampling: ', height_feat_train.shape)
        print('X_rgb_train shape before resampling: ', X_rgb_train.shape)
        print('X_lidar_train shape before resampling: ', X_lidar_train.shape)
        print('X_hsi_train shape before resampling: ', X_hsi_train.shape)
        
        y_cls_train, X_rgb_train, X_lidar_train, X_hsi_train, pos_feat_train, box_feat_train, height_feat_train = resampling(y_cls_train, X_rgb_train, X_lidar_train, X_hsi_train, pos_feat_train, box_feat_train, height_feat_train)
                
        print('y_cls_train unique after resampling: ', np.unique(y_cls_train, return_counts=True), y_cls_train.shape)
        print('box_feat_train shape after resampling: ', box_feat_train.shape)
        print('height_feat_train shape after resampling: ', height_feat_train.shape)
        print('X_rgb_train shape after resampling: ', X_rgb_train.shape)
        print('X_lidar_train shape before resampling: ', X_lidar_train.shape)
        print('X_hsi_train shape before resampling: ', X_hsi_train.shape)
        
        
#         # Get unique classes and their first occurrence indices
#         unique_classes, indices = np.unique(y_cls_train, return_index=True)

#         # Extract corresponding data for each class
#         y_cls_train = y_cls_train[indices]
#         X_rgb_train = X_rgb_train[indices]
#         X_lidar_train = X_lidar_train[indices]
#         X_hsi_train = X_hsi_train[indices]
#         box_feat_train = box_feat_train[indices]
#         height_feat_train = height_feat_train[indices]
        
#         print('y_cls_train unique after 1 resampling: ', np.unique(y_cls_train, return_counts=True), y_cls_train.shape)
#         print('box_feat_train shape after 1 resampling: ', box_feat_train.shape)
#         print('height_feat_train shape after 1 resampling: ', height_feat_train.shape)
#         print('X_rgb_train shape after 1 resampling: ', X_rgb_train.shape)
#         print('X_lidar_train shape after 1 resampling: ', X_lidar_train.shape)
#         print('X_hsi_train shape after 1 resampling: ', X_hsi_train.shape)
        
        print(f"Fold {run + 1}:", flush=True)

        
        # Step 2: Convert integer labels to categorical format (one-hot encoding)
        y_train_categorical = tf.keras.utils.to_categorical(y_cls_train)
        y_val_categorical = tf.keras.utils.to_categorical(y_cls_val)
        
        X_rgb_train = X_rgb_train.astype(np.float32)
        X_lidar_train = X_lidar_train.astype(np.float32)
        X_hsi_train = X_hsi_train.astype(np.float32)
        
        train_features_dict = {
        "class_labels": y_train_categorical,
        "pos_features": pos_feat_train,
        "box_features": box_feat_train,
        "height_features": height_feat_train,
    }

        val_features_dict = {
        "class_labels": y_val_categorical,
        "pos_features": pos_feat_val,
        "box_features": box_feat_val,
        "height_features": height_feat_val,
    }
            
        weights_dict = {
        "class_labels": 0.5,
        "pos_features": 1.0,
        "box_features": 1.0,
        "height_features": 1.0,
    }
        
        normalize_dict = {
        "class_labels": False,
        "pos_features": True,
        "box_features": True,
        "height_features": True,
    }

        combined_train = weigh_features_multi(train_features_dict, weights_dict, normalize_dict)
        combined_val = weigh_features_multi(val_features_dict, weights_dict, normalize_dict)
        
        print('combined_train[0]: ', combined_train[0])
        print('combined_train.shape: ', combined_train.shape)
        print('combined_val.shape: ', combined_val.shape)
        
        print('combined_train[0]: ', combined_train[0])
        print('combined_train.shape: ', combined_train.shape)
        print('combined_val.shape: ', combined_val.shape)

        RGB_DIM = X_rgb_train[0].shape
        LIDAR_DIM = X_lidar_train[0].shape
        HSI_DIM = X_hsi_train[0].shape
        EMB_DIM = parameters['embedding_dimension']
        
        model = image_reconstruction_multi_modalv1(RGB_DIM, HSI_DIM, LIDAR_DIM, EMB_DIM)
        
        # Compile the model
        model.compile(optimizer=Adam(0.0001), loss=[keras_batch_all_triplet_continuous_loss_multimodal(EMB_DIM=EMB_DIM), chosen_loss_rgb, chosen_loss_hsi, chosen_loss_lidar], loss_weights = [1, 1, 0.001, 1])
    
        # Define the ModelCheckpoint callback
        model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_save_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=0
    )
        comet_callback = CometCallback(experiment, total_epochs=parameters['iterations']*parameters['epochs'])

        target_train = [combined_train, X_rgb_train, X_hsi_train, X_lidar_train]
        target_val = [combined_val, X_rgb_val, X_hsi_val, X_lidar_val]
        input_data_train = [X_rgb_train, X_hsi_train, X_lidar_train]
        input_data_val = [X_rgb_val, X_hsi_val, X_lidar_val]

        for i in range(parameters['iterations']):
            
            # Fit the model on the current modality
            model.fit(
                input_data_train,  # X_rgb_train, X_lidar_train, or X_hsi_train
                target_train, 
                validation_data=(input_data_val, target_val), 
                epochs=parameters['epochs'], 
                batch_size=parameters['batch_size'], 
                verbose=0, 
                callbacks=[model_checkpoint_callback, comet_callback]
            )

            # # Fit the model on the current modality
            # model.fit(
            #     input_data_train,  # X_rgb_train, X_lidar_train, or X_hsi_train
            #     target_train, 
            #     validation_data=(input_data_val, target_val), 
            #     epochs=parameters['epochs'], 
            #     batch_size=parameters['batch_size'], 
            #     verbose=0, 
            #     shuffle=True,
            #     callbacks=[model_checkpoint_callback, comet_callback]
            # )

            # Save model every 5th iteration
            if (i + 1) % 5 == 0:
                save_path = os.path.join(
                    base_save_path,
                    f"run_{run+1}_iteration_{i + 1}_model.h5"
                )
                model.save(save_path)
                print(f"Model saved for run {run+1}, iteration {i + 1}: {save_path}")




        experiment.end()