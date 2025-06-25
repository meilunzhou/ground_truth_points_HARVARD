from comet_ml import Experiment
import os, sys, matplotlib.pyplot as plt, numpy as np, time, random
import gc
import psutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from triplet_loss import batch_all_triplet_loss
from dual_triplet_loss_clf_aspect import batch_all_triplet_double_loss
from continuous_triplet_loss import batch_all_continuous_triplet_loss, batch_all_continuous_triplet_loss_final_lambda1, batch_all_continuous_triplet_loss_final_multimodal, batch_all_triplet_continuous_loss_class_margin

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

# def keras_batch_all_triplet_continuous_loss(margin = 0.0, B = 1.0): 
#     def compute_keras_batch_all_triplet_continuous_loss(box_features, embeddings):
#         loss = batch_all_continuous_triplet_loss(box_features, embeddings,  margin=0.0, B=1.0, squared=False)
#         return loss
#     return compute_keras_batch_all_triplet_continuous_loss

def keras_batch_all_triplet_continuous_loss_class_margin(margin = 0.01, class_margin=0.2): 
    def compute_batch_all_continuous_triplet_loss_class_margin(label_vector, embeddings):
        loss = batch_all_triplet_continuous_loss_class_margin(label_vector, embeddings, margin=0.01, class_margin=0.2, squared=False)
        return loss
    return compute_batch_all_continuous_triplet_loss_class_margin


def keras_batch_all_triplet_continuous_loss_final(margin = 0.01): 
    def compute_keras_batch_all_triplet_continuous_loss_final(box_features, embeddings):
        loss = batch_all_continuous_triplet_loss_final_lambda1(box_features, embeddings, margin=0.01, squared=False)
        return loss
    return compute_keras_batch_all_triplet_continuous_loss_final

def keras_batch_all_triplet_continuous_loss_multimodal(margin = 0.01, EMB_DIM=1024): 
    def compute_keras_batch_all_triplet_continuous_loss_multimodal(annotation_vector, embeddings):
        loss = batch_all_continuous_triplet_loss_final_multimodal(annotation_vector, embeddings, margin=0.01, squared=False, EMB_DIM=1024)
        return loss
    return compute_keras_batch_all_triplet_continuous_loss_multimodal

def keras_batch_all_triplet_continuous_loss_final_weighted(margin = 0.01): 
    def compute_keras_batch_all_triplet_continuous_loss_final_weighted(unweighted_features, embeddings_weights):
        weights = embeddings_weights[:, -3:]
        embeddings = embeddings_weights[:, :-3]
        # Split features and weights into corresponding groups
        f1, f2, f3 = tf.split(unweighted_features, [4, 3, 2], axis=-1)
        w1, w2, w3 = tf.split(weights, [1, 1, 1], axis=-1)

        # Apply weights to the feature groups
        weighted_f1 = f1 * w1
        weighted_f2 = f2 * w2
        weighted_f3 = f3 * w3

        # Concatenate weighted features back together
        features = tf.concat([weighted_f1, weighted_f2, weighted_f3], axis=-1)

        # Compute the triplet loss using the weighted features
        loss = batch_all_continuous_triplet_loss_final_lambda1(features, embeddings, margin=margin, squared=False)
        return loss
    
    return compute_keras_batch_all_triplet_continuous_loss_final_weighted


def keras_batch_all_triplet_double_loss(margin = 0.0, box_weight = 0.5): 
    def compute_keras_batch_all_triplet_double_loss(labels, embeddings):
        labels_clf = labels[:,0]
        labels_AR = labels[:,1]

        loss = batch_all_triplet_double_loss(labels_clf, labels_AR, embeddings, margin, box_weight)
        return loss
    return compute_keras_batch_all_triplet_double_loss


def keras_batch_all_triplet_loss(margin = 0.0) : 
    def compute_keras_batch_all_triplet_loss(labels, embeddings) :
        labels = K.squeeze(labels, axis= -1)
        loss, fraction = batch_all_triplet_loss(labels, embeddings, margin) 
        return loss
    return compute_keras_batch_all_triplet_loss

            
def ciou_loss(y_true, y_pred, alpha=0.5):
    """
    CIoU loss for bounding box prediction.

    Parameters:
    - y_true: Ground truth bounding box coordinates (xcenter, ycenter, width, height) (normalized)
    - y_pred: Predicted bounding box coordinates (xcenter, ycenter, width, height) (normalized)
    - alpha: Trade-off parameter for aspect ratio term (default 0.5)

    Returns:
    - CIoU loss
    """
    # Extract coordinates
    x_center_true, y_center_true, width_true, height_true = tf.split(y_true, 4, axis=-1)
    x_center_pred, y_center_pred, width_pred, height_pred = tf.split(y_pred, 4, axis=-1)

    # Calculate corners of the boxes
    x1_true = x_center_true - (width_true / 2)
    x2_true = x_center_true + (width_true / 2)
    y1_true = y_center_true - (height_true / 2)
    y2_true = y_center_true + (height_true / 2)

    x1_pred = x_center_pred - (width_pred / 2)
    x2_pred = x_center_pred + (width_pred / 2)
    y1_pred = y_center_pred - (height_pred / 2)
    y2_pred = y_center_pred + (height_pred / 2)

    ### CIoU Loss Calculation ###

    # Calculate intersection
    x1_inter = tf.maximum(x1_true, x1_pred)
    y1_inter = tf.maximum(y1_true, y1_pred)
    x2_inter = tf.minimum(x2_true, x2_pred)
    y2_inter = tf.minimum(y2_true, y2_pred)

    inter_area = tf.maximum(0.0, x2_inter - x1_inter) * tf.maximum(0.0, y2_inter - y1_inter)

    # Calculate union
    true_area = width_true * height_true
    pred_area = width_pred * height_pred
    union_area = true_area + pred_area - inter_area

    # Calculate IoU
    iou = inter_area / tf.maximum(union_area, 1e-10)

    # Calculate center points and diagonal of the smallest enclosing box
    center_true = tf.concat([x_center_true, y_center_true], axis=-1)
    center_pred = tf.concat([x_center_pred, y_center_pred], axis=-1)

    # Distance between centers
    center_distance = tf.reduce_sum(tf.square(center_true - center_pred), axis=-1)

    # Calculate enclosing box diagonal
    c_x1 = tf.minimum(x1_true, x1_pred)
    c_x2 = tf.maximum(x2_true, x2_pred)
    c_y1 = tf.minimum(y1_true, y1_pred)
    c_y2 = tf.maximum(y2_true, y2_pred)

    c_area = (c_x2 - c_x1) * (c_y2 - c_y1)
    c_diagonal_squared = tf.square(c_x2 - c_x1) + tf.square(c_y2 - c_y1)

    # Aspect ratio term
    v = (4 / (tf.square(tf.constant(np.pi)))) * tf.square(tf.atan(height_true / width_true) - tf.atan(height_pred / width_pred))

    # CIoU loss
    ciou_loss_value = 1 - iou + (center_distance / tf.maximum(c_diagonal_squared, 1e-10)) + alpha * v

    return tf.reduce_mean(ciou_loss_value)

def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss for binary segmentation.
    
    Args:
        y_true (tensor): Ground truth binary mask.
        y_pred (tensor): Predicted binary mask.
        smooth (float): Smoothing factor to avoid division by zero.
        
    Returns:
        dice_loss (float): Dice loss value.
    """
    # Flatten to simplify calculation
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    # Calculate intersection and union
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    
    # Dice coefficient and Dice loss
    dice_coefficient = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_coefficient
    
    return dice_loss

# Custom loss function that combines SSIM and MSE losses
def ssim_loss(alpha=0.1):

    # Define the SSIM loss
    def inner_ssim_loss(y_true, y_pred):

        loss = 1 - tf.image.ssim(y_true, y_pred, 1.0, filter_size= 3)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis= (-3, -2, -1))
        combined_loss = tf.reduce_mean(alpha * loss + (1 - alpha) * mse_loss)

        return combined_loss

    return inner_ssim_loss

# Custom loss function that combines SSIM and MSE losses
def rgb_ssim_loss(alpha=0.1):

    # Define the SSIM loss
    def rgb_inner_ssim_loss(y_true, y_pred):

        loss = 1 - tf.image.ssim(y_true, y_pred, 1.0, filter_size = 7)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis= (-3, -2, -1))
        combined_loss = tf.reduce_mean(alpha * loss + (1 - alpha) * mse_loss)

        return combined_loss

    return rgb_inner_ssim_loss

# Custom loss function that combines SSIM and MSE losses
def lidar_ssim_loss(alpha=0.1):

    # Define the SSIM loss
    def lidar_inner_ssim_loss(y_true, y_pred):

        loss = 1 - tf.image.ssim(y_true, y_pred, 1.0, filter_size= 3)
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis= (-3, -2, -1))
        combined_loss = tf.reduce_mean(alpha * loss + (1 - alpha) * mse_loss)

        return combined_loss

    return lidar_inner_ssim_loss


def vae_kl_loss(y_true, y_pred, EMB_DIM=1024, smooth=1e-6):
    """
    """
    z_mean = y_pred[:,:EMB_DIM]
    z_log_var = y_pred[:, EMB_DIM:]
    
    kl_loss = 1. + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.mean(kl_loss, axis=-1)
    kl_loss *= -0.5
    
    kl_loss = tf.reduce_mean(kl_loss)
    
    return kl_loss

def multilabel_focal_loss(alpha=0.25, gamma=2.0):
    """
    Multilabel focal loss function with flexible alpha (scalar or class-wise list).
    
    Args:
        alpha (float or list): Balancing factor for positive samples.
                               If a scalar is provided, the same alpha is used for all classes.
                               If a list is provided, it must have a length equal to the number of classes.
        gamma (float): Focusing parameter to reduce the contribution of easy examples.
        
    Returns:
        A function that computes the multilabel focal loss.
    """
    def loss(y_true, y_pred):
        # Ensure alpha is a tensor of the correct shape
        alpha_tensor = tf.convert_to_tensor(alpha, dtype=tf.float32)
        if len(alpha_tensor.shape) == 0:  # If scalar, expand to match y_true shape
            alpha_tensor = tf.fill(tf.shape(y_true)[1:], alpha_tensor)
        
        # Clip predictions to avoid log(0) errors
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Compute the cross-entropy term
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Compute the focal weight
        focal_weight = tf.math.pow(1 - y_pred, gamma)
        
        # Apply alpha (class-wise or scalar) and focal weight
        loss_per_class = alpha_tensor * focal_weight * cross_entropy
        
        # Return the mean loss across all samples and classes
        return tf.reduce_mean(tf.reduce_sum(loss_per_class, axis=-1))
    
    return loss

def mean_squared_loss(y_true, y_pred):
    # Compute element-wise squared error
    mse_loss = tf.square(y_true - y_pred)
    
    # Reduce across all spatial and channel dimensions (height, width, channels)
    mse_loss = tf.reduce_mean(mse_loss, axis=[1, 2, 3])
    
    # Reduce across batch dimension
    mse_loss = tf.reduce_mean(mse_loss) 
    
    return mse_loss


def spectral_angle_loss(y_true, y_pred):
    """
    Spectral Angle Mapper (SAM) loss for hyperspectral image reconstruction.
    Preserves spectral shape by minimizing angular differences between spectra.
    """

    # Normalize each spectral vector
    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)  # Normalize across spectral bands
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)

    # Compute cosine similarity
    cosine_similarity = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)  

    # Convert cosine similarity to angle (higher angle → higher error)
    spectral_angle = tf.acos(tf.clip_by_value(cosine_similarity, -1.0, 1.0))

    # Take the mean spectral angle across all pixels and batch
    return tf.reduce_mean(spectral_angle)




def spectral_correlation_loss(y_true, y_pred):
    """
    Encourages the correlation between spectral bands to remain consistent.
    """

    # Flatten to (batch_size, num_pixels, num_bands)
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1, tf.shape(y_true)[-1]])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1, tf.shape(y_pred)[-1]])

    # Compute correlation matrices
    y_true_corr = tf.linalg.matmul(y_true_flat, y_true_flat, transpose_a=True)  # (batch, bands, bands)
    y_pred_corr = tf.linalg.matmul(y_pred_flat, y_pred_flat, transpose_a=True)

    # Compute Frobenius norm difference
    loss = tf.norm(y_true_corr - y_pred_corr, ord='fro', axis=[-2, -1])

    return tf.reduce_mean(loss)


def hybrid_hsi_loss(y_true, y_pred, alpha=0.5):
    """
    Hybrid loss combining Mean Squared Error (MSE) and Spectral Angle Loss (SAL) and Spectral Correlation Loss (SCL).
    α controls the balance between intensity preservation and spectral shape preservation.
    """

    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))  # Standard MSE
    sal_loss = spectral_angle_loss(y_true, y_pred)  # Spectral similarity
    scl_loss = spectral_correlation_loss(y_true, y_pred)
    
    combined_loss = alpha * mse_loss + (1 - alpha) * sal_loss + 0.1 * scl_loss

    return tf.reduce_mean(combined_loss)  # Weighted combination