from comet_ml import Experiment
import os, sys, matplotlib.pyplot as plt, numpy as np, time, random
import gc
import psutil
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from dual_triplet_loss_clf_aspect import batch_all_triplet_loss
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
from scipy.spatial.distance import cdist
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import f_oneway

def filter_npz_in_memory(npz_path, test_indices):
    '''
    Loads a .npz file, removes the specified indices from all arrays,
    and returns the filtered arrays as a dictionary.

    Parameters:
        npz_path (str): Path to the .npz file.
        test_indices (list or array): Indices to remove.

    Returns:
        dict: Dictionary of filtered arrays keyed by the original npz keys.
    '''
    data = np.load(npz_path)
    
    # If no indices to remove, return everything unchanged
    if test_indices is None or len(test_indices) == 0:
        return {k: v for k, v in data.items()}
    
    test_indices = np.array(test_indices)
    n_samples = len(next(iter(data.values())))
    mask = np.ones(n_samples, dtype=bool)
    mask[test_indices] = False

    # Return filtered arrays in a dictionary
    filtered = {k: v[mask] for k, v in data.items()}
    return filtered


def center_crop_on_box(array, boxes, crop_size):
    '''
    Crops a centered (N x N) square around the center of each bounding box for all samples.
    Skips samples where the crop would exceed image bounds.

    Parameters:
        array (numpy.ndarray): Input array of shape (num_samples, x, y, bands).
        boxes (numpy.ndarray): Array of bounding boxes with shape (num_samples, 4),
                               each in the form [xmin, xmax, ymin, ymax].
        crop_size (int): Size of the square to crop (N x N).
    
    Returns:
        cropped_array (numpy.ndarray): Cropped array of shape (valid_samples, N, N, bands).
        skipped_indices (list): Indices of samples skipped due to out-of-bounds crop.
    '''
    num_samples, x_dim, y_dim, bands = array.shape
    half_crop = crop_size // 2

    cropped_list = []
    skipped_indices = []

    for i in range(num_samples):
        xmin, xmax, ymin, ymax = boxes[i]

        # Compute center of the bounding box and cast to int
        center_x = int(round((xmin + xmax) / 2))
        center_y = int(round((ymin + ymax) / 2))

        # Compute crop bounds and cast to int
        start_x = int(center_x - half_crop)
        end_x = int(center_x + half_crop)
        start_y = int(center_y - half_crop)
        end_y = int(center_y + half_crop)

        # Check bounds
        if start_x < 0 or end_x > x_dim or start_y < 0 or end_y > y_dim:
            skipped_indices.append(i)
            continue

        # Crop and store
        crop = array[i, start_x:end_x, start_y:end_y, :]
        cropped_list.append(crop)

    # Stack cropped results
    if cropped_list:
        cropped_array = np.stack(cropped_list)
    else:
        cropped_array = np.empty((0, crop_size, crop_size, bands), dtype=array.dtype)

    return cropped_array, skipped_indices


def compute_label_weights(y_cls, feature_groups, group_names=None, group_weights=None):
    """
    Compute label vector weights for multi-task learning using mutual information and ANOVA.

    Parameters:
        y_cls (np.ndarray): Class label array of shape (N,)
        feature_groups (list of np.ndarray): List of arrays of shape (N, D_i) for each annotation group
        group_names (list of str): Optional list of group names, e.g., ['class', 'box', 'height']
        group_weights (list of float): Optional list of relative weights for each group, must sum to 1

    Returns:
        np.ndarray: Label weight vector of shape (total_dims,)
    """
    N = len(y_cls)
    num_classes = len(np.unique(y_cls))
    onehot_cls = np.eye(num_classes)[y_cls]  # (N, num_classes)

    # Initialize lists
    all_weights = []

    # Default group names and weights
    if group_names is None:
        group_names = ['group_{}'.format(i) for i in range(len(feature_groups))]
    if group_weights is None:
        group_weights = [1.0 / (len(feature_groups) + 1)] * (len(feature_groups) + 1)

    # ----- Class weights -----
    class_weights = np.ones(num_classes)
    class_weights = group_weights[0] * (class_weights / np.sum(class_weights))
    all_weights.append(class_weights)

    # ----- Feature group weights -----
    for idx, group in enumerate(feature_groups):
        mi = mutual_info_classif(group, y_cls)
        f_vals = []
        for i in range(group.shape[1]):
            groups = [group[y_cls == c, i] for c in np.unique(y_cls)]
            f_stat, _ = f_oneway(*groups)
            f_vals.append(f_stat)

        # Normalize and combine MI and F
        mi = mi / (np.max(mi) + 1e-8)
        f_vals = np.array(f_vals) / (np.max(f_vals) + 1e-8)
        scores = 0.5 * (mi + f_vals)

        weights = group_weights[idx + 1] * (scores / np.sum(scores))
        all_weights.append(weights)

    label_weights = np.concatenate(all_weights)
    return label_weights



def filter_and_remap_top_classes(rgb, hsi, lidar, y_cls, y_pos, y_box, y_height, top_X):
    """
    Filters the dataset to include only samples from the top X most frequent classes
    and remaps class labels to a 0-based range.

    Parameters:
        rgb (np.ndarray): RGB images array.
        hsi (np.ndarray): Hyperspectral images array.
        lidar (np.ndarray): LiDAR data array.
        y_cls (np.ndarray): Class labels.
        y_box (np.ndarray): Bounding box info.
        y_height (np.ndarray): Height info.
        top_X (int): Number of most frequent classes to retain.

    Returns:
        tuple: Filtered and remapped (rgb, hsi, lidar, y_cls, y_box, y_height), class_map
    """
    # Count class occurrences and get top X
    class_counts = Counter(y_cls)
    most_common_classes = [cls for cls, _ in class_counts.most_common(top_X)]

    # Create mask to filter samples
    mask = np.isin(y_cls, most_common_classes)

    # Apply mask to all arrays
    rgb_filtered = rgb[mask]
    hsi_filtered = hsi[mask]
    lidar_filtered = lidar[mask]
    y_cls_filtered = y_cls[mask]
    y_pos_filtered = y_pos[mask]
    y_box_filtered = y_box[mask]
    y_height_filtered = y_height[mask]

    # Remap class labels to 0-based indices
    class_map = {old_label: new_label for new_label, old_label in enumerate(sorted(most_common_classes))}
    y_cls_remapped = np.array([class_map[label] for label in y_cls_filtered])

    return rgb_filtered, hsi_filtered, lidar_filtered, y_cls_remapped, y_pos_filtered, y_box_filtered, y_height_filtered, class_map

def center_crop(array, crop_size):
    '''
    Crops a centered (N x N) square from the spatial dimensions of the input array.
    
    Parameters:
        array (numpy.ndarray): Input array of shape (num_samples, x, y, bands).
        crop_size (int): Size of the square to crop (N x N).
    
    Returns:
        numpy.ndarray: Cropped array of shape (num_samples, N, N, bands).
    '''
    num_samples, x_dim, y_dim, bands = array.shape
    
    if crop_size > x_dim or crop_size > y_dim:
        raise ValueError("Crop size must be smaller than the spatial dimensions of the input array.")

    # Compute crop start indices
    start_x = (x_dim - crop_size) // 2
    start_y = (y_dim - crop_size) // 2

    # Crop from center
    cropped_array = array[:, start_x:start_x + crop_size, start_y:start_y + crop_size, :]
    
    return cropped_array

def remove_bands(array):
    '''
    Removes specific bands from the last dimension of the input array.
    
    Parameters:
        array (numpy.ndarray): Input array of shape (batch, x, y, bands).
    
    Returns:
        numpy.ndarray: Modified array with selected bands removed.
    '''
    modified_array = np.copy(array)
    print("Original shape:", modified_array.shape)
    
    # Remove bands from the last dimension (bands axis)
    modified_array = np.delete(modified_array, np.r_[419:426], axis=-1)
    modified_array = np.delete(modified_array, np.r_[283:315], axis=-1)
    modified_array = np.delete(modified_array, np.r_[192:210], axis=-1)
    
    print("Modified shape:", modified_array.shape)
    return modified_array

def create_label_mask(X_rgb, y_cls, y_box):
    """
    Create an array of zeros with the same shape as X_rgb and replace pixels within bounding boxes 
    with the corresponding class labels.

    Parameters:
    - X_rgb (numpy.ndarray): Array of images with shape (num_samples, height, width, channels).
    - y_cls (numpy.ndarray): Array of class labels as strings (num_samples,).
    - y_box (numpy.ndarray): Array of bounding box coordinates (num_samples, 4) 
                              with [xmin, xmax, ymin, ymax].

    Returns:
    - label_masks (numpy.ndarray): Array with the same spatial shape as X_rgb, where bounding boxes 
                                    are filled with the encoded class labels.
    - label_mapping (dict): Dictionary mapping original class labels to their integer encodings.
    """
    num_samples, height, width, _ = X_rgb.shape
    label_masks = np.zeros((num_samples, height, width), dtype=np.int32)+3
    
    # Encode class labels as integers
    unique_labels = np.unique(y_cls)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_mapping[label] for label in y_cls])
    
    for i in range(num_samples):
        cls = encoded_labels[i]  # Encoded class label
        xmin, xmax, ymin, ymax = y_box[i]  # Convert box coordinates to integers
        
        # Replace pixels in the bounding box with the class label
        label_masks[i, ymin:ymax, xmin:xmax] = cls
    
    return label_masks, label_mapping


def log_weight_distribution_with_errorbars(experiment, weights, name, epoch):
    """
    Create a bar plot for averaged weights with standard deviation error bars and log it to Comet.

    Parameters:
    - experiment: Comet Experiment object
    - weights: 2D array with shape (num_samples, 3), where columns are weights for ['box', 'class', 'position']
    - name: String specifying the label for the weights (e.g., 'Train' or 'Validation')
    - epoch: Current epoch number
    """
    # Calculate mean and standard deviation for the weights
    means = weights.mean(axis=0)
    stds = weights.std(axis=0)
    
    # Validate input
    categories = ['box', 'class', 'position']
    if len(means) != len(categories):
        raise ValueError(f"Expected {len(categories)} weights, but got {len(means)} weights: {means}")
    
    # Create the figure
    fig, ax = plt.subplots()
    
    # Plot bars with error bars
    ax.bar(categories, means, yerr=stds, color='blue', alpha=0.7, capsize=5, label=name)
    
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add labels and title
    ax.set_ylabel('Weight Value')
    ax.set_title(f'{name} Weights with Error Bars by Category (Epoch {epoch})')
    ax.legend()
    
    # Log the figure to Comet
    experiment.log_figure(figure_name=f"{name} Weight Distribution with Error Bars Epoch {epoch}", figure=fig)
    plt.close(fig)
    

def weigh_features_multi(features_dict, weights_dict, normalize_dict=None):
    """
    Scale multiple arrays and concatenate them to balance their contributions.

    Parameters:
        features_dict (dict): A dictionary where keys are feature names and values are np.ndarray arrays.
                             Each array should have shape (n_samples, n_features).
        weights_dict (dict): A dictionary where keys are feature names and values are weights for scaling.
                             Keys must match those in features_dict.
        normalize_dict (dict, optional): A dictionary where keys are feature names and values are booleans indicating
                                         whether to normalize the corresponding array. Defaults to None (all normalized).

    Returns:
        np.ndarray: Weighted concatenation of all normalized and scaled features.
    """
    scaled_features = []

    # Find the feature with the largest length (number of features)
    max_length = max(array.shape[1] for array in features_dict.values())

    for feature_name, feature_array in features_dict.items():
        if feature_name not in weights_dict:
            raise ValueError(f"Weight for '{feature_name}' not provided in weights_dict.")

        normalize = normalize_dict.get(feature_name, True) if normalize_dict else True

        if normalize:
            # Normalize the feature array between 0 and 1
            feature_min = feature_array.min(axis=0)
            feature_max = feature_array.max(axis=0)
            feature_normalized = (feature_array - feature_min) / (feature_max - feature_min)
        else:
            feature_normalized = feature_array

        # Scale the feature array by its corresponding weight, adjusted by max_length
        weight = weights_dict[feature_name] * (max_length / feature_array.shape[1])
        scaled_feature = feature_normalized * weight

        scaled_features.append(scaled_feature)

    # Concatenate all scaled features
    return np.hstack(scaled_features)


def weigh_features(box_features, class_labels, class_weight=1.0):
    """
    Scale box features and class labels to balance their contributions.
    
    Parameters:
        box_features (np.ndarray): Array of box features (n_samples, n_box_features).
        class_labels (np.ndarray): Array of one-hot encoded class labels (n_samples, n_classes).
        class_weight (float): Weight to apply to the class labels' contribution.
    
    Returns:
        np.ndarray: Weighted concatenation of box features and class labels.
    """
    # Normalize box features between 0 and 1
    box_features_min = box_features.min(axis=0)
    box_features_max = box_features.max(axis=0)
    box_features_normalized = (box_features - box_features_min) / (box_features_max - box_features_min)
    
    # Determine scaling factors
    num_box_features = box_features_normalized.shape[1]
    num_class_labels = class_labels.shape[1]
    
    box_feature_scale = 1.0
    class_label_scale = class_weight * (num_box_features / num_class_labels)
    
    # Scale features
    scaled_box_features = box_features_normalized * box_feature_scale
    scaled_class_labels = class_labels * class_label_scale
    
    # Concatenate scaled features
    return np.hstack([scaled_box_features, scaled_class_labels])

def plot_pca_umap(embeddings, string_labels, experiment, title):
    """
    Plots 3D PCA and 2D UMAP of the unimodal embeddings with different colors for different classes.
    
    Args:
        embeddings (numpy array): The unimodal embeddings of shape (num_samples, EMB_DIM).
        string_labels (list of str): The corresponding string labels for the embeddings.
    """
    # Perform PCA to reduce dimensions to 3
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(embeddings)

    # Perform UMAP to reduce dimensions to 2
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_result = reducer.fit_transform(embeddings)

    # Create a figure with 2 subplots: one for 3D PCA and one for 2D UMAP
    fig = plt.figure(figsize=(14, 7))

    # Create a 3D plot for PCA
    ax1 = fig.add_subplot(121, projection='3d')

    # Collect handles and labels for a shared legend
    handles = []
    
    unique_classes = np.unique(string_labels)
    cmap = plt.get_cmap('tab10')  # Use tab10 colormap

    # Only use the first three colors (for cow, deer, horse)
    # color_map = {class_name: cmap(i) for i, class_name in enumerate(['cow', 'deer', 'horse'])}
    color_map = {class_name: cmap(i) for i, class_name in enumerate(unique_classes)}

    # Plot 3D PCA result
    for class_name in color_map.keys():
        indices = [idx for idx, label in enumerate(string_labels) if label == class_name]
        color = color_map[class_name]

        # Plot embeddings
        handle = ax1.scatter(pca_result[indices, 0], pca_result[indices, 1], pca_result[indices, 2],
                             label=f'{class_name}', alpha=0.7, marker='o', color=color)
        
        # Only append one handle per class for the legend
        if class_name == 'cow':
            handles.append(handle)

    ax1.set_title('3D PCA Visualization of ' + title)
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.set_zlabel('PCA Component 3')

    # Plot UMAP result in 2D
    ax2 = fig.add_subplot(122)
    for class_name in color_map.keys():
        indices = [idx for idx, label in enumerate(string_labels) if label == class_name]
        color = color_map[class_name]

        # Plot embeddings
        ax2.scatter(umap_result[indices, 0], umap_result[indices, 1],
                    label=f'{class_name}', alpha=0.7, marker='o', color=color)

    ax2.set_title('2D UMAP Visualization of ' + title)

    # Create a single legend for both plots
    ax1.legend(loc='center left', bbox_to_anchor=(-0.4, 0.5))

    # Adjust layout to make space for the legend
    #plt.tight_layout()
    experiment.log_figure(figure=fig)
    
    
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for bounding boxes in the format (xcenter, ycenter, width, height).

    Parameters:
    - box1: First bounding box in the format (xcenter, ycenter, width, height)
    - box2: Second bounding box in the format (xcenter, ycenter, width, height)

    Returns:
    - IoU: Intersection over Union value
    """
    # Unpack boxes
    x_center1, y_center1, width1, height1 = box1
    x_center2, y_center2, width2, height2 = box2

    # Calculate half dimensions
    half_width1 = width1 / 2
    half_height1 = height1 / 2
    half_width2 = width2 / 2
    half_height2 = height2 / 2

    # Calculate intersection coordinates
    inter_x_min = max(x_center1 - half_width1, x_center2 - half_width2)
    inter_x_max = min(x_center1 + half_width1, x_center2 + half_width2)
    inter_y_min = max(y_center1 - half_height1, y_center2 - half_height2)
    inter_y_max = min(y_center1 + half_height1, y_center2 + half_height2)

    # Calculate intersection area
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate areas
    box1_area = width1 * height1
    box2_area = width2 * height2

    # Calculate IoU
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / (union_area + 1e-10)
    
    return iou

def calculate_average_iou(box_pred, y_test_box, labels_test):
    total_iou = 0
    num_samples = len(labels_test)

    for idx in range(num_samples):
        pred_box = (box_pred[idx, :]).astype(int)
        gt_box = (y_test_box[idx, :]).astype(int)

        iou = calculate_iou(pred_box, gt_box)
        total_iou += iou

    average_iou = total_iou / num_samples
    return average_iou

def generate_triplet_box_label_standardized(y_box, y_cls, n_clusters=6):
    # Convert y_box to width and height
    width = y_box[:, 1] - y_box[:, 0]
    height = y_box[:, 3] - y_box[:, 2]
    area = width * height

    # Calculate perimeter
    perimeter = 2 * (width + height)

    # Calculate compactness
    compactness = area / (perimeter ** 2)

    # Calculate symmetric squareness deviation
    symmetric_squareness_deviation = 1 - np.minimum(width / height, height / width)

    # Create a DataFrame with the metrics
    df = pd.DataFrame({
        "class": y_cls,
        "width": width,
        "height": height,
        "compactness": compactness,
        "symmetric_squareness_deviation": symmetric_squareness_deviation
    })

    # Standardize each feature

    df['std_width'] = (df['width'] - df['width'].mean()) / df['width'].std()
    df['std_height'] = (df['height'] - df['height'].mean()) / df['height'].std()
    df['std_compactness'] = (df['compactness'] - df['compactness'].mean()) / df['compactness'].std()
    df['std_squareness'] = (df['symmetric_squareness_deviation'] - df['symmetric_squareness_deviation'].mean()) / df['symmetric_squareness_deviation'].std()

    # Combine the standardized features into a single array with [width, height, compactness, squareness]
    features = df[['std_width', 'std_height', 'std_compactness', 'std_squareness']].values

    # Apply KMeans clustering with a fixed random_state for consistent results
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['y_box_triplet_label'] = kmeans.fit_predict(features)

    # Calculate average and standard deviation for standardized area and squareness for each class
    average_std_metrics = df.groupby('y_box_triplet_label')[['std_compactness', 'std_squareness']].agg(['mean', 'std']).reset_index()
    print(average_std_metrics)

    # Return the triplet box label column and the standardized features
    return df['y_box_triplet_label'].values, features

def generate_triplet_box_label_normalized(y_box, y_cls, n_clusters=6):
    # Convert y_box to width and height
    width = y_box[:, 1] - y_box[:, 0]
    height = y_box[:, 3] - y_box[:, 2]
    area = width * height

    # Calculate perimeter
    perimeter = 2 * (width + height)

    # Calculate compactness
    compactness = area / (perimeter ** 2)

    # Calculate symmetric squareness deviation
    symmetric_squareness_deviation = 1 - np.minimum(width / height, height / width)

    # Create a DataFrame with the metrics
    df = pd.DataFrame({
        "class": y_cls,
        "width": width,
        "height": height,
        "area": area,
        "compactness": compactness,
        "symmetric_squareness_deviation": symmetric_squareness_deviation
    })

    # Independent Min-Max Normalization for each feature
   
    df['norm_area'] = (df['area'] - df['area'].min()) / (df['area'].max() - df['area'].min())
    df['norm_width'] = (df['width'] - df['width'].min()) / (df['width'].max() - df['width'].min())
    df['norm_height'] = (df['height'] - df['height'].min()) / (df['height'].max() - df['height'].min())
    df['norm_compactness'] = (df['compactness'] - df['compactness'].min()) / (df['compactness'].max() - df['compactness'].min())
    df['norm_squareness'] = (df['symmetric_squareness_deviation'] - df['symmetric_squareness_deviation'].min()) / (df['symmetric_squareness_deviation'].max() - df['symmetric_squareness_deviation'].min())

    # Combine the normalized features into a single array with [width, height, compactness, squareness]
    features = df[['norm_width', 'norm_height', 'norm_area', 'norm_squareness']].values

    # Apply KMeans clustering with a fixed random_state for consistent results
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['y_box_triplet_label'] = kmeans.fit_predict(features)

    # Calculate average and standard deviation for normalized area and squareness for each class
    average_std_metrics = df.groupby('y_box_triplet_label')[['norm_area', 'norm_squareness']].agg(['mean', 'std']).reset_index()
    print(average_std_metrics)

    # Return the triplet box label column and the features
    return df['y_box_triplet_label'].values, features

def generate_triplet_box_label_one_hot(y_box, y_cls, n_clusters=6):
    

    # Convert y_box to width and height
    width = y_box[:, 1] - y_box[:, 0]
    height = y_box[:, 3] - y_box[:, 2]
    area = width * height

    # Calculate perimeter
    perimeter = 2 * (width + height)

    # Calculate compactness
    compactness = area / (perimeter ** 2)

    # Calculate symmetric squareness deviation
    symmetric_squareness_deviation = 1 - np.minimum(width / height, height / width)

    # Create a DataFrame with the metrics
    df = pd.DataFrame({
        "class": y_cls,
        "width": width,
        "height": height,
        "area": area,
        "compactness": compactness,
        "symmetric_squareness_deviation": symmetric_squareness_deviation
    })

    # Normalize features
    df['norm_area'] = (df['area'] - df['area'].min()) / (df['area'].max() - df['area'].min())
    df['norm_width'] = (df['width'] - df['width'].min()) / (df['width'].max() - df['width'].min())
    df['norm_height'] = (df['height'] - df['height'].min()) / (df['height'].max() - df['height'].min())
    df['norm_compactness'] = (df['compactness'] - df['compactness'].min()) / (df['compactness'].max() - df['compactness'].min())
    df['norm_squareness'] = (df['symmetric_squareness_deviation'] - df['symmetric_squareness_deviation'].min()) / (df['symmetric_squareness_deviation'].max() - df['symmetric_squareness_deviation'].min())

    # Combine normalized features into an array
    features = df[['norm_width', 'norm_height', 'norm_area', 'norm_squareness']].values

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['y_box_triplet_label'] = kmeans.fit_predict(features)

    # One-hot encode the class labels
    encoder = OneHotEncoder(sparse=False, dtype=np.float32)
    one_hot_encoded_classes = encoder.fit_transform(df[['class']])

    # Append the one-hot encoded classes to the features
    combined_features = np.hstack((features, one_hot_encoded_classes))

    # Calculate average and standard deviation for normalized area and squareness for each class
    average_std_metrics = df.groupby('y_box_triplet_label')[['norm_area', 'norm_squareness']].agg(['mean', 'std']).reset_index()
    print(average_std_metrics)

    # Return the triplet box label column and the combined features
    return df['y_box_triplet_label'].values, combined_features

def global_max_normalize(data):
    """
    Perform Max Normalization using the global maximum value of the input data.

    Parameters:
    - data (numpy.ndarray): Input array of shape (n_samples, height, width, channels)

    Returns:
    - normalized_data (numpy.ndarray): Max normalized array with the same shape as input

    """
    # Find the global maximum value across the entire dataset
    global_max = np.max(data)
    print(global_max)

    # Perform Max normalization by dividing by the global maximum
    normalized_data = data / global_max

    # Clip values to ensure they stay in the range [0, 1]
    # normalized_data = np.clip(normalized_data, 0, 1)

    return normalized_data

def assign_tile_labels(function_y_box_cenwh):
    """
    Assign discrete labels to bounding boxes based on their (x_center, y_center) positions 
    in a 3x3 grid.

    Parameters:
        y_box_cenwh (array): Array of bounding boxes with shape (num_boxes, 4), where each row 
                             contains [x_center, y_center, width, height] (normalized by image size).

    Returns:
        array: An array of discrete tile labels for each bounding box.
    """
    # Extract x_center and y_center from y_box_cenwh
    x_center = function_y_box_cenwh[:, 0]
    y_center = function_y_box_cenwh[:, 1]
    
    # Compute tile indices for the x and y positions
    x_tile = np.floor(x_center * 3).astype(int)  # 0, 1, or 2
    y_tile = np.floor(y_center * 3).astype(int)  # 0, 1, or 2

    # Clip values to ensure they are within the range [0, 2] (in case of edge cases like 1.0)
    x_tile = np.clip(x_tile, 0, 2)
    y_tile = np.clip(y_tile, 0, 2)
    
    # Compute the tile label as (row_index * 3 + column_index)
    tile_labels = y_tile * 3 + x_tile

    return tile_labels

def calculate_mask_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def adjust_and_calculate_iou_mask(rgb_box_mask_pred, y_box_test, y_cls_test, threshold=0.5):
    num_samples = len(y_cls_test)
    iou_by_class = {'cow': [], 'deer': [], 'horse': []}
    
    for i in range(num_samples):
        adjusted_pred_mask = (rgb_box_mask_pred[i, :, :, 0] > threshold).astype(int)
        mask_shape = (300, 300)
        
        # Extract and scale the true box coordinates
        x_center, y_center, width, height = y_box_test[i]
        x_center, y_center, width, height = x_center * 300, y_center * 300, width * 300, height * 300
        xmin_true = int(x_center - width / 2)
        xmax_true = int(x_center + width / 2)
        ymin_true = int(y_center - height / 2)
        ymax_true = int(y_center + height / 2)
        
        # Create true mask with the calculated coordinates
        true_mask = np.zeros(mask_shape)
        true_mask[ymin_true:ymax_true, xmin_true:xmax_true] = 1
        
        # Calculate IoU
        iou = calculate_mask_iou(adjusted_pred_mask, true_mask)
        class_label = y_cls_test[i]
        
        # Append IoU value to the class-specific list
        if class_label in iou_by_class:
            iou_by_class[class_label].append(iou)
    
    return iou_by_class

def print_and_log_average_iou(iou_by_class, experiment, step, iou_type='mask', train_val='train'):
    total_iou = 0
    total_count = 0
    
    #print(f"Average {iou_type} IoU by class for {train_val} set:")
    for class_label, iou_list in iou_by_class.items():
        if iou_list:  # Check if there are IoU values for this class
            avg_iou = sum(iou_list) / len(iou_list)
            #print(f"{class_label}: {avg_iou:.4f}")
            metric_name = f"{train_val}_{class_label}_{iou_type}_iou"
            experiment.log_metric(metric_name, avg_iou, step=step)  # Log IoU with specified naming
            total_iou += sum(iou_list)
            total_count += len(iou_list)
        else:
            print(f"{class_label}: No samples available")
            metric_name = f"{train_val}_{class_label}_{iou_type}_iou"
            experiment.log_metric(metric_name, None)  # Log None if no samples available
    
    # Calculate and print the overall average IoU
    if total_count > 0:
        overall_avg_iou = total_iou / total_count
        #print(f"\nOverall average {iou_type} IoU for {train_val} set: {overall_avg_iou:.4f}")
        experiment.log_metric(f"{train_val}_overall_{iou_type}_iou", overall_avg_iou, step=step)  # Log overall IoU
    else:
        print(f"\nNo IoU values available for calculation in {train_val} set.")
        experiment.log_metric(f"{train_val}_overall_{iou_type}_iou", None)  # Log None if no IoU values
        
    return overall_avg_iou
        
def log_one_image_per_class(experiment, log_rgb_recon, log_y_cls, step, train_val='train'):
    """
    Logs one consistent reconstructed and ground truth image per class.

    Parameters:
        experiment: The logging experiment object (e.g., Comet.ml or similar).
        rgb_train_recon (array): Reconstructed images, shape (num_samples, 300, 300, 3).
        X_rgb_train (array): Ground truth images, shape (num_samples, 300, 300, 3).
        y_cls_train (list/array): Class labels corresponding to the images.
        step (int): The current step for logging images.
        train_val (str): Indicates whether it's the 'train' or 'val' set (default is 'train').
    """
    logged_classes = {}  # To keep track of logged images by class
    count = 0
    
    # Log one image per class
    for idx, class_label in enumerate(log_y_cls):
        
        if class_label not in logged_classes:
            count += 1
            # Log reconstructed image
  
            reconstructed_image = log_rgb_recon[idx]
            experiment.log_image(
                reconstructed_image,
                name=f"{train_val}_class_{class_label}_reconstructed_image",
                step=step
            )
            
            # Mark this class as logged
            logged_classes[class_label] = idx
            
        if count == 5:
            break
            
def log_one_hsi_recon_per_class(experiment, log_hsi_recon, log_y_cls, step, train_val='train'):
    """
    Logs one consistent reconstructed pseudo-RGB image per class.

    Parameters:
        experiment: The logging experiment object (e.g., Comet.ml or similar).
        log_hsi_recon (array): Reconstructed HSI images, shape (num_samples, height, width, bands).
        log_y_cls (list/array): Class labels corresponding to the images.
        step (int): The current step for logging images.
        train_val (str): Indicates whether it's the 'train' or 'val' set (default is 'train').
    """
    logged_classes = {}  # To keep track of logged images by class
    selected_bands = [10, 50, 60]  # Bands for pseudo-RGB

    # Log one reconstructed pseudo-RGB image per class
    for idx, class_label in enumerate(log_y_cls):
        if class_label not in logged_classes:
            # Extract selected bands and stack them into a pseudo-RGB image
            pseudo_rgb_recon = np.stack([
                log_hsi_recon[idx, :, :, selected_bands[0]],  # Red channel
                log_hsi_recon[idx, :, :, selected_bands[1]],  # Green channel
                log_hsi_recon[idx, :, :, selected_bands[2]]   # Blue channel
            ], axis=-1)  # Ensure shape is (height, width, 3)

            experiment.log_image(
                pseudo_rgb_recon,
                name=f"{train_val}_class_{class_label}_reconstructed_hsi",
                step=step
            )

            # Mark this class as logged
            logged_classes[class_label] = idx
            
def log_gt_image_per_class(experiment, X_rgb_train, y_cls_train, step, train_val='train'):
    """
    Logs one consistent reconstructed and ground truth image per class.

    Parameters:
        experiment: The logging experiment object (e.g., Comet.ml or similar).
        rgb_train_recon (array): Reconstructed images, shape (num_samples, 300, 300, 3).
        X_rgb_train (array): Ground truth images, shape (num_samples, 300, 300, 3).
        y_cls_train (list/array): Class labels corresponding to the images.
        step (int): The current step for logging images.
        train_val (str): Indicates whether it's the 'train' or 'val' set (default is 'train').
    """
    logged_classes = {}  # To keep track of logged images by class

    # Log one image per class
    count = 0 
    for idx, class_label in enumerate(y_cls_train):
        if class_label not in logged_classes:
            count += 1
             # Log ground truth image
            ground_truth_image = X_rgb_train[idx]
            experiment.log_image(
                ground_truth_image,
                name=f"{train_val}_class_{class_label}_ground_truth_image",
                step=step
            )

            # Mark this class as logged
            logged_classes[class_label] = idx
            
        if count == 5:
            break
            
def log_gt_hsi_per_class(experiment, X_hsi_train, y_cls_train, step, train_val='train'):
    """
    Logs one consistent pseudo-RGB ground truth HSI image per class.

    Parameters:
        experiment: The logging experiment object (e.g., Comet.ml or similar).
        X_hsi_train (array): Ground truth hyperspectral images, shape (num_samples, height, width, bands).
        y_cls_train (list/array): Class labels corresponding to the images.
        step (int): The current step for logging images.
        train_val (str): Indicates whether it's the 'train' or 'val' set (default is 'train').
    """
    logged_classes = {}  # To keep track of logged images by class
    selected_bands = [10, 50, 60]  # Bands for pseudo-RGB
    print(X_hsi_train.shape)

    # Log one image per class
    for idx, class_label in enumerate(y_cls_train):
        if class_label not in logged_classes:
       
            # Extract selected bands and stack them into a pseudo-RGB image
            pseudo_rgb_image = np.stack([
                X_hsi_train[idx, :, :, selected_bands[0]],  # Red channel
                X_hsi_train[idx, :, :, selected_bands[1]],  # Green channel
                X_hsi_train[idx, :, :, selected_bands[2]]   # Blue channel
            ], axis=-1)  # Ensure the shape is (height, width, 3)

            experiment.log_image(
                pseudo_rgb_image,
                name=f"{train_val}_class_{class_label}_hsi_ground_truth",
                step=step
            )

            # Mark this class as logged
            logged_classes[class_label] = idx

# Function to compute all possible distances within y_box_cenwh_train
def calculate_all_distances(y_box_cenwh_train, metric='cosine'):
    num_samples = y_box_cenwh_train.shape[0]
    
    # Compute all pairwise distances
    distances = cdist(y_box_cenwh_train, y_box_cenwh_train, metric=metric)
    
    # Extract unique pairs to avoid redundant calculations and self-distances
    unique_distances = distances[np.triu_indices(num_samples, k=1)]
    
    return unique_distances


# Function to compute all possible similarities within y_box_cenwh_train
def calculate_all_similarities(y_box_cenwh_train, metric='cosine'):
    num_samples = y_box_cenwh_train.shape[0]
    
    # Compute all pairwise distances
    distances = cdist(y_box_cenwh_train, y_box_cenwh_train, metric=metric)
    
    # Convert distances to similarities
    if metric == 'cosine':
        # Cosine similarity: similarity = 1 - cosine distance
        similarities = 1 - distances
    elif metric == 'euclidean':
        # Euclidean similarity: similarity = 1 / (1 + euclidean distance)
        similarities = 1 / (1 + distances)
    else:
        raise ValueError("Unsupported metric. Use 'cosine' or 'euclidean'.")
    
    # Extract unique pairs to avoid redundant calculations and self-similarities
    unique_similarities = similarities[np.triu_indices(num_samples, k=1)]
    
    return unique_similarities


def resampling(y, *arrays):
    """
    Resample the input arrays to balance the frequency of unique labels in 'y' by oversampling.

    Parameters:
        y (array-like): The label array used to determine balancing.
        *arrays (array-like): The arrays to resample according to balanced indices.

    Returns:
        tuple: A tuple of resampled arrays in the same order as provided in 'arrays' and the balanced 'y'.
    """
    # Find unique labels and their frequencies
    uniq, freq = np.unique(y, return_counts=True)
    uniq = uniq.tolist()
    freq = freq.tolist()

    # Determine the maximum frequency for balancing
    F = max(freq)
    indx_balanced = []

    # Balance the indices by oversampling
    for u in uniq:
        indx, = np.where(y == u)
        indx = indx.tolist()

        if len(indx) < F:
            indx_balanced_u = indx
            while len(indx_balanced_u) < F:
                indx_balanced_u += indx
            indx = indx_balanced_u[:F]
        
        indx_balanced += indx

    # Apply the balanced indices to all input arrays
    resampled_arrays = [np.array(arr)[indx_balanced] for arr in arrays]
    y_balanced = np.array(y)[indx_balanced]

    return (y_balanced, *resampled_arrays)