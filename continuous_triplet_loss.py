"""
Disclaimer:
This file is coped as-is from https://github.com/omoindrot/tensorflow-triplet-loss/blob/fc698369bb6c9acdc9f0e9e1ea00de0ddf782f12/model/triplet_loss.py
"""
# Import libraries
import os, sys
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import Ones
from tensorflow.keras.layers import Dot, Layer, Input, UpSampling3D, ReLU,GlobalAveragePooling3D, Conv3D, AveragePooling3D, GlobalMaxPooling3D, GlobalMaxPooling2D, MaxPooling2D, GlobalAveragePooling1D, AveragePooling1D, Dense, UpSampling1D, Conv2D, BatchNormalization, GlobalMaxPool2D, Multiply, GaussianNoise, UpSampling2D, GlobalAveragePooling2D, AveragePooling2D, ReLU, Reshape, Dropout, Embedding, Add, concatenate, dot, GlobalMaxPool1D, Masking, Activation, MaxPool1D, Conv1D, Flatten, TimeDistributed, Lambda, Conv2DTranspose, Cropping2D, SeparableConv2D
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers.experimental import preprocessing


"""Define functions to create the triplet loss with online triplet mining."""

import tensorflow as tf
from fractions import Fraction

def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def _pairwise_distances_cosine(embeddings, squared=False):
    """Compute the 2D matrix of cosine distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared cosine distance matrix.
                 If false, output is the pairwise cosine distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Normalize the embeddings to unit length to get cosine similarity
    embeddings = tf.nn.l2_normalize(embeddings, axis=1)

    # Compute the cosine similarity matrix
    cosine_similarity = tf.matmul(embeddings, tf.transpose(embeddings))

    # Convert cosine similarity to cosine distance
    pairwise_distances = 1.0 - cosine_similarity

    # If squared is True, return the squared cosine distance
    if squared:
        pairwise_distances = tf.square(pairwise_distances)

    # Ensure distances are non-negative due to possible small numerical errors
    pairwise_distances = tf.maximum(pairwise_distances, 0.0)

    return pairwise_distances


def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # Combine the two masks
    mask = tf.logical_and(indices_not_equal, labels_equal)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    mask = tf.logical_not(labels_equal)

    return mask


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # Combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask

def batch_all_triplet_continuous_loss_class_margin(label_vector, embeddings, margin=0.01, class_margin=0.2, squared=False):
    label_vector = tf.cast(label_vector, tf.float32)
    embeddings = tf.cast(embeddings, tf.float32)

    # 1. Split label_vector into class label and continuous box features
    class_labels = tf.cast(label_vector[:, 0], tf.int32)              # shape: [batch_size]
    box_features = label_vector[:, 1:]                                # shape: [batch_size, feature_dim]

    # 2. Compute pairwise cosine distance on continuous box features
    cosine_dist_matrix = _pairwise_distances_cosine(box_features, squared=squared)

    # 3. Add class margin if classes are different
    labels_equal = tf.equal(tf.expand_dims(class_labels, 1), tf.expand_dims(class_labels, 0))  # [B, B]
    class_margin_matrix = tf.cast(~labels_equal, tf.float32) * class_margin
    cosine_dist_matrix += class_margin_matrix

    # 4. Compute pairwise cosine distance on embeddings
    emb_dist_matrix = _pairwise_distances_cosine(embeddings, squared=squared)

    # 5. Expand distances for triplet construction
    anchor_positive_cosine_dist = tf.expand_dims(cosine_dist_matrix, 2)
    anchor_negative_cosine_dist = tf.expand_dims(cosine_dist_matrix, 1)
    anchor_positive_emb_dist = tf.expand_dims(emb_dist_matrix, 2)
    anchor_negative_emb_dist = tf.expand_dims(emb_dist_matrix, 1)

    # 6. Generalized triplet loss
    generalized_triplet_loss = (
        tf.square(anchor_positive_emb_dist)
        - tf.square(anchor_negative_emb_dist)
        + (anchor_negative_cosine_dist - anchor_positive_cosine_dist)
    )
    generalized_triplet_loss = tf.maximum(generalized_triplet_loss, 0.0)

    # 7. Regularization: encourage emb distances to match label distances
    pos_reg_term = tf.square(anchor_positive_emb_dist - anchor_positive_cosine_dist)
    neg_reg_term = tf.square(anchor_negative_emb_dist - anchor_negative_cosine_dist)

    triplet_loss = generalized_triplet_loss + pos_reg_term + neg_reg_term

    # 8. Mask out invalid triplets
    cosine_mask = tf.less(anchor_positive_cosine_dist, anchor_negative_cosine_dist)
    cosine_mask = tf.cast(cosine_mask, tf.float32)
    
    triplet_loss = tf.multiply(tf.cast(cosine_mask, tf.float32), triplet_loss)

    # 9. Final reduction
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    fraction_positive_triplets = num_positive_triplets / (tf.reduce_sum(cosine_mask) + 1e-16)

    return triplet_loss, fraction_positive_triplets

def batch_all_continuous_triplet_loss_v1(box_features, embeddings, margin=0.01, squared=False):
    box_features = tf.cast(box_features, tf.float32)
    embeddings = tf.cast(embeddings, tf.float32)

    cosine_dist_matrix = _pairwise_distances_cosine(box_features, squared=squared)
    emb_dist_matrix = _pairwise_distances_cosine(embeddings, squared=squared)
    
    anchor_positive_cosine_dist = tf.expand_dims(cosine_dist_matrix, 2)
    anchor_positive_emb_dist = tf.expand_dims(emb_dist_matrix, 2)
    anchor_negative_cosine_dist = tf.expand_dims(cosine_dist_matrix, 1)
    anchor_negative_emb_dist = tf.expand_dims(emb_dist_matrix, 1)
    
    pos_neg_distance = anchor_negative_cosine_dist - anchor_positive_cosine_dist
    # print('pos_neg_distance shape: ', pos_neg_distance.shape)
    
    separable_weight = tf.where(pos_neg_distance < margin, tf.zeros_like(pos_neg_distance), pos_neg_distance)
    # print('separable_weight shape 0: ', separable_weight.shape)
    
    separable_weight = tf.where(separable_weight > 0, tf.ones_like(separable_weight), separable_weight)
    # print('separable_weight shape 1: ', separable_weight.shape)

    # # Count number of positive triplets (where triplet_loss > 0)
    # valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    # num_positive_triplets = tf.reduce_sum(valid_triplets)
    # num_valid_triplets = tf.reduce_sum(cosine_mask)
    # fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)


    generalized_triplet_loss = tf.square(anchor_positive_emb_dist) - tf.square(anchor_negative_emb_dist) + (1-separable_weight)*(anchor_positive_cosine_dist+margin) + separable_weight*(anchor_negative_cosine_dist - anchor_positive_cosine_dist)
    
    # print('generalized_triplet_loss shape: ', generalized_triplet_loss.shape)
    
    generalized_triplet_loss = tf.maximum(generalized_triplet_loss, 0.0)
    
    pos_reg_term = tf.square(anchor_positive_emb_dist-anchor_positive_cosine_dist)
    
    # print('pos_reg_term shape: ', pos_reg_term.shape)
    
    neg_reg_term = tf.square(anchor_negative_emb_dist-anchor_negative_cosine_dist)
    
    # print('neg_reg_term shape: ', neg_reg_term.shape)
    
    triplet_loss = generalized_triplet_loss + pos_reg_term + separable_weight*neg_reg_term
    
    # triplet_loss = w_positive*anchor_positive_emb_dist - w_negative*anchor_negative_emb_dist + margin
    # print('triplet_loss shape', triplet_loss.shape)
    
    # Create mask to enforce cosine distance condition
    cosine_mask = tf.less(anchor_positive_cosine_dist, anchor_negative_cosine_dist)
    cosine_mask = tf.cast(cosine_mask, tf.float32)
    # print('cosine_mask shape', cosine_mask.shape)
    
    # Apply the mask to triplet loss
    triplet_loss = tf.multiply(cosine_mask, triplet_loss)
    # generalized_triplet_loss = tf.multiply(cosine_mask, generalized_triplet_loss)
    # print('triplet loss after mask shape', triplet_loss.shape)
    
    # Remove negative losses (i.e., the easy triplets)
    # triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(cosine_mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    # print('triplet loss final shape', triplet_loss.shape)
    return triplet_loss, fraction_positive_triplets

def batch_all_continuous_triplet_loss_final_multimodal(annotatation_vector, embeddings, margin=0.01, squared=False, EMB_DIM=1024):
    
    embeddings1, embeddings2, embeddings3 = embeddings[:, :EMB_DIM], embeddings[:, EMB_DIM:2*EMB_DIM], embeddings[:, 2*EMB_DIM:3*EMB_DIM]
    embeddings = concatenate([embeddings1, embeddings2, embeddings3], axis=0)
    
    annotatation_vector = concatenate([annotatation_vector, annotatation_vector, annotatation_vector], axis=0)
    
    annotatation_vector = tf.cast(annotatation_vector, tf.float32)
    embeddings = tf.cast(embeddings, tf.float32)

    cosine_dist_matrix = _pairwise_distances_cosine(annotatation_vector, squared=squared)
    emb_dist_matrix = _pairwise_distances_cosine(embeddings, squared=squared)
    
    anchor_positive_cosine_dist = tf.expand_dims(cosine_dist_matrix, 2)
    anchor_positive_emb_dist = tf.expand_dims(emb_dist_matrix, 2)
    anchor_negative_cosine_dist = tf.expand_dims(cosine_dist_matrix, 1)
    anchor_negative_emb_dist = tf.expand_dims(emb_dist_matrix, 1)
    
    pos_neg_distance = anchor_negative_cosine_dist - anchor_positive_cosine_dist
    # print('pos_neg_distance shape: ', pos_neg_distance.shape)
    
#     separable_weight = tf.where(pos_neg_distance < margin, tf.zeros_like(pos_neg_distance), pos_neg_distance)
#     # print('separable_weight shape 0: ', separable_weight.shape)
    
#     separable_weight = tf.where(separable_weight > 0, tf.ones_like(separable_weight), separable_weight)
    # print('separable_weight shape 1: ', separable_weight.shape)

    # # Count number of positive triplets (where triplet_loss > 0)
    # valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    # num_positive_triplets = tf.reduce_sum(valid_triplets)
    # num_valid_triplets = tf.reduce_sum(cosine_mask)
    # fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)


    generalized_triplet_loss = tf.square(anchor_positive_emb_dist) - tf.square(anchor_negative_emb_dist) +(anchor_negative_cosine_dist - anchor_positive_cosine_dist)
    
    # print('generalized_triplet_loss shape: ', generalized_triplet_loss.shape)
    
    generalized_triplet_loss = tf.maximum(generalized_triplet_loss, 0.0)
    
    pos_reg_term = tf.square(anchor_positive_emb_dist-anchor_positive_cosine_dist)
    
    # print('pos_reg_term shape: ', pos_reg_term.shape)
    
    neg_reg_term = tf.square(anchor_negative_emb_dist-anchor_negative_cosine_dist)
    
    # print('neg_reg_term shape: ', neg_reg_term.shape)
    
    triplet_loss = generalized_triplet_loss + pos_reg_term + neg_reg_term
    
    # triplet_loss = w_positive*anchor_positive_emb_dist - w_negative*anchor_negative_emb_dist + margin
    # print('triplet_loss shape', triplet_loss.shape)
    
    # Create mask to enforce cosine distance condition
    cosine_mask = tf.less(anchor_positive_cosine_dist, anchor_negative_cosine_dist)
    cosine_mask = tf.cast(cosine_mask, tf.float32)
    # print('cosine_mask shape', cosine_mask.shape)
    
    # Apply the mask to triplet loss
    triplet_loss = tf.multiply(cosine_mask, triplet_loss)
    # generalized_triplet_loss = tf.multiply(cosine_mask, generalized_triplet_loss)
    # print('triplet loss after mask shape', triplet_loss.shape)
    
    # Remove negative losses (i.e., the easy triplets)
    # triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(cosine_mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    # print('triplet loss final shape', triplet_loss.shape)
    return triplet_loss, fraction_positive_triplets


def batch_all_continuous_triplet_loss_final_lambda1(box_features, embeddings, margin=0.01, squared=False):
    box_features = tf.cast(box_features, tf.float32)
    embeddings = tf.cast(embeddings, tf.float32)

    cosine_dist_matrix = _pairwise_distances_cosine(box_features, squared=squared)
    emb_dist_matrix = _pairwise_distances_cosine(embeddings, squared=squared)
    
    anchor_positive_cosine_dist = tf.expand_dims(cosine_dist_matrix, 2)
    anchor_positive_emb_dist = tf.expand_dims(emb_dist_matrix, 2)
    anchor_negative_cosine_dist = tf.expand_dims(cosine_dist_matrix, 1)
    anchor_negative_emb_dist = tf.expand_dims(emb_dist_matrix, 1)
    
    pos_neg_distance = anchor_negative_cosine_dist - anchor_positive_cosine_dist
    # print('pos_neg_distance shape: ', pos_neg_distance.shape)
    
#     separable_weight = tf.where(pos_neg_distance < margin, tf.zeros_like(pos_neg_distance), pos_neg_distance)
#     # print('separable_weight shape 0: ', separable_weight.shape)
    
#     separable_weight = tf.where(separable_weight > 0, tf.ones_like(separable_weight), separable_weight)
    # print('separable_weight shape 1: ', separable_weight.shape)

    # # Count number of positive triplets (where triplet_loss > 0)
    # valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    # num_positive_triplets = tf.reduce_sum(valid_triplets)
    # num_valid_triplets = tf.reduce_sum(cosine_mask)
    # fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)


    generalized_triplet_loss = tf.square(anchor_positive_emb_dist) - tf.square(anchor_negative_emb_dist) +(anchor_negative_cosine_dist - anchor_positive_cosine_dist)
    
    # print('generalized_triplet_loss shape: ', generalized_triplet_loss.shape)
    
    generalized_triplet_loss = tf.maximum(generalized_triplet_loss, 0.0)
    
    pos_reg_term = tf.square(anchor_positive_emb_dist-anchor_positive_cosine_dist)
    
    # print('pos_reg_term shape: ', pos_reg_term.shape)
    
    neg_reg_term = tf.square(anchor_negative_emb_dist-anchor_negative_cosine_dist)
    
    # print('neg_reg_term shape: ', neg_reg_term.shape)
    
    triplet_loss = generalized_triplet_loss + pos_reg_term + neg_reg_term
    
    # triplet_loss = w_positive*anchor_positive_emb_dist - w_negative*anchor_negative_emb_dist + margin
    # print('triplet_loss shape', triplet_loss.shape)
    
    # Create mask to enforce cosine distance condition
    cosine_mask = tf.less(anchor_positive_cosine_dist, anchor_negative_cosine_dist)
    cosine_mask = tf.cast(cosine_mask, tf.float32)
    # print('cosine_mask shape', cosine_mask.shape)
    
    # Apply the mask to triplet loss
    triplet_loss = tf.multiply(cosine_mask, triplet_loss)
    # generalized_triplet_loss = tf.multiply(cosine_mask, generalized_triplet_loss)
    # print('triplet loss after mask shape', triplet_loss.shape)
    
    # Remove negative losses (i.e., the easy triplets)
    # triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(cosine_mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    # print('triplet loss final shape', triplet_loss.shape)
    return triplet_loss, fraction_positive_triplets


# def batch_all_continuous_triplet_loss(box_features, embeddings, margin=0.0, B=1.0, squared=False):
#     box_features = tf.cast(box_features, tf.float32)
#     embeddings = tf.cast(embeddings, tf.float32)
    
    
#     # Get the pairwise cosine distance matrix
#     cosine_dist_matrix = _pairwise_distances_cosine(box_features, squared=squared)
#     #print('created cosine_dist_matrix')
#     emb_dist_matrix = _pairwise_distances(embeddings, squared=squared)
#     #print('created emb_dist_matrix')
    
    

    
#     # shape (batch_size, batch_size, 1)
#     anchor_positive_cosine_dist = tf.expand_dims(cosine_dist_matrix, 2)
#     # print('anchor_postive_cosine_dist shape ',anchor_positive_cosine_dist.shape)
    
#     anchor_positive_emb_dist = tf.expand_dims(emb_dist_matrix, 2)
#     # print('anchor_positive_emb_dist shape ',anchor_positive_emb_dist.shape)
    
    
#     # shape (batch_size, 1, batch_size)
#     anchor_negative_cosine_dist = tf.expand_dims(cosine_dist_matrix, 1)
#     # print('anchor_negative_cosine_dist shape ',anchor_negative_cosine_dist.shape)
    
#     anchor_negative_emb_dist = tf.expand_dims(emb_dist_matrix, 1)
#     # print('anchor_negative_emb_dist shape ',anchor_negative_emb_dist.shape)
    
#     w_positive = anchor_negative_cosine_dist
#     w_negative = anchor_positive_cosine_dist
    
#     # Calculate weights for positives using w_positive = e^(-B * cosine_distance)
# #     w_positive = tf.exp(B * anchor_negative_cosine_dist)
# #     #print('created w_positive')
    
# #     w_negative = tf.exp(B * anchor_positive_cosine_dist)
    
#     #print('created w_negative')
    
#     w_sum = w_positive+w_negative
    
#     w_pos_normed = w_positive/w_sum
#     w_neg_normed = w_negative/w_sum
    
#     # w_pos_new = 
    
#     # print('w_pos shape: ', w_positive.shape, 'w_neg shape: ', w_negative.shape)
    
#     # Compute triplet loss with cosine distance criterion
    
#     triplet_positive_distance = w_positive*anchor_positive_emb_dist
#     triplet_negative_distance = w_negative*anchor_negative_emb_dist
    
#     # triplet_positive_distance = w_pos_normed*anchor_positive_emb_dist
#     # triplet_negative_distance = w_neg_normed*anchor_negative_emb_dist
    
#     # print('triplet_positive_distance shape: ', triplet_positive_distance.shape)
#     # print('triplet_negative_distance shape: ', triplet_negative_distance.shape)
    
#     triplet_loss = triplet_positive_distance - triplet_negative_distance + margin
#     # triplet_loss = w_positive*anchor_positive_emb_dist - w_negative*anchor_negative_emb_dist + margin
#     # print('triplet_loss shape', triplet_loss.shape)
    
#     # Create mask to enforce cosine distance condition
#     cosine_mask = tf.less(anchor_positive_cosine_dist, anchor_negative_cosine_dist)
#     cosine_mask = tf.cast(cosine_mask, tf.float32)
#     #print('cosine_mask shape', cosine_mask.shape)
    
#     # Apply the mask to triplet loss
#     triplet_loss = tf.multiply(cosine_mask, triplet_loss)
#     # print('triplet loss after mask shape', triplet_loss.shape)
    
#     # Remove negative losses (i.e., the easy triplets)
#     triplet_loss = tf.maximum(triplet_loss, 0.0)

#     # Count number of positive triplets (where triplet_loss > 0)
#     valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
#     num_positive_triplets = tf.reduce_sum(valid_triplets)
#     num_valid_triplets = tf.reduce_sum(cosine_mask)
#     fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

#     # Get final mean triplet loss over the positive valid triplets
#     triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

#     # print('triplet loss final shape', triplet_loss.shape)
#     return triplet_loss, fraction_positive_triplets

def batch_all_continuous_triplet_loss(box_features, embeddings, margin=0.0, B=1.0, squared=False):
    box_features = tf.cast(box_features, tf.float32)
    embeddings = tf.cast(embeddings, tf.float32)
    
    
    # Get the pairwise cosine distance matrix
    cosine_dist_matrix = _pairwise_distances_cosine(box_features, squared=squared)
    #print('created cosine_dist_matrix')
    emb_dist_matrix = _pairwise_distances(embeddings, squared=squared)
    #print('created emb_dist_matrix')
    
    
#     Calculate weights for positives using w_positive = e^(-B * cosine_distance)
#     w_positive = tf.exp(-B * cosine_dist_matrix)
#     print('created w_positive')
    
#     w_negative = tf.exp(-B * (tf.constant(2.0, dtype=tf.float32)-cosine_dist_matrix))
    # print('created w_negative')
    
    # shape (batch_size, batch_size, 1)
    anchor_positive_cosine_dist = tf.expand_dims(cosine_dist_matrix, 2)
    anchor_positive_emb_dist = tf.expand_dims(emb_dist_matrix, 2)

    # shape (batch_size, 1, batch_size)
    anchor_negative_cosine_dist = tf.expand_dims(cosine_dist_matrix, 1)
    anchor_negative_emb_dist = tf.expand_dims(emb_dist_matrix, 1)
    
    

    # Compute triplet loss with cosine distance criterion
    triplet_loss = anchor_positive_emb_dist - anchor_negative_emb_dist + margin
    
    # triplet_loss = w_positive*anchor_positive_emb_dist - w_negative*anchor_negative_emb_dist + margin
    #print('triplet_loss shape', triplet_loss.shape)
    
    # Create mask to enforce cosine distance condition
    cosine_mask = tf.less(anchor_positive_cosine_dist, anchor_negative_cosine_dist)
    cosine_mask = tf.cast(cosine_mask, tf.float32)
    #print('cosine_mask shape', cosine_mask.shape)
    
    # Apply the mask to triplet loss
    triplet_loss = tf.multiply(cosine_mask, triplet_loss)
    #print('triplet loss after mask shape', triplet_loss.shape)
    
    # Remove negative losses (i.e., the easy triplets)
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(cosine_mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    #print('triplet loss final shape', triplet_loss.shape)
    return triplet_loss, fraction_positive_triplets
    
    
    
# def batch_all_triplet_loss(labels_clf, labels_AR, embeddings, margin, box_weight = 0.5, squared=False):
#     """Build the triplet loss over a batch of embeddings.

#     We generate all the valid triplets and average the loss over the positive ones.

#     Args:
#         labels: labels for classification of the batch, of size (batch_size,)
#         labels: labels for aspect ratio of different boxes of the batch, of size (batch_size,)
#         embeddings: tensor of shape (batch_size, embed_dim)
#         margin: margin for triplet loss
#         squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
#                  If false, output is the pairwise euclidean distance matrix.

#     Returns:
#         triplet_loss: scalar tensor containing the triplet loss
#     """
#     clf_triplet_loss, _ = task_triplet_loss(labels_clf, embeddings, margin)
#     box_triplet_loss, _ = task_triplet_loss(labels_AR, embeddings, margin)
    
#     triplet_loss = (1-box_weight) * clf_triplet_loss + box_weight * box_triplet_loss
    
#     return triplet_loss

# def batch_all_triplet_loss(labels, embeddings, margin, box_weight = 0.5, squared=False):
#     """Build the triplet loss over a batch of embeddings.

#     We generate all the valid triplets and average the loss over the positive ones.

#     Args:
#         labels: labels for classification of the batch, of size (batch_size,)
#         labels: labels for aspect ratio of different boxes of the batch, of size (batch_size,)
#         embeddings: tensor of shape (batch_size, embed_dim)
#         margin: margin for triplet loss
#         squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
#                  If false, output is the pairwise euclidean distance matrix.

#     Returns:
#         triplet_loss: scalar tensor containing the triplet loss
#     """
#     labels_clf = labels[0]
#     labels_AR = labels[1]
    
#     clf_triplet_loss, _ = task_triplet_loss(labels_clf, embeddings, margin)
#     box_triplet_loss, _ = task_triplet_loss(labels_AR, embeddings, margin)
    
#     triplet_loss = (1-box_weight) * clf_triplet_loss + box_weight * box_triplet_loss
    
#     return triplet_loss

