'''
This module creates a triplet network for two modalities and classifies their fused embeddings.


Date created: Feb 29, 2024
Author:  Aditya Dutt

'''

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


KERNEL_SIZE = 3

def spectral_conv_layer(conv_x, filters):
    """Applies 1×1 convolutions to extract spectral features before spatial processing."""
    conv_x = Conv2D(filters, (1, 1), kernel_initializer=tf.keras.initializers.GlorotNormal(), padding='same', use_bias=True)(conv_x)
    conv_x = Activation("relu")(conv_x)
    conv_x = BatchNormalization()(conv_x)
    return conv_x


def spectral_dense_block(block_x, filters, growth_rate, layers_in_block):
    """Dense block with spectral feature extraction."""
    for i in range(layers_in_block):
        each_layer = spectral_conv_layer(block_x, growth_rate)  # Use spectral convs instead of standard ones
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate
    return block_x, filters

def hsi_encoder(input):
    """
    Hyperspectral encoder designed for 3×3×369 input.
    - Uses 1×1 spectral convolutions before spatial convolutions.
    - Includes a dense block for enhanced spectral feature extraction.
    - Uses Global Average Pooling for compact representation.
    """

    growth_rate = 16  # Increased to better model spectral dependencies
    filters = 16

    # Initial spectral feature extraction with 1×1 convolutions
    out = spectral_conv_layer(input, filters)

    # Apply a Dense Block for richer spectral feature extraction
    out, _ = spectral_dense_block(out, filters, growth_rate, 4)

    # Global pooling to obtain a compact feature representation
    out = GlobalAveragePooling2D()(out)

    # Final dense layer for latent representation
    latent_vector = Dense(1024, kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return latent_vector

# def squeeze_excite_block(input_tensor, ratio=8):
#     """Squeeze-and-Excitation (SE) block for spectral feature reweighting."""
#     filters = input_tensor.shape[-1]
#     se = GlobalAveragePooling2D()(input_tensor)
#     se = Dense(filters // ratio, activation="relu", kernel_initializer="he_normal")(se)
#     se = Dense(filters, activation="sigmoid", kernel_initializer="he_normal")(se)
#     return tf.keras.layers.multiply([input_tensor, se])

# def hsi_encoder_3x3(input):
#     """Hyperspectral Image (HSI) Encoder optimized for spectral feature extraction."""

#     # Spectral feature extraction using 1x1 Conv (Pointwise Convolution)
#     out = Conv2D(256, (1, 1), padding="valid", kernel_initializer="he_normal", kernel_regularizer=l2(0.005))(input)
#     out = BatchNormalization()(out)
#     out = Dropout(0.1)(out)
#     out = Activation("relu")(out)
#     # out = squeeze_excite_block(out)  # Apply SE block to learn spectral importance

    
#     out = Conv2D(128, (1, 1), padding="valid", kernel_initializer="he_normal", kernel_regularizer=l2(0.005))(input)
#     out = BatchNormalization()(out)
#     out = Dropout(0.1)(out)
#     out = Activation("relu")(out)
#     # out = squeeze_excite_block(out)  # Apply SE block to learn spectral importance

#     # Another spectral transformation layer
#     out = Conv2D(64, (1, 1), padding="valid", kernel_initializer="he_normal", kernel_regularizer=l2(0.005))(out)
#     out = BatchNormalization()(out)
#     out = Dropout(0.1)(out)
#     out = Activation("relu")(out)
#     # out = squeeze_excite_block(out)  # Apply SE block to learn spectral importance
    
#     # Use 3x3 depthwise separable convolution to refine spatial and spectral features
#     out = SeparableConv2D(64, (3, 3), padding="valid", depthwise_regularizer=l2(0.005), pointwise_regularizer=l2(0.005))(out)
#     out = BatchNormalization()(out)
#     # out = Dropout(0.2)(out)
#     out = Activation("relu")(out)

#     # Global average pooling for compact feature representation
#     # out = GlobalAveragePooling2D()(out)

#     # Fully connected layer
#     out = Dense(128, kernel_initializer="he_normal", kernel_regularizer=l2(0.005))(out)
#     out = BatchNormalization()(out)
#     out = Activation("relu")(out)
    
#     # Latent space representation
#     latent_vector = Dense(1024, kernel_initializer="he_normal")(out)
#     # latent_vector = tf.nn.l2_normalize(latent_vector, axis=-1)  # Ensures unit-norm embeddings


#     return latent_vector

# def hsi_encoder_3x3(input):
#     """Improved HSI Encoder with spectral attention, L2 regularization, and dropout."""
    

#     out = SeparableConv2D(128, (3, 3), padding="same", kernel_initializer="he_normal", 
#                           depthwise_regularizer=l2(0.005), pointwise_regularizer=l2(0.005))(input)
#     out = BatchNormalization()(out)
#     out = Activation("relu")(out)
#     out = GaussianNoise(0.1)(out)
#     out = Dropout(0.2)(out)
#     out = squeeze_excite_block(out)

#     out = SeparableConv2D(64, (3, 3), padding="same", kernel_initializer="he_normal",
#                           depthwise_regularizer=l2(0.005), pointwise_regularizer=l2(0.005))(out)
#     out = BatchNormalization()(out)
#     out = Activation("relu")(out)
#     out = GaussianNoise(0.1)(out)
#     out = Dropout(0.2)(out)
#     out = squeeze_excite_block(out)

#     out = SeparableConv2D(32, (3, 3), padding="same", kernel_initializer="he_normal",
#                           depthwise_regularizer=l2(0.005), pointwise_regularizer=l2(0.005))(out)
#     out = BatchNormalization()(out)
#     out = Activation("relu")(out)
#     out = GaussianNoise(0.1)(out)
#     out = Dropout(0.2)(out)
#     out = squeeze_excite_block(out)

#     out = Flatten()(out)

#     out = Dense(128, kernel_initializer="he_normal", kernel_regularizer=l2(0.005))(out)
#     out = BatchNormalization()(out)
#     out = Activation("relu")(out)
#     out = Dropout(0.2)(out)

#     latent_vector = Dense(1024, kernel_initializer="he_normal")(out)
#     # latent_vector = Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1))(latent_vector)  # Ensures unit-norm embeddings

#     return latent_vector


def conv_layer(conv_x, filters, dilation=1):
    conv_x = Conv2D(filters, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='same', use_bias=True)(conv_x)
    conv_x = Activation("relu")(conv_x)
    conv_x = BatchNormalization()(conv_x)
    return conv_x

def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate
    return block_x, filters

def rgb_lidar_encoder(input) :
        
    growth_rate = 12
    filters= 12

    out = Conv2D(16, (KERNEL_SIZE, KERNEL_SIZE), activation = 'relu', kernel_initializer = tf.keras.initializers.GlorotNormal(), padding = 'same', use_bias = True)(input)

    out, _ = dense_block(out, filters, growth_rate, 4)

    out = GlobalAveragePooling2D()(out)        

    latent_vector = Dense(1024, kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    
    
    #latent_vector = tf.nn.l2_normalize(out, axis= -1)
    
    return latent_vector


# def hsi_decoder_3x3(input) :
    
#     out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input)
#     out = Reshape((1, 1, 128))(out)
#     out = Conv2DTranspose(128, (3, 3), strides= (1, 1), activation= 'relu', padding='valid', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
#     out = Conv2D(256, (3, 3), activation= 'relu', padding='same', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
#     out = Conv2D(369, (1, 1), activation= 'sigmoid', padding='same', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    
#     return out

def hsi_decoder_3x3(input) :
    
    out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input)
    out = Dense(3 * 3 * 128, activation='relu')(out)  # Ensure enough features for reconstruction
    out = Reshape((3, 3, 128))(out)  # Reshape to match spatial size

    # Upsampling with Conv2DTranspose
    out = Conv2DTranspose(128, (3, 3), padding='same', kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Final layer to reconstruct 369 spectral channels
    out = Conv2D(369, (1, 1), activation='sigmoid', padding='same', kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out

def hsi_decoder_4x4(input):
    # Project from flat embedding to spatial features
    out = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Dense(4 * 4 * 128, activation='relu')(out)
    out = Reshape((4, 4, 128))(out)

    # Conv block 1
    out = Conv2D(256, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Conv block 2
    out = Conv2D(256, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Conv block 3
    out = Conv2D(128, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Final output layer: project to 369 HSI bands
    out = Conv2D(369, (1, 1), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out


def hsi_decoder_6x6(input):
    # Project from flat embedding to spatial features
    out = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Dense(6 * 6 * 128, activation='relu')(out)
    out = Reshape((6, 6, 128))(out)

    # Conv block 1
    out = Conv2D(256, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Conv block 2
    out = Conv2D(256, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Conv block 3
    out = Conv2D(128, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Final output layer: project to 369 HSI bands
    out = Conv2D(369, (1, 1), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out

def hsi_decoder_8x8(input):
    # Project from flat embedding to spatial features
    out = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Dense(8 * 8 * 128, activation='relu')(out)
    out = Reshape((8, 8, 128))(out)

    # Conv block 1
    out = Conv2D(256, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Conv block 2
    out = Conv2D(256, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Conv block 3
    out = Conv2D(128, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Final output layer: project to 369 HSI bands
    out = Conv2D(369, (1, 1), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out


def hsi_decoder_7x7(input):
    # Dense to flatten latent space and reshape to 3×3×128
    out = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Dense(3 * 3 * 128, activation='relu')(out)
    out = Reshape((3, 3, 128))(out)

    # Upsample: 3×3 → 6×6
    out = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # Upsample: 6×6 → 7×7 with a second transpose conv (1 stride)
    out = Conv2DTranspose(128, kernel_size=(2, 2), strides=(1, 1), padding='valid',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # Optional convolutional refinement
    out = Conv2D(256, kernel_size=(1, 1), activation='relu', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Final spectral channel output
    out = Conv2D(369, kernel_size=(1, 1), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out

def hsi_decoder_11x11(input):
    # Project and reshape latent vector to 3×3×128
    out = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Dense(3 * 3 * 128, activation='relu')(out)
    out = Reshape((3, 3, 128))(out)

    # 3×3 → 6×6
    out = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # 6×6 → 12×12
    out = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # Crop 12×12 → 11×11
    out = Cropping2D(cropping=((0, 1), (0, 1)))(out)

    # Optional refinement
    out = Conv2D(256, kernel_size=(1, 1), activation='relu', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Final spectral output (e.g., 369 channels)
    out = Conv2D(369, kernel_size=(1, 1), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out

def lidar_decoder_3x3(input) :
    
    out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input)
    out = Reshape((1, 1, 128))(out)
    out = Conv2DTranspose(64, (3, 3), strides= (1, 1), activation= 'relu', padding='valid', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Conv2D(32, (3, 3), activation= 'relu', padding='same', kernel_regularizer=l2(0.001), kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Conv2D(16, (3, 3), activation= 'relu', padding='same', kernel_regularizer=l2(0.001), kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Conv2D(8, (3, 3), activation= 'relu', padding='same', kernel_regularizer=l2(0.001), kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Conv2D(1, (1, 1), activation= 'sigmoid', padding='same', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    return out

def lidar_decoder_4x4(input):
    # Project from flat embedding to spatial tensor
    out = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Dense(4 * 4 * 128, activation='relu')(out)
    out = Reshape((4, 4, 128))(out)

    # Conv block 1
    out = Conv2D(256, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Conv block 2
    out = Conv2D(256, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Conv block 3
    out = Conv2D(128, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Final output layer: project to 1 channel for LiDAR
    out = Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out

def lidar_decoder_6x6(input):
    # Project from flat embedding to spatial tensor
    out = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Dense(6 * 6 * 128, activation='relu')(out)
    out = Reshape((6, 6, 128))(out)

    # Conv block 1
    out = Conv2D(256, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Conv block 2
    out = Conv2D(256, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Conv block 3
    out = Conv2D(128, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Final output layer: project to 1 channel for LiDAR
    out = Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out

def lidar_decoder_8x8(input):
    # Project from flat embedding to spatial tensor
    out = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Dense(8 * 8 * 128, activation='relu')(out)
    out = Reshape((8, 8, 128))(out)

    # Conv block 1
    out = Conv2D(256, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Conv block 2
    out = Conv2D(256, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Conv block 3
    out = Conv2D(128, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)

    # Final output layer: project to 1 channel for LiDAR
    out = Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out

def lidar_decoder_7x7(input):
    """
    Decoder to reconstruct a 7x7x1 LiDAR patch from latent input.
    """
    out = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Reshape((1, 1, 128))(out)

    # Upsample: 1x1 → 3x3
    out = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='valid',
                          activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # Upsample: 3x3 → 7x7
    out = Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='valid',
                          activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # Additional conv layers to refine output
    out = Conv2D(32, (3, 3), activation='relu', padding='same',
                 kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = Conv2D(16, (3, 3), activation='relu', padding='same',
                 kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = Conv2D(8, (3, 3), activation='relu', padding='same',
                 kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # Final 1-channel output
    out = Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out

def lidar_decoder_11x11(input):
    """
    Decoder to reconstruct an 11x11x1 LiDAR patch from a latent input vector.
    """
    out = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Reshape((1, 1, 128))(out)

    # Upsample: 1x1 → 3x3
    out = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='valid',
                          activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # Upsample: 3x3 → 7x7
    out = Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='valid',
                          activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # Upsample: 7x7 → 11x11
    out = Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='valid',
                          activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # Refinement layers
    out = Conv2D(32, (3, 3), activation='relu', padding='same',
                 kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = Conv2D(16, (3, 3), activation='relu', padding='same',
                 kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = Conv2D(8, (3, 3), activation='relu', padding='same',
                 kernel_regularizer=l2(0.001), kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # Final 1-channel output
    out = Conv2D(1, (1, 1), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out


def rgb_decoder_30x30(input):
    out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input)
    out = Reshape((1, 1, 128))(out)
    out = UpSampling2D((4,4))(out)
    out = Conv2DTranspose(64, (3, 3), strides= (1, 1), activation= 'relu', padding='same', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Conv2DTranspose(64, (3, 3), strides= (1, 1), activation= 'relu', padding='same', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)

    out = UpSampling2D((3,3))(out)
    out = Conv2DTranspose(32, (3, 3), strides= (1, 1), activation= 'relu', padding='same', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Conv2DTranspose(32, (3, 3), strides= (1, 1), activation= 'relu', padding='same', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)

    out = UpSampling2D((3,3))(out)
    out = Conv2DTranspose(16, (3, 3), strides= (1, 1), activation= 'relu', padding='same', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Conv2DTranspose(16, (3, 3), strides= (1, 1), activation= 'relu', padding='same', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)

    out = Conv2D(16, (3, 3), activation= 'relu', padding='valid', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Conv2D(16, (3, 3), activation= 'relu', padding='valid', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Conv2D(8, (3, 3), activation= 'relu', padding='valid', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Conv2D(3, (3, 3), activation= 'sigmoid', padding='same', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    
    return out

def rgb_decoder_40x40(input):
    # Project flat embedding to initial spatial tensor
    out = Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Dense(5 * 5 * 128, activation='relu')(out)
    out = Reshape((5, 5, 128))(out)

    # Upsample to 10x10
    out = Conv2DTranspose(128, (3, 3), strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # Upsample to 20x20
    out = Conv2DTranspose(64, (3, 3), strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # Upsample to 40x40
    out = Conv2DTranspose(32, (3, 3), strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # Final 40x40x3 output
    out = Conv2D(3, (1, 1), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out

def rgb_decoder_60x60(input):
    # Project flat embedding to initial 6x6 feature map
    out = Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Dense(6 * 6 * 128, activation='relu')(out)
    out = Reshape((6, 6, 128))(out)

    # Upsample: 6x6 → 12x12
    out = Conv2DTranspose(128, (3, 3), strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # Upsample: 12x12 → 24x24
    out = Conv2DTranspose(64, (3, 3), strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # Upsample: 24x24 → 48x48
    out = Conv2DTranspose(32, (3, 3), strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # Final upsample: 48x48 → 60x60 using large kernel
    out = Conv2DTranspose(16, (13, 13), strides=1, padding='valid',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # Output layer: 60x60x3
    out = Conv2D(3, (1, 1), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out

def rgb_decoder_80x80(input):
    # Project flat embedding to initial 5x5 feature map
    out = Dense(256, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Dense(5 * 5 * 128, activation='relu')(out)
    out = Reshape((5, 5, 128))(out)

    # Upsample: 5x5 → 10x10
    out = Conv2DTranspose(128, (3, 3), strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # 10x10 → 20x20
    out = Conv2DTranspose(64, (3, 3), strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # 20x20 → 40x40
    out = Conv2DTranspose(32, (3, 3), strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # 40x40 → 80x80
    out = Conv2DTranspose(16, (3, 3), strides=2, padding='same',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    # Final RGB output
    out = Conv2D(3, (1, 1), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out


def rgb_decoder_70x70(input):
    # 1x1 → 5x5
    out = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Reshape((1, 1, 128))(out)
    out = UpSampling2D((5, 5))(out)  # 1x1 → 5x5
    out = Conv2D(128, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # 5x5 → 10x10
    out = UpSampling2D((2, 2))(out)
    out = Conv2D(128, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    
    out = Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='valid', activation='relu',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # 14x14 → 28x28
    out = UpSampling2D((2, 2))(out)
    out = Conv2D(64, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # 28x28 → 35x35
    out = Conv2DTranspose(32, (8, 8), strides=(1, 1), padding='valid', activation='relu',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # 35x35 → 70x70
    out = UpSampling2D((2, 2))(out)
    out = Conv2D(16, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # Final RGB output
    out = Conv2D(3, (3, 3), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out

def rgb_decoder_110x110(input):
    # 1x1 → 5x5
    out = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input)
    out = Reshape((1, 1, 128))(out)
    out = UpSampling2D((5, 5))(out)  # 1x1 → 5x5
    out = Conv2D(128, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # 5x5 → 10x10
    out = UpSampling2D((2, 2))(out)
    out = Conv2D(128, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # 10x10 → 22x22
    out = Conv2DTranspose(128, (13, 13), strides=(1, 1), padding='valid', activation='relu',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # 22x22 → 44x44
    out = UpSampling2D((2, 2))(out)
    out = Conv2D(64, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # 44x44 → 55x55
    out = Conv2DTranspose(64, (12, 12), strides=(1, 1), padding='valid', activation='relu',
                          kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # 55x55 → 110x110
    out = UpSampling2D((2, 2))(out)
    out = Conv2D(32, (3, 3), padding='same', activation='relu',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    # Final RGB output
    out = Conv2D(3, (3, 3), activation='sigmoid', padding='same',
                 kernel_initializer=tf.keras.initializers.GlorotNormal())(out)

    return out



# # Define parameters
# EMB_DIM = 32 # embeddings dimension
# MARGIN = 0.4 # margin parameter
# MAE loss function 
def mean_squared_loss(y_true, y_pred):
    # Compute element-wise squared error
    mse_loss = tf.square(y_true - y_pred)
    
    # Reduce across all spatial and channel dimensions (height, width, channels)
    mse_loss = tf.reduce_mean(mse_loss, axis=[1, 2, 3])
    
    # Reduce across batch dimension
    mse_loss = tf.reduce_mean(mse_loss) 
    
    return mse_loss



# # CBAM Spatial Attention
# def spatial_attention(input, id):
    
#     input = input[:,:,:,0]
       
#     mu = tf.reduce_mean(input, axis= -1, keepdims= True)
#     maximum = tf.reduce_max(input, axis= -1, keepdims= True)

#     out = concatenate([mu, maximum], axis= -1)
#     out = Conv2D(1, (1,1), padding= 'same', use_bias= False, name= 'attn'+str(id))(out)
#     out = out / 2 # Temperature parameter of sigmoid
#     out = Activation('sigmoid')(out)
#     out = tf.expand_dims(out, axis= -1)

#     return out

def sampling(input_tensor, EMB_DIM=1024):
    z_mean, z_log_var = input_tensor
    # batch = K.shape(z_mean)[0]
    # dim = K.int_shape(z_mean)[1]
    
    epsilon = K.random_normal(shape=(EMB_DIM, ))
    
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def encoder_weighted_softmax(input, EMB_DIM = 1024, dilation = 1) :
        
    out = Conv2D(16, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(input)
    out = BatchNormalization()(out)
    
    out = Conv2D(16, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)    
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(32, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)    
    
    out = Conv2D(64, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
    
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(128, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(256, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
#     out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
#     out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(1024, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)

    encoder_features = GlobalMaxPooling2D()(out)
    
    #out = Dense(EMB_DIM, activation= 'tanh', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Dense(EMB_DIM, kernel_initializer= tf.keras.initializers.GlorotNormal())(encoder_features)
    
    out_weights = Dense(3, activation='softmax',  kernel_initializer='zeros', use_bias=False)(encoder_features)
    # out = tf.nn.l2_normalize(out, axis= -1)
    
    encoder_out = concatenate([out, out_weights], axis=-1)
    
    return encoder_out

def encoder_weighted(input, EMB_DIM = 1024, dilation = 1) :
        
    out = Conv2D(16, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(input)
    out = BatchNormalization()(out)
    
    out = Conv2D(16, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)    
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(32, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)    
    
    out = Conv2D(64, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
    
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(128, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(256, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
#     out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
#     out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(1024, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)

    encoder_features = GlobalMaxPooling2D()(out)
    
    #out = Dense(EMB_DIM, activation= 'tanh', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Dense(EMB_DIM, kernel_initializer= tf.keras.initializers.GlorotNormal())(encoder_features)
    
    out_weights = Dense(3, activation='sigmoid',  kernel_initializer='zeros', use_bias=False)(encoder_features)
    # out = tf.nn.l2_normalize(out, axis= -1)
    
    encoder_out = concatenate([out, out_weights], axis=-1)
    
    return encoder_out

def encoder_vae(input, EMB_DIM = 1024, dilation = 1) :
        
    out = Conv2D(16, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(input)
    out = BatchNormalization()(out)
    
    out = Conv2D(16, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)    
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(32, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)    
    
    out = Conv2D(64, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
    
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(128, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(256, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
#     out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
#     out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(1024, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)

    out = GlobalMaxPooling2D()(out)
    
    #out = Dense(EMB_DIM, activation= 'tanh', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    # out = Dense(EMB_DIM, kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    # out = tf.nn.l2_normalize(out, axis= -1)

    z_mean = Dense(EMB_DIM, name='encoder_z_mean')(out)
    z_log_var = Dense(EMB_DIM, name='encoder_z_log_var')(out)
    
    # Sample z
    z = Lambda(sampling, output_shape=(EMB_DIM,), name='encoder_z')([z_mean, z_log_var])
    
    # encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    # params['filters'] = filters
    return z_mean, z_log_var, z


def encoder(input, EMB_DIM = 1024, dilation = 1) :
        
    out = Conv2D(16, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(input)
    out = BatchNormalization()(out)
    
    out = Conv2D(16, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)    
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(32, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)    
    
    out = Conv2D(64, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
    
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(128, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
    out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(256, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
#     out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(512, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)
#     out = MaxPooling2D((3,3))(out)
    
    out = Conv2D(1024, (KERNEL_SIZE, KERNEL_SIZE), dilation_rate= dilation, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal(), padding='valid', use_bias=True)(out)
    out = BatchNormalization()(out)

    out = GlobalMaxPooling2D()(out)
    
    #out = Dense(EMB_DIM, activation= 'tanh', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Dense(EMB_DIM, kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    # out = tf.nn.l2_normalize(out, axis= -1)
    return out

def mask_decoder(input):
    x = Dense(128, kernel_initializer= tf.keras.initializers.GlorotNormal())(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)    

    x = Reshape((1, 1, 128))(x)

    x = UpSampling2D((3,3))(x)
    previous_block_activation = x

    filters = [128, 128, 64, 64, 32]
    upsample_factors = [3, 3, 3, 2, 2]
    for i in range(5) :    

        # Block i
        x = Conv2DTranspose(filters[i], (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 1x1 to 2x2
        x = ReLU()(x)    
        x = Conv2DTranspose(filters[i], (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 2x2 to 6x6
        x = ReLU()(x)

        x = UpSampling2D((upsample_factors[i], upsample_factors[i]))(x)

        # Project residual
        residual = UpSampling2D(upsample_factors[i])(previous_block_activation)
        residual = Conv2D(filters[i], 1, padding="same")(residual)
        x = Add()([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside the next residual

    x = Cropping2D(((0,24), (0, 24)))(x)

    x = Conv2D(32, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    # x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(16, (2, 2), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    x = ReLU()(x)

    x = Conv2D(8, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    x = ReLU()(x)

    x = Conv2D(1, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33

    output = Activation('sigmoid')(x)
    
    model = Model(input,output, name = 'decoder')
    return model

# def mask_decoder(input):
#     x = Dense(256, kernel_initializer= tf.keras.initializers.GlorotNormal())(input)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)    

#     x = Reshape((1, 1, 256))(x)

#     x = UpSampling2D((3,3))(x)
#     previous_block_activation = x

#     filters = [128, 128, 64, 64, 32]
#     upsample_factors = [3, 3, 3, 2, 2]
#     for i in range(5) :    

#         # Block i
#         x = Conv2DTranspose(filters[i], (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 1x1 to 2x2
#         x = ReLU()(x)    
#         x = Conv2DTranspose(filters[i], (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 2x2 to 6x6
#         x = ReLU()(x)

#         x = UpSampling2D((upsample_factors[i], upsample_factors[i]))(x)

#         # Project residual
#         residual = UpSampling2D(upsample_factors[i])(previous_block_activation)
#         residual = Conv2D(filters[i], 1, padding="same")(residual)
#         x = Add()([x, residual])  # Add back residual
#         previous_block_activation = x  # Set aside the next residual

#     x = Cropping2D(((0,24), (0, 24)))(x)

#     x = Conv2D(32, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
#     # x = BatchNormalization()(x)
#     x = ReLU()(x)

#     x = Conv2D(16, (2, 2), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
#     x = ReLU()(x)

#     x = Conv2D(8, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
#     x = ReLU()(x)

#     x = Conv2D(1, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33

#     output = Activation('sigmoid')(x)
    
#     model = Model(input,output, name = 'decoder')
#     return model

def image_reconstruction_vae(S1_DIM, EMB_DIM=1024):
    '''
    Image reconstruction for no triplet
    '''
    input_encoder = Input(S1_DIM)    
    output_encoder = encoder_vae(input_encoder, EMB_DIM)
    
    model_encoder = Model(input_encoder, output_encoder, name='encoder')
    
    # print(EncoderModality1.summary())

    # latent_space = model_encoder(input_encoder)
    
    input_decoder = Input(EMB_DIM)
    
    output_decoder = image_decoder(input_decoder)
    model_decoder = Model(input_decoder, output_decoder, name='decoder')
    
    input_AE = Input(S1_DIM)
    
    z_mean, z_log_var, z = model_encoder(input_AE)
    
    concat = concatenate([z_mean, z_log_var], axis=-1)
    
    AE = Model(input_AE, [model_decoder(z), concat])
    
    print(AE.summary())
    
    return AE


def image_decoder(input):
    x = Reshape((1, 1, 1024))(input)

    x = UpSampling2D((3,3))(x)
    previous_block_activation = x

    filters = [256, 256, 128, 128, 64]
    upsample_factors = [3, 3, 3, 2, 2]
    for i in range(5) :    

        # Block i
        x = Conv2DTranspose(filters[i], (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 1x1 to 2x2
        x = ReLU()(x)    
        x = Conv2DTranspose(filters[i], (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 2x2 to 6x6
        x = ReLU()(x)

        x = UpSampling2D((upsample_factors[i], upsample_factors[i]))(x)

        # Project residual
        # residual = UpSampling2D(upsample_factors[i])(previous_block_activation)
        # residual = Conv2D(filters[i], 1, padding="same")(residual)
        # x = Add()([x, residual])  # Add back residual
        # previous_block_activation = x  # Set aside the next residual

    x = Cropping2D(((0,24), (0, 24)))(x)

    x = Conv2D(32, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    # x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(32, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    x = ReLU()(x)

    x = Conv2D(16, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    x = ReLU()(x)

    x = Conv2D(3, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33

    output = Activation('sigmoid')(x)
    
    return output

def image_decoder_new(input):
    '''
    image decoder for pixel level classification
    '''
    x = Reshape((1, 1, 1024))(input)

    x = UpSampling2D((3,3))(x)
    previous_block_activation = x

    filters = [256, 256, 128, 128, 64]
    upsample_factors = [3, 3, 3, 2, 2]
    for i in range(5) :    

        # Block i
        x = Conv2DTranspose(filters[i], (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 1x1 to 2x2
        x = ReLU()(x)    
        x = Conv2DTranspose(filters[i], (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 2x2 to 6x6
        x = ReLU()(x)

        x = UpSampling2D((upsample_factors[i], upsample_factors[i]))(x)

        # Project residual
        # residual = UpSampling2D(upsample_factors[i])(previous_block_activation)
        # residual = Conv2D(filters[i], 1, padding="same")(residual)
        # x = Add()([x, residual])  # Add back residual
        # previous_block_activation = x  # Set aside the next residual

    x = Cropping2D(((0,24), (0, 24)))(x)

    x = Conv2D(32, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    # x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(32, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    x = ReLU()(x)

    x = Conv2D(16, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    x = ReLU()(x)

    x = Conv2D(4, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33

    output = Activation('softmax')(x)
    
    return output

def image_reconstruction(S1_DIM, modality):
    '''
    Image reconstruction with modality-specific encoders and decoders.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
        modality (str): One of "rgb", "lidar", or "hsi", determining which encoder and decoder to use.
    
    Returns:
        AE (Model): Autoencoder model.
    '''
    input_encoder = Input(S1_DIM)
    
    # Select encoder based on modality
    if modality == "hsi":
        output_encoder = hsi_encoder(input_encoder)
    elif modality in ["rgb", "lidar"]:
        output_encoder = rgb_lidar_encoder(input_encoder)
    else:
        raise ValueError("Invalid modality. Choose from 'rgb', 'lidar', or 'hsi'.")

    model_encoder = Model(input_encoder, output_encoder, name=f"{modality}_encoder")

    input_decoder = Input(output_encoder.shape[1:])  # Match latent space shape
    
    # Select decoder based on modality
    if modality == "hsi":
        output_decoder = hsi_decoder_3x3(input_decoder)
    elif modality == "rgb":
        output_decoder = rgb_decoder_30x30(input_decoder)
    elif modality == "lidar":
        output_decoder = lidar_decoder_3x3(input_decoder)

    model_decoder = Model(input_decoder, output_decoder, name=f"{modality}_decoder")

    input_AE = Input(S1_DIM)
    
    AE = Model(input_AE, model_decoder(model_encoder(input_AE)), name=f"{modality}_autoencoder")
    
    print(AE.summary())
    
    return AE

def image_reconstruction_4x4(S1_DIM, modality):
    '''
    Image reconstruction with modality-specific encoders and decoders.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
        modality (str): One of "rgb", "lidar", or "hsi", determining which encoder and decoder to use.
    
    Returns:
        AE (Model): Autoencoder model.
    '''
    input_encoder = Input(S1_DIM)
    
    # Select encoder based on modality
    if modality == "hsi":
        output_encoder = hsi_encoder(input_encoder)
    elif modality in ["rgb", "lidar"]:
        output_encoder = rgb_lidar_encoder(input_encoder)
    else:
        raise ValueError("Invalid modality. Choose from 'rgb', 'lidar', or 'hsi'.")

    model_encoder = Model(input_encoder, output_encoder, name=f"{modality}_encoder")

    input_decoder = Input(output_encoder.shape[1:])  # Match latent space shape
    
    # Select decoder based on modality
    if modality == "hsi":
        output_decoder = hsi_decoder_4x4(input_decoder)
    elif modality == "rgb":
        output_decoder = rgb_decoder_40x40(input_decoder)
    elif modality == "lidar":
        output_decoder = lidar_decoder_4x4(input_decoder)

    model_decoder = Model(input_decoder, output_decoder, name=f"{modality}_decoder")

    input_AE = Input(S1_DIM)
    
    AE = Model(input_AE, model_decoder(model_encoder(input_AE)), name=f"{modality}_autoencoder")
    
    print(AE.summary())
    
    return AE

def image_reconstruction_6x6(S1_DIM, modality):
    '''
    Image reconstruction with modality-specific encoders and decoders.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
        modality (str): One of "rgb", "lidar", or "hsi", determining which encoder and decoder to use.
    
    Returns:
        AE (Model): Autoencoder model.
    '''
    input_encoder = Input(S1_DIM)
    
    # Select encoder based on modality
    if modality == "hsi":
        output_encoder = hsi_encoder(input_encoder)
    elif modality in ["rgb", "lidar"]:
        output_encoder = rgb_lidar_encoder(input_encoder)
    else:
        raise ValueError("Invalid modality. Choose from 'rgb', 'lidar', or 'hsi'.")

    model_encoder = Model(input_encoder, output_encoder, name=f"{modality}_encoder")

    input_decoder = Input(output_encoder.shape[1:])  # Match latent space shape
    
    # Select decoder based on modality
    if modality == "hsi":
        output_decoder = hsi_decoder_6x6(input_decoder)
    elif modality == "rgb":
        output_decoder = rgb_decoder_60x60(input_decoder)
    elif modality == "lidar":
        output_decoder = lidar_decoder_6x6(input_decoder)

    model_decoder = Model(input_decoder, output_decoder, name=f"{modality}_decoder")

    input_AE = Input(S1_DIM)
    
    AE = Model(input_AE, model_decoder(model_encoder(input_AE)), name=f"{modality}_autoencoder")
    
    print(AE.summary())
    
    return AE

def image_reconstruction_7x7(S1_DIM, modality):
    '''
    Image reconstruction with modality-specific encoders and decoders.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
        modality (str): One of "rgb", "lidar", or "hsi", determining which encoder and decoder to use.
    
    Returns:
        AE (Model): Autoencoder model.
    '''
    input_encoder = Input(S1_DIM)
    
    # Select encoder based on modality
    if modality == "hsi":
        output_encoder = hsi_encoder(input_encoder)
    elif modality in ["rgb", "lidar"]:
        output_encoder = rgb_lidar_encoder(input_encoder)
    else:
        raise ValueError("Invalid modality. Choose from 'rgb', 'lidar', or 'hsi'.")

    model_encoder = Model(input_encoder, output_encoder, name=f"{modality}_encoder")

    input_decoder = Input(output_encoder.shape[1:])  # Match latent space shape
    
    # Select decoder based on modality
    if modality == "hsi":
        output_decoder = hsi_decoder_7x7(input_decoder)
    elif modality == "rgb":
        output_decoder = rgb_decoder_70x70(input_decoder)
    elif modality == "lidar":
        output_decoder = lidar_decoder_7x7(input_decoder)

    model_decoder = Model(input_decoder, output_decoder, name=f"{modality}_decoder")

    input_AE = Input(S1_DIM)
    
    AE = Model(input_AE, model_decoder(model_encoder(input_AE)), name=f"{modality}_autoencoder")
    
    print(AE.summary())
    
    return AE

def image_reconstruction_11x11(S1_DIM, modality):
    '''
    Image reconstruction with modality-specific encoders and decoders.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
        modality (str): One of "rgb", "lidar", or "hsi", determining which encoder and decoder to use.
    
    Returns:
        AE (Model): Autoencoder model.
    '''
    input_encoder = Input(S1_DIM)
    
    # Select encoder based on modality
    if modality == "hsi":
        output_encoder = hsi_encoder(input_encoder)
    elif modality in ["rgb", "lidar"]:
        output_encoder = rgb_lidar_encoder(input_encoder)
    else:
        raise ValueError("Invalid modality. Choose from 'rgb', 'lidar', or 'hsi'.")

    model_encoder = Model(input_encoder, output_encoder, name=f"{modality}_encoder")

    input_decoder = Input(output_encoder.shape[1:])  # Match latent space shape
    
    # Select decoder based on modality
    if modality == "hsi":
        output_decoder = hsi_decoder_11x11(input_decoder)
    elif modality == "rgb":
        output_decoder = rgb_decoder_110x110(input_decoder)
    elif modality == "lidar":
        output_decoder = lidar_decoder_11x11(input_decoder)

    model_decoder = Model(input_decoder, output_decoder, name=f"{modality}_decoder")

    input_AE = Input(S1_DIM)
    
    AE = Model(input_AE, model_decoder(model_encoder(input_AE)), name=f"{modality}_autoencoder")
    
    print(AE.summary())
    
    return AE

def image_reconstruction_multi_modal(RGB_DIM, HSI_DIM, LIDAR_DIM, EMB_DIM=1024):
    '''
    Image reconstruction with triplet loss for multiple modalities.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
        modalities (list of str): List of modalities to include, such as ["rgb", "lidar", "hsi"].
    
    Returns:
        AE (Model): Multi-modal autoencoder model outputting both the latent space and reconstructed image.
    '''
    rgb_input_encoder = Input(RGB_DIM)
    lidar_input_encoder = Input(LIDAR_DIM)
    hsi_input_encoder = Input(HSI_DIM)
    
    # Select encoder based on modality
    
    hsi_output_encoder = hsi_encoder(hsi_input_encoder)
    
    rgb_output_encoder = rgb_lidar_encoder(rgb_input_encoder)
    
    lidar_output_encoder = rgb_lidar_encoder(lidar_input_encoder)
    
    rgb_model_encoder = Model(rgb_input_encoder, rgb_output_encoder, name="rgb_encoder")
    lidar_model_encoder = Model(lidar_input_encoder, lidar_output_encoder, name="lidar_encoder")
    hsi_model_encoder = Model(hsi_input_encoder, hsi_output_encoder, name="hsi_encoder")

    rgb_input_decoder = Input(EMB_DIM)
    lidar_input_decoder = Input(EMB_DIM)
    hsi_input_decoder = Input(EMB_DIM)
    
    # Select encoder based on modality
    
    hsi_output_decoder = hsi_decoder_3x3(hsi_input_decoder)
    
    rgb_output_decoder = rgb_decoder_30x30(rgb_input_decoder)
    
    lidar_output_decoder = lidar_decoder_3x3(lidar_input_decoder)
    
    rgb_model_decoder = Model(rgb_input_decoder, rgb_output_decoder, name="rgb_decoder")
    lidar_model_decoder = Model(lidar_input_decoder, lidar_output_decoder, name="lidar_decoder")
    hsi_model_decoder = Model(hsi_input_decoder, hsi_output_decoder, name="hsi_decoder")
    
    rgb_emb = rgb_model_encoder(rgb_input_encoder)
    lidar_emb = lidar_model_encoder(lidar_input_encoder)
    hsi_emb = hsi_model_encoder(hsi_input_encoder)
    
    concatenated_emb = concatenate([rgb_emb, lidar_emb, hsi_emb], axis=-1)
    
    # Output both latent representation and reconstructed image
    AE = Model([rgb_input_encoder, lidar_input_encoder, hsi_input_encoder], [concatenated_emb, rgb_model_decoder(rgb_emb), lidar_model_decoder(lidar_emb), hsi_model_decoder(hsi_emb)], 
               name="multimodal_autoencoder_triplet")

    print(AE.summary())
    
    return AE

def image_reconstruction_multi_modal_ultimate(RGB_DIM, HSI_DIM, LIDAR_DIM, EMB_DIM=1024):
    '''
    Image reconstruction with triplet loss for multiple modalities.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
        modalities (list of str): List of modalities to include, such as ["rgb", "lidar", "hsi"].
    
    Returns:
        AE (Model): Multi-modal autoencoder model outputting both the latent space and reconstructed image.
    '''
    rgb_input_encoder = Input(RGB_DIM)
    lidar_input_encoder = Input(LIDAR_DIM)
    hsi_input_encoder = Input(HSI_DIM)
    
    # Select encoder based on modality
    
    hsi_output_encoder = hsi_encoder(hsi_input_encoder)
    
    rgb_output_encoder = rgb_lidar_encoder(rgb_input_encoder)
    
    lidar_output_encoder = rgb_lidar_encoder(lidar_input_encoder)
    
    rgb_model_encoder = Model(rgb_input_encoder, rgb_output_encoder, name="rgb_encoder")
    lidar_model_encoder = Model(lidar_input_encoder, lidar_output_encoder, name="lidar_encoder")
    hsi_model_encoder = Model(hsi_input_encoder, hsi_output_encoder, name="hsi_encoder")

    rgb_input_decoder = Input(EMB_DIM)
    lidar_input_decoder = Input(EMB_DIM)
    hsi_input_decoder = Input(EMB_DIM)
    
    # Select encoder based on modality
    
    hsi_output_decoder = hsi_decoder_3x3(hsi_input_decoder)
    
    rgb_output_decoder = rgb_decoder_30x30(rgb_input_decoder)
    
    lidar_output_decoder = lidar_decoder_3x3(lidar_input_decoder)
    
    rgb_model_decoder = Model(rgb_input_decoder, rgb_output_decoder, name="rgb_decoder")
    lidar_model_decoder = Model(lidar_input_decoder, lidar_output_decoder, name="lidar_decoder")
    hsi_model_decoder = Model(hsi_input_decoder, hsi_output_decoder, name="hsi_decoder")
    
    rgb_emb = rgb_model_encoder(rgb_input_encoder)
    lidar_emb = lidar_model_encoder(lidar_input_encoder)
    hsi_emb = hsi_model_encoder(hsi_input_encoder)
    
    concatenated_emb = concatenate([rgb_emb, lidar_emb, hsi_emb], axis=-1)
    
    # Output both latent representation and reconstructed image
    AE = Model([rgb_input_encoder, lidar_input_encoder, hsi_input_encoder], [concatenated_emb, rgb_model_decoder(rgb_emb), rgb_model_decoder(lidar_emb), rgb_model_decoder(hsi_emb), lidar_model_decoder(lidar_emb), lidar_model_decoder(rgb_emb), lidar_model_decoder(hsi_emb), hsi_model_decoder(hsi_emb), hsi_model_decoder(rgb_emb), hsi_model_decoder(lidar_emb)], name="multimodal_autoencoder_triplet")

    print(AE.summary())
    
    return AE

def image_reconstruction_multi_modalv1(RGB_DIM, HSI_DIM, LIDAR_DIM, EMB_DIM=1024):
    '''
    Image reconstruction with triplet loss for multiple modalities v1.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
    
    Returns:
        AE (Model): Multi-modal autoencoder model outputting both the latent space and reconstructed image.
    '''
    rgb_input_encoder = Input(RGB_DIM)
    hsi_input_encoder = Input(HSI_DIM)
    lidar_input_encoder = Input(LIDAR_DIM)
    
    rgb_output_encoder = rgb_lidar_encoder(rgb_input_encoder)
    hsi_output_encoder = hsi_encoder(hsi_input_encoder)
    lidar_output_encoder = rgb_lidar_encoder(lidar_input_encoder)
    
    rgb_model_encoder = Model(rgb_input_encoder, rgb_output_encoder, name="rgb_encoder")
    hsi_model_encoder = Model(hsi_input_encoder, hsi_output_encoder, name="hsi_encoder")
    lidar_model_encoder = Model(lidar_input_encoder, lidar_output_encoder, name="lidar_encoder")
  
    rgb_input_decoder = Input(EMB_DIM)
    hsi_input_decoder = Input(EMB_DIM)
    lidar_input_decoder = Input(EMB_DIM)

    
    # Select encoder based on modality
    
    hsi_output_decoder = hsi_decoder_6x6(hsi_input_decoder)
    
    rgb_output_decoder = rgb_decoder_60x60(rgb_input_decoder)
    
    lidar_output_decoder = lidar_decoder_6x6(lidar_input_decoder)
    
    rgb_model_decoder = Model(rgb_input_decoder, rgb_output_decoder, name="rgb_decoder")
    lidar_model_decoder = Model(lidar_input_decoder, lidar_output_decoder, name="lidar_decoder")
    hsi_model_decoder = Model(hsi_input_decoder, hsi_output_decoder, name="hsi_decoder")
    
    rgb_emb = rgb_model_encoder(rgb_input_encoder)
    lidar_emb = lidar_model_encoder(lidar_input_encoder)
    hsi_emb = hsi_model_encoder(hsi_input_encoder)
    
    concatenated_emb = concatenate([rgb_emb, hsi_emb, lidar_emb], axis=-1)
    
    # Output both latent representation and reconstructed image
    AE = Model([rgb_input_encoder, hsi_input_encoder, lidar_input_encoder], [concatenated_emb, rgb_model_decoder(rgb_emb), hsi_model_decoder(hsi_emb), lidar_model_decoder(lidar_emb)], 
               name="multimodal_autoencoder_triplet")

    print(AE.summary())
    
    return AE


def image_reconstruction_multi_modalv2(RGB_DIM, HSI_DIM, LIDAR_DIM, EMB_DIM=1024):
    '''
    Image reconstruction with triplet loss for multiple modalities v3.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
    
    Returns:
        AE (Model): Multi-modal autoencoder model outputting both the latent space and reconstructed image.
    '''
    rgb_input_encoder = Input(RGB_DIM)
    hsi_input_encoder = Input(HSI_DIM)
    lidar_input_encoder = Input(LIDAR_DIM)
    
    rgb_output_encoder = rgb_lidar_encoder(rgb_input_encoder)
    hsi_output_encoder = hsi_encoder(hsi_input_encoder)
    lidar_output_encoder = rgb_lidar_encoder(lidar_input_encoder)
    
    rgb_model_encoder = Model(rgb_input_encoder, rgb_output_encoder, name="rgb_encoder")
    hsi_model_encoder = Model(hsi_input_encoder, hsi_output_encoder, name="hsi_encoder")
    lidar_model_encoder = Model(lidar_input_encoder, lidar_output_encoder, name="lidar_encoder")
  
    rgb_input_decoder = Input(EMB_DIM*3)
    hsi_input_decoder = Input(EMB_DIM*3)
    lidar_input_decoder = Input(EMB_DIM*3)

    
    # Select encoder based on modality
    
    hsi_output_decoder = hsi_decoder_6x6(hsi_input_decoder)
    
    rgb_output_decoder = rgb_decoder_60x60(rgb_input_decoder)
    
    lidar_output_decoder = lidar_decoder_6x6(lidar_input_decoder)
    
    rgb_model_decoder = Model(rgb_input_decoder, rgb_output_decoder, name="rgb_decoder")
    lidar_model_decoder = Model(lidar_input_decoder, lidar_output_decoder, name="lidar_decoder")
    hsi_model_decoder = Model(hsi_input_decoder, hsi_output_decoder, name="hsi_decoder")
    
    rgb_emb = rgb_model_encoder(rgb_input_encoder)
    lidar_emb = lidar_model_encoder(lidar_input_encoder)
    hsi_emb = hsi_model_encoder(hsi_input_encoder)
    
    concatenated_emb = concatenate([rgb_emb, hsi_emb, lidar_emb], axis=-1)
    
    # Output both latent representation and reconstructed image
    AE = Model([rgb_input_encoder, hsi_input_encoder, lidar_input_encoder], [concatenated_emb, rgb_model_decoder(concatenated_emb), hsi_model_decoder(concatenated_emb), lidar_model_decoder(concatenated_emb)], 
               name="multimodal_autoencoder_triplet")

    print(AE.summary())
    
    return AE


def image_reconstruction_multi_modalv3(RGB_DIM, THERMAL_DIM, IND_EMB_DIM=1024, COMB_EMB_DIM=2048):
    '''
    Multi-modal image reconstruction autoencoder with combined and individual modality reconstructions.

    Parameters:
        RGB_DIM (tuple): Input shape for the RGB encoder.
        THERMAL_DIM (tuple): Input shape for the Thermal encoder.
        IND_EMB_DIM (int): Embedding dimension for individual modality reconstructions.
        COMB_EMB_DIM (int): Embedding dimension for combined modality reconstruction.

    Returns:
        AE (Model): Multi-modal autoencoder model.
    '''
    # Encoder inputs
    rgb_input_encoder = Input(RGB_DIM, name="rgb_input")
    thermal_input_encoder = Input(THERMAL_DIM, name="thermal_input")

    # Individual modality encodings
    rgb_output_encoder = encoder(rgb_input_encoder)
    thermal_output_encoder = encoder(thermal_input_encoder)

    # Individual modality encoders
    rgb_model_encoder = Model(rgb_input_encoder, rgb_output_encoder, name="rgb_encoder")
    thermal_model_encoder = Model(thermal_input_encoder, thermal_output_encoder, name="thermal_encoder")

    # Decoders for individual modality reconstruction (IND_EMB_DIM)
    rgb_input_decoder_individual = Input(IND_EMB_DIM, name="rgb_decoder_input_individual")
    thermal_input_decoder_individual = Input(IND_EMB_DIM, name="thermal_decoder_input_individual")

    rgb_output_decoder_individual = image_decoder(rgb_input_decoder_individual)
    thermal_output_decoder_individual = image_decoder(thermal_input_decoder_individual)

    rgb_model_decoder_individual = Model(rgb_input_decoder_individual, rgb_output_decoder_individual, name="rgb_decoder_individual")
    thermal_model_decoder_individual = Model(thermal_input_decoder_individual, thermal_output_decoder_individual, name="thermal_decoder_individual")

    # Decoders for combined reconstruction (COMB_EMB_DIM)
    rgb_input_decoder_combined = Input(COMB_EMB_DIM, name="rgb_decoder_input_combined")
    thermal_input_decoder_combined = Input(COMB_EMB_DIM, name="thermal_decoder_input_combined")

    rgb_output_decoder_combined = image_decoder_mm(rgb_input_decoder_combined)
    thermal_output_decoder_combined = image_decoder_mm(thermal_input_decoder_combined)

    rgb_model_decoder_combined = Model(rgb_input_decoder_combined, rgb_output_decoder_combined, name="rgb_decoder_combined")
    thermal_model_decoder_combined = Model(thermal_input_decoder_combined, thermal_output_decoder_combined, name="thermal_decoder_combined")

    # Generate embeddings
    rgb_emb_individual = rgb_model_encoder(rgb_input_encoder)
    thermal_emb_individual = thermal_model_encoder(thermal_input_encoder)

    concatenated_emb = concatenate([rgb_emb_individual, thermal_emb_individual], axis=-1, name="combined_embedding")

    # Model outputs
    AE = Model(
        [rgb_input_encoder, thermal_input_encoder],
        [
            concatenated_emb,
            rgb_model_decoder_combined(concatenated_emb),
            thermal_model_decoder_combined(concatenated_emb),
            rgb_model_decoder_individual(rgb_emb_individual),
            thermal_model_decoder_individual(thermal_emb_individual)
        ],
        name="multimodal_autoencoder_v4"
    )

    print(AE.summary())

    return AE



def image_reconstruction_triplet(S1_DIM, modality):
    '''
    Image reconstruction with triplet loss, supporting different encoders and decoders based on modality.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
        modality (str): One of "rgb", "lidar", or "hsi", determining which encoder and decoder to use.
    
    Returns:
        AE (Model): Autoencoder model outputting both the latent space and reconstructed image.
    '''
    input_encoder = Input(S1_DIM)
    
    # Select encoder based on modality
    if modality == "hsi":
        output_encoder = hsi_encoder(input_encoder)
    elif modality in ["rgb", "lidar"]:
        output_encoder = rgb_lidar_encoder(input_encoder)
    else:
        raise ValueError("Invalid modality. Choose from 'rgb', 'lidar', or 'hsi'.")

    model_encoder = Model(input_encoder, output_encoder, name=f"{modality}_encoder")

    input_decoder = Input(output_encoder.shape[1:])  # Match latent space shape
    
    # Select decoder based on modality
    if modality == "hsi":
        output_decoder = hsi_decoder_3x3(input_decoder)
    elif modality == "rgb":
        output_decoder = rgb_decoder_30x30(input_decoder)
    elif modality == "lidar":
        output_decoder = lidar_decoder_3x3(input_decoder)

    model_decoder = Model(input_decoder, output_decoder, name=f"{modality}_decoder")

    input_AE = Input(S1_DIM)
    
    # Output both latent representation and reconstructed image
    AE = Model(input_AE, [model_encoder(input_AE), model_decoder(model_encoder(input_AE))], 
               name=f"{modality}_autoencoder_triplet")

    print(AE.summary())
    
    return AE

def image_reconstruction_triplet4x4(S1_DIM, modality):
    '''
    Image reconstruction with triplet loss, supporting different encoders and decoders based on modality.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
        modality (str): One of "rgb", "lidar", or "hsi", determining which encoder and decoder to use.
    
    Returns:
        AE (Model): Autoencoder model outputting both the latent space and reconstructed image.
    '''
    input_encoder = Input(S1_DIM)
    
    # Select encoder based on modality
    if modality == "hsi":
        output_encoder = hsi_encoder(input_encoder)
    elif modality in ["rgb", "lidar"]:
        output_encoder = rgb_lidar_encoder(input_encoder)
    else:
        raise ValueError("Invalid modality. Choose from 'rgb', 'lidar', or 'hsi'.")

    model_encoder = Model(input_encoder, output_encoder, name=f"{modality}_encoder")

    input_decoder = Input(output_encoder.shape[1:])  # Match latent space shape
    
    # Select decoder based on modality
    if modality == "hsi":
        output_decoder = hsi_decoder_4x4(input_decoder)
    elif modality == "rgb":
        output_decoder = rgb_decoder_40x40(input_decoder)
    elif modality == "lidar":
        output_decoder = lidar_decoder_4x4(input_decoder)

    model_decoder = Model(input_decoder, output_decoder, name=f"{modality}_decoder")

    input_AE = Input(S1_DIM)
    
    # Output both latent representation and reconstructed image
    AE = Model(input_AE, [model_encoder(input_AE), model_decoder(model_encoder(input_AE))], 
               name=f"{modality}_autoencoder_triplet")

    print(AE.summary())
    
    return AE

def image_reconstruction_triplet6x6(S1_DIM, modality):
    '''
    Image reconstruction with triplet loss, supporting different encoders and decoders based on modality.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
        modality (str): One of "rgb", "lidar", or "hsi", determining which encoder and decoder to use.
    
    Returns:
        AE (Model): Autoencoder model outputting both the latent space and reconstructed image.
    '''
    input_encoder = Input(S1_DIM)
    
    # Select encoder based on modality
    if modality == "hsi":
        output_encoder = hsi_encoder(input_encoder)
    elif modality in ["rgb", "lidar"]:
        output_encoder = rgb_lidar_encoder(input_encoder)
    else:
        raise ValueError("Invalid modality. Choose from 'rgb', 'lidar', or 'hsi'.")

    model_encoder = Model(input_encoder, output_encoder, name=f"{modality}_encoder")

    input_decoder = Input(output_encoder.shape[1:])  # Match latent space shape
    
    # Select decoder based on modality
    if modality == "hsi":
        output_decoder = hsi_decoder_6x6(input_decoder)
    elif modality == "rgb":
        output_decoder = rgb_decoder_60x60(input_decoder)
    elif modality == "lidar":
        output_decoder = lidar_decoder_6x6(input_decoder)

    model_decoder = Model(input_decoder, output_decoder, name=f"{modality}_decoder")

    input_AE = Input(S1_DIM)
    
    # Output both latent representation and reconstructed image
    AE = Model(input_AE, [model_encoder(input_AE), model_decoder(model_encoder(input_AE))], 
               name=f"{modality}_autoencoder_triplet")

    print(AE.summary())
    
    return AE

def image_reconstruction_triplet8x8(S1_DIM, modality):
    '''
    Image reconstruction with triplet loss, supporting different encoders and decoders based on modality.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
        modality (str): One of "rgb", "lidar", or "hsi", determining which encoder and decoder to use.
    
    Returns:
        AE (Model): Autoencoder model outputting both the latent space and reconstructed image.
    '''
    input_encoder = Input(S1_DIM)
    
    # Select encoder based on modality
    if modality == "hsi":
        output_encoder = hsi_encoder(input_encoder)
    elif modality in ["rgb", "lidar"]:
        output_encoder = rgb_lidar_encoder(input_encoder)
    else:
        raise ValueError("Invalid modality. Choose from 'rgb', 'lidar', or 'hsi'.")

    model_encoder = Model(input_encoder, output_encoder, name=f"{modality}_encoder")

    input_decoder = Input(output_encoder.shape[1:])  # Match latent space shape
    
    # Select decoder based on modality
    if modality == "hsi":
        output_decoder = hsi_decoder_8x8(input_decoder)
    elif modality == "rgb":
        output_decoder = rgb_decoder_80x80(input_decoder)
    elif modality == "lidar":
        output_decoder = lidar_decoder_8x8(input_decoder)

    model_decoder = Model(input_decoder, output_decoder, name=f"{modality}_decoder")

    input_AE = Input(S1_DIM)
    
    # Output both latent representation and reconstructed image
    AE = Model(input_AE, [model_encoder(input_AE), model_decoder(model_encoder(input_AE))], 
               name=f"{modality}_autoencoder_triplet")

    print(AE.summary())
    
    return AE

def image_reconstruction_triplet7x7(S1_DIM, modality):
    '''
    Image reconstruction with triplet loss, supporting different encoders and decoders based on modality.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
        modality (str): One of "rgb", "lidar", or "hsi", determining which encoder and decoder to use.
    
    Returns:
        AE (Model): Autoencoder model outputting both the latent space and reconstructed image.
    '''
    input_encoder = Input(S1_DIM)
    
    # Select encoder based on modality
    if modality == "hsi":
        output_encoder = hsi_encoder(input_encoder)
    elif modality in ["rgb", "lidar"]:
        output_encoder = rgb_lidar_encoder(input_encoder)
    else:
        raise ValueError("Invalid modality. Choose from 'rgb', 'lidar', or 'hsi'.")

    model_encoder = Model(input_encoder, output_encoder, name=f"{modality}_encoder")

    input_decoder = Input(output_encoder.shape[1:])  # Match latent space shape
    
    # Select decoder based on modality
    if modality == "hsi":
        output_decoder = hsi_decoder_7x7(input_decoder)
    elif modality == "rgb":
        output_decoder = rgb_decoder_70x70(input_decoder)
    elif modality == "lidar":
        output_decoder = lidar_decoder_7x7(input_decoder)

    model_decoder = Model(input_decoder, output_decoder, name=f"{modality}_decoder")

    input_AE = Input(S1_DIM)
    
    # Output both latent representation and reconstructed image
    AE = Model(input_AE, [model_encoder(input_AE), model_decoder(model_encoder(input_AE))], 
               name=f"{modality}_autoencoder_triplet")

    print(AE.summary())
    
    return AE

def image_reconstruction_triplet11x11(S1_DIM, modality):
    '''
    Image reconstruction with triplet loss, supporting different encoders and decoders based on modality.
    
    Parameters:
        S1_DIM (tuple): Input shape for the encoder.
        modality (str): One of "rgb", "lidar", or "hsi", determining which encoder and decoder to use.
    
    Returns:
        AE (Model): Autoencoder model outputting both the latent space and reconstructed image.
    '''
    input_encoder = Input(S1_DIM)
    
    # Select encoder based on modality
    if modality == "hsi":
        output_encoder = hsi_encoder(input_encoder)
    elif modality in ["rgb", "lidar"]:
        output_encoder = rgb_lidar_encoder(input_encoder)
    else:
        raise ValueError("Invalid modality. Choose from 'rgb', 'lidar', or 'hsi'.")

    model_encoder = Model(input_encoder, output_encoder, name=f"{modality}_encoder")

    input_decoder = Input(output_encoder.shape[1:])  # Match latent space shape
    
    # Select decoder based on modality
    if modality == "hsi":
        output_decoder = hsi_decoder_11x11(input_decoder)
    elif modality == "rgb":
        output_decoder = rgb_decoder_110x110(input_decoder)
    elif modality == "lidar":
        output_decoder = lidar_decoder_11x11(input_decoder)

    model_decoder = Model(input_decoder, output_decoder, name=f"{modality}_decoder")

    input_AE = Input(S1_DIM)
    
    # Output both latent representation and reconstructed image
    AE = Model(input_AE, [model_encoder(input_AE), model_decoder(model_encoder(input_AE))], 
               name=f"{modality}_autoencoder_triplet")

    print(AE.summary())
    
    return AE

def image_reconstruction_triplet_task_head(EMB_DIM=1024):
    '''
    Image reconstruction for all kinds of triplets with different loss functions
    '''
    input_decoder = Input(EMB_DIM)
    
    output_decoder = image_decoder_new(input_decoder)
    model_decoder = Model(input_decoder, output_decoder, name='decoder')
    
   
    print(model_decoder.summary())
    
    return model_decoder




def classifier_triplet_setweight(S1_DIM, EMB_DIM=1024):
    '''
    Image classification for all kinds of triplets with different loss functions with set weighting
    '''
    input_encoder = Input(S1_DIM)    
    output_encoder_weights = encoder(input_encoder, EMB_DIM)
    model_encoder = Model(input_encoder, output_encoder_weights, name='encoder')
    
    # Classifier model
    cls_model = get_classifier_3class(EMB_DIM)
    
    input_classifier = Input(S1_DIM)
    encoded = model_encoder(input_classifier)
    
    e2e_classifier = Model(input_classifier, [encoded, cls_model(encoded)])
    print(e2e_classifier.summary())
    
    return e2e_classifier

def classifier_triplet_encoder_weighted(S1_DIM, EMB_DIM=1024):
    '''
    Image classification for all kinds of triplets with different loss functions with weights from encoder
    '''
    input_encoder = Input(S1_DIM)    
    output_encoder_weights = encoder_weighted(input_encoder, EMB_DIM)
    model_encoder = Model(input_encoder, output_encoder_weights, name='encoder')
    
    # Classifier model
    cls_model = get_classifier_3class_encoder_weighted(EMB_DIM+3)
    
    input_classifier = Input(S1_DIM)
    encoded = model_encoder(input_classifier)
    
    e2e_classifier = Model(input_classifier, [encoded, cls_model(encoded)])
    print(e2e_classifier.summary())
    
    return e2e_classifier

def classifier_triplet_encoder_weighted_softmax(S1_DIM, EMB_DIM=1024):
    '''
    Image classification for all kinds of triplets with different loss functions with weights from encoder
    '''
    input_encoder = Input(S1_DIM)    
    output_encoder_weights = encoder_weighted_softmax(input_encoder, EMB_DIM)
    model_encoder = Model(input_encoder, output_encoder_weights, name='encoder')
    
    # Classifier model
    cls_model = get_classifier_3class_encoder_weighted(EMB_DIM+3)
    
    input_classifier = Input(S1_DIM)
    encoded = model_encoder(input_classifier)
    
    e2e_classifier = Model(input_classifier, [encoded, cls_model(encoded)])
    print(e2e_classifier.summary())
    
    return e2e_classifier



def classifier_triplet_taskhead_weighted(S1_DIM, EMB_DIM=1024):
    '''
    End-to-end triplet loss + classifier with attention weights from the task head.
    '''
    # Encoder: Only outputs embeddings
    input_encoder = Input(S1_DIM)
    output_encoder = encoder(input_encoder, EMB_DIM)  # Update encoder to return only embeddings
    model_encoder = Model(input_encoder, output_encoder, name='encoder')

    cls_model = get_classifier_3class_task_head_weighted(EMB_DIM)

    # Input for the complete model
    input_classifier = Input(S1_DIM)
    
    # Pass through encoder
    encoded = model_encoder(input_classifier)
    
    # Pass through classifier to get both outputs
    cls_logits, attention_weights = cls_model(encoded)
    
    emb_weights = concatenate([encoded, attention_weights], axis=-1)
    

    # End-to-End Model
    e2e_classifier = Model(input_classifier, [emb_weights, cls_logits], name='e2e_classifier')
    print(e2e_classifier.summary())

    return e2e_classifier

def classifier_triplet_taskhead_weighted_softmax(S1_DIM, EMB_DIM=1024):
    '''
    End-to-end triplet loss + classifier with attention weights from the task head.
    '''
    # Encoder: Only outputs embeddings
    input_encoder = Input(S1_DIM)
    output_encoder = encoder(input_encoder, EMB_DIM)  # Update encoder to return only embeddings
    model_encoder = Model(input_encoder, output_encoder, name='encoder')

    cls_model = get_classifier_3class_task_head_weighted_softmax(EMB_DIM)

    # Input for the complete model
    input_classifier = Input(S1_DIM)
    
    # Pass through encoder
    encoded = model_encoder(input_classifier)
    
    # Pass through classifier to get both outputs
    cls_logits, attention_weights = cls_model(encoded)
    
    emb_weights = concatenate([encoded, attention_weights], axis=-1)
    

    # End-to-End Model
    e2e_classifier = Model(input_classifier, [emb_weights, cls_logits], name='e2e_classifier')
    print(e2e_classifier.summary())

    return e2e_classifier


def regressor_triplet_setweight(S1_DIM, EMB_DIM=1024):
    '''
    Image classification for all kinds of triplets with different loss functions with set weighting
    '''
    input_encoder = Input(S1_DIM)    
    output_encoder_weights = encoder(input_encoder, EMB_DIM)
    model_encoder = Model(input_encoder, output_encoder_weights, name='encoder')
    
    # Classifier model
    reg_model = continuous_regression4(EMB_DIM)
    
    input_regressor = Input(S1_DIM)
    encoded = model_encoder(input_regressor)
    
    e2e_regressor = Model(input_regressor, [encoded, reg_model(encoded)])
    print(e2e_regressor.summary())
    
    return e2e_regressor

def regressor_triplet_encoder_weighted_sigmoid(S1_DIM, EMB_DIM=1024):
    '''
    Image classification for all kinds of triplets with different loss functions with weights from encoder
    '''
    input_encoder = Input(S1_DIM)    
    output_encoder_weights = encoder_weighted(input_encoder, EMB_DIM)
    model_encoder = Model(input_encoder, output_encoder_weights, name='encoder')
    
    # Classifier model
    cls_model = corner_predictions4_encoder_weighted(EMB_DIM+3)
    
    input_classifier = Input(S1_DIM)
    encoded = model_encoder(input_classifier)
    
    e2e_regressor = Model(input_classifier, [encoded, cls_model(encoded)])
    print(e2e_regressor.summary())
    
    return e2e_regressor

def regressor_triplet_encoder_weighted_softmax(S1_DIM, EMB_DIM=1024):
    '''
    Image classification for all kinds of triplets with different loss functions with weights from encoder
    '''
    input_encoder = Input(S1_DIM)    
    output_encoder_weights = encoder_weighted_softmax(input_encoder, EMB_DIM)
    model_encoder = Model(input_encoder, output_encoder_weights, name='encoder')
    
    # Classifier model
    cls_model = corner_predictions4_encoder_weighted(EMB_DIM+3)
    
    input_classifier = Input(S1_DIM)
    encoded = model_encoder(input_classifier)
    
    e2e_regressor = Model(input_classifier, [encoded, cls_model(encoded)])
    print(e2e_regressor.summary())
    
    return e2e_regressor



def regressor_triplet_taskhead_weighted_sigmoid(S1_DIM, EMB_DIM=1024):
    '''
    End-to-end triplet loss + classifier with attention weights from the task head.
    '''
    # Encoder: Only outputs embeddings
    input_encoder = Input(S1_DIM)
    output_encoder = encoder(input_encoder, EMB_DIM)  # Update encoder to return only embeddings
    model_encoder = Model(input_encoder, output_encoder, name='encoder')

    task_head = corner_predictions4_task_head_weighted_sigmoid(EMB_DIM)

    # Input for the complete model
    input_regressor = Input(S1_DIM)
    
    # Pass through encoder
    encoded = model_encoder(input_regressor)
    
    # Pass through classifier to get both outputs
    reg_out, attention_weights = task_head(encoded)
    
    emb_weights = concatenate([encoded, attention_weights], axis=-1)
    

    # End-to-End Model
    e2e_regressor = Model(input_regressor, [emb_weights, reg_out], name='e2e_regressor')
    print(e2e_regressor.summary())

    return e2e_regressor

def regressor_triplet_taskhead_weighted_softmax(S1_DIM, EMB_DIM=1024):
    '''
    End-to-end triplet loss + classifier with attention weights from the task head.
    '''
    # Encoder: Only outputs embeddings
    input_encoder = Input(S1_DIM)
    output_encoder = encoder(input_encoder, EMB_DIM)  # Update encoder to return only embeddings
    model_encoder = Model(input_encoder, output_encoder, name='encoder')

    task_head = corner_predictions4_task_head_weighted_softmax(EMB_DIM)

    # Input for the complete model
    input_regressor = Input(S1_DIM)
    
    # Pass through encoder
    encoded = model_encoder(input_regressor)
    
    # Pass through classifier to get both outputs
    reg_out, attention_weights = task_head(encoded)
    
    emb_weights = concatenate([encoded, attention_weights], axis=-1)
    

    # End-to-End Model
    e2e_regressor = Model(input_regressor, [emb_weights, reg_out], name='e2e_regressor')
    print(e2e_regressor.summary())

    return e2e_regressor

def DC_CB_triplet_setweight(S1_DIM, EMB_DIM=1024):
    '''
    Image classification for all kinds of triplets with different loss functions with set weighting
    '''
    input_encoder = Input(S1_DIM)    
    output_encoder = encoder(input_encoder, EMB_DIM)
    model_encoder = Model(input_encoder, output_encoder, name='encoder')
    
    # Classifier model
    cls_model_class = get_classifier_3class(EMB_DIM)
   
    # Regression model
    reg_model = corner_predictions4(EMB_DIM)
    
    input_DC_CB = Input(S1_DIM)
    
    
    encoded = model_encoder(input_DC_CB)
    
    cls_out_class = cls_model_class(encoded)
    
    reg_out_box = reg_model(encoded)
    
    e2e_DC_CB = Model(input_DC_CB, [encoded, cls_out_class, reg_out_box])
    print(e2e_DC_CB.summary())
    
    return e2e_DC_CB

def DC_CB_triplet_encoder_weighted_sigmoid(S1_DIM, EMB_DIM=1024):
    '''
    Image classification for all kinds of triplets with different loss functions with weights from encoder
    '''
    input_encoder = Input(S1_DIM)    
    output_encoder_weights = encoder_weighted(input_encoder, EMB_DIM)
    model_encoder = Model(input_encoder, output_encoder_weights, name='encoder')
    
    # Classifier model
    reg_model = corner_predictions4_encoder_weighted(EMB_DIM+3)
    cls_model = get_classifier_3class_encoder_weighted(EMB_DIM+3)
    
    input_classifier = Input(S1_DIM)
    encoded = model_encoder(input_classifier)
    
    e2e_DC_CB = Model(input_classifier, [encoded, cls_model(encoded), reg_model(encoded)])
    print(e2e_DC_CB.summary())
    
    return e2e_DC_CB

def DC_CB_triplet_encoder_weighted_softmax(S1_DIM, EMB_DIM=1024):
    '''
    Image classification for all kinds of triplets with different loss functions with weights from encoder
    '''
    input_encoder = Input(S1_DIM)    
    output_encoder_weights = encoder_weighted_softmax(input_encoder, EMB_DIM)
    model_encoder = Model(input_encoder, output_encoder_weights, name='encoder')
    
    # Classifier model
    reg_model = corner_predictions4_encoder_weighted(EMB_DIM+3)
    cls_model = get_classifier_3class_encoder_weighted(EMB_DIM+3)
    
    input_classifier = Input(S1_DIM)
    encoded = model_encoder(input_classifier)
    
    e2e_DC_CB = Model(input_classifier, [encoded, cls_model(encoded), reg_model(encoded)])
    print(e2e_DC_CB.summary())
    
    return e2e_DC_CB


def box_metric_predictor(input_dim) :
    
    input1 = Input(input_dim)
    out = Dense(64, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input1)
    out = BatchNormalization()(out)
    
    out = Dense(4, kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Activation('sigmoid')(out)

    metric_predictor = Model(input1, out)
    
    return metric_predictor

def getTripletBoxMetric(S1_DIM, EMB_DIM=1024) :
    '''
    Box metric predictor with triplet loss
    '''

#     input_modality1 = Input(S1_DIM)    
#     output_modality1 = encoder(input_modality1, EMB_DIM)
    
#     EncoderModality1 = Model(input_modality1, output_modality1)
    # print(EncoderModality1.summary())

    input_modality1 = Input(EMB_DIM)    
    output_modality1 = image_decoder(input_modality1)
    
    EncoderModality1 = Model(input_modality1, output_modality1)
    print(EncoderModality1.summary())
    
    
#     # Get 3 instances of the Modality 1 encoder
#     S1_instance1_inp = Input(S1_DIM)
#     S1_instance1 = EncoderModality1(S1_instance1_inp)
    
    

    
#     # Classifier model
#     metric_predictor = box_metric_predictor(EMB_DIM)
    
#     metric_out = metric_predictor(S1_instance1)
#     #print(cls_model.summary())
    
#     model = Model(S1_instance1_inp, [S1_instance1, metric_out])
    
#     model.compile(optimizer= 'Adam', loss= [mean_squared_loss, mean_squared_loss])

    # print(model.summary())

    return

def getBoxMetric(S1_DIM, EMB_DIM=256) :
    '''
    Box metric predictor without triplet loss
    '''

    input_modality1 = Input(S1_DIM)    
    output_modality1 = encoder(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1)


    # Get instances of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)

    
    # Classifier model
    metric_predictor = box_metric_predictor(EMB_DIM)
    
    metric_out = metric_predictor(S1_instance1)
    #print(cls_model.summary())
    
    model = Model(S1_instance1_inp,  metric_out)
    
    model.compile(optimizer= 'Adam', loss = mean_squared_loss)

    # print(model.summary())

    return model

# Multimodal triplet loss
def multimodal_triplet_loss(margin=0.1, EMB_DIM=32) :


    def inner_multimodal_triplet_loss(y_true, y_pred) :

        anchorS1, positiveS1, negativeS1, anchorS2 = y_pred[:,:EMB_DIM], y_pred[:,EMB_DIM:2*EMB_DIM], y_pred[:,2*EMB_DIM:3*EMB_DIM], y_pred[:,3*EMB_DIM:]

        distance1 = tf.reduce_mean(tf.square(anchorS1 - positiveS1), axis=1)
        distance2 = tf.reduce_mean(tf.square(anchorS1 - negativeS1), axis=1)
        distance3 = tf.reduce_mean(tf.square(anchorS2 - positiveS1), axis=1) 
        distance4 = tf.reduce_mean(tf.square(anchorS2 - negativeS1), axis=1) 
        distance5 = tf.reduce_mean(tf.square(anchorS1 - anchorS2), axis=1)
        
        loss1 = tf.reduce_mean(tf.maximum(distance1 - distance2 + margin, 0))
        loss2 = tf.reduce_mean(tf.maximum(distance3 - distance4 + margin, 0))

        multimodal_loss = loss1 + loss2

        # SE_loss = tf.reduce_mean(distance5) # Similarity Enhancement term

        loss = multimodal_loss #+ gamma * SE_loss # gamma was set to 0.4 earlier
        #gamma is a simility enhancement term. brings the embeddings of the two modalities closer
        #further experiment here
        
        return loss
        

    return inner_multimodal_triplet_loss

# This function computes the Triplet loss
def triplet_loss(margin=0.1, EMB_DIM=32) :
    """
    Creates a triplet loss function with a specified margin and embedding dimension.

    Args:
    margin (float): The margin by which the distance between the anchor and negative
                    should be greater than the distance between the anchor and positive.
    EMB_DIM (int):   The dimension of the embeddings.

    Returns:
    function: A loss function that can be used in model training.
    """

    def inner_triplet_loss(y_true, y_pred) :
        """
        Computes the triplet loss for a batch of data.

        Args:
        y_true (tensor): True labels (not used in this loss function but required for compatibility).
        y_pred (tensor): Predicted embeddings, expected to be in the form of concatenated embeddings
                         for the anchor, positive, and negative examples.

        Returns:
        tensor: The computed triplet loss.
        """
        # Split the predicted embeddings into anchor, positive, and negative embeddings
        anchorS1, positiveS1, negativeS1 = y_pred[:,:EMB_DIM], y_pred[:,EMB_DIM:2*EMB_DIM], y_pred[:,2*EMB_DIM:3*EMB_DIM]
        
        # Compute the squared distances between the anchor and positive embeddings
        distance1 = tf.reduce_mean(tf.square(anchorS1 - positiveS1), axis=1)
        
        # Compute the squared distances between the anchor and negative embeddings
        distance2 = tf.reduce_mean(tf.square(anchorS1 - negativeS1), axis=1)
        
        # Compute the triplet loss using the margin
        loss = tf.reduce_mean(tf.maximum(distance1 - distance2 + margin, 0))
        
        return loss

    return inner_triplet_loss




# # CBAM Spatial Attention
# def spatial_attention(input, id):
    
#     input = input[:,:,:,0]
       
#     mu = tf.reduce_mean(input, axis= -1, keepdims= True)
#     maximum = tf.reduce_max(input, axis= -1, keepdims= True)

#     out = concatenate([mu, maximum], axis= -1)
#     out = Conv2D(1, (1,1), padding= 'same', use_bias= False, name= 'attn'+str(id))(out)
#     out = out / 2 # Temperature parameter of sigmoid
#     out = Activation('sigmoid')(out)
#     out = tf.expand_dims(out, axis= -1)

#     return out



# Classifier: Outputs both classification logits and attention weights
def get_classifier_3class_task_head_weighted_softmax(input_dim):
    '''
    Classifier that outputs both class probabilities and attention weights.
    '''
    input_cls = Input(input_dim)

    # Generate embeddings for classification
    cls_features = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input_cls)
    cls_features = BatchNormalization()(cls_features)
    cls_features = Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(cls_features)
    cls_features = BatchNormalization()(cls_features)
    cls_features = Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(cls_features)
    cls_features = BatchNormalization()(cls_features)
    cls_features = Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(cls_features)
    cls_features = BatchNormalization()(cls_features)

    # Classification logits
    cls_logits = Dense(3, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotNormal(), name='classification_output')(cls_features)

    # Attention weights
    attention_weights = Dense(3, activation='softmax', kernel_initializer='zeros', name='attention_weights')(cls_features)

    # Combined Model
    classifier = Model(input_cls, [cls_logits, attention_weights], name='classifier_with_attention')
    return classifier

# Classifier: Outputs both classification logits and attention weights
def get_classifier_3class_task_head_weighted(input_dim):
    '''
    Classifier that outputs both class probabilities and attention weights.
    '''
    input_cls = Input(input_dim)

    # Generate embeddings for classification
    cls_features = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input_cls)
    cls_features = BatchNormalization()(cls_features)
    cls_features = Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(cls_features)
    cls_features = BatchNormalization()(cls_features)
    cls_features = Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(cls_features)
    cls_features = BatchNormalization()(cls_features)
    cls_features = Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(cls_features)
    cls_features = BatchNormalization()(cls_features)

    # Classification logits
    cls_logits = Dense(3, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotNormal(), name='classification_output')(cls_features)

    # Attention weights
    attention_weights = Dense(3, activation='sigmoid', kernel_initializer='zeros', name='attention_weights')(cls_features)

    # Combined Model
    classifier = Model(input_cls, [cls_logits, attention_weights], name='classifier_with_attention')
    return classifier

def get_classifier_3class_task_head_mtl(input_dim):
    '''
    Classifier that outputs both class probabilities and the
    second-last layer (features before the final dense layers).
    '''
    input_cls = Input(input_dim)

    # Generate embeddings for classification
    cls_features = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input_cls)
    cls_features = BatchNormalization()(cls_features)
    cls_features = Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(cls_features)
    cls_features = BatchNormalization()(cls_features)
    cls_features = Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(cls_features)
    cls_features = BatchNormalization()(cls_features)
    cls_features = Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(cls_features)
    cls_features = BatchNormalization()(cls_features)

    # Second-last layer output
    second_last_layer = cls_features

    # Classification logits
    cls_logits = Dense(3, activation='softmax', kernel_initializer=tf.keras.initializers.GlorotNormal(), name='classification_output')(cls_features)

    # Combined Model
    classifier = Model(input_cls, [second_last_layer, cls_logits], name='classifier')
    return classifier

def get_classifier_3class_encoder_weighted(input_dim):
    
    input1 = Input(input_dim)
    # input2 = input1[:, :-3]
    input2 = Lambda(lambda x: x[:, :-3], name='slice_input')(input1)
    
    out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input2)
    out = BatchNormalization()(out)
    out = Dense(64, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(32, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(16, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(3, activation= 'softmax', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)

    # Create model
    classifier = Model(input1, out)
    
    return classifier


def get_classifier_3class(input_dim):
    
    input1 = Input(input_dim)
    out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input1)
    out = BatchNormalization()(out)
    out = Dense(64, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(32, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(16, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(3, activation= 'softmax', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)

    # Create model
    classifier = Model(input1, out)
    
    return classifier

def get_classifier_10class(input_dim):
    
    input1 = Input(input_dim)
    out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input1)
    out = BatchNormalization()(out)
    out = Dense(64, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(32, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(16, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(10, activation= 'softmax', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)

    # Create model
    classifier = Model(input1, out)
    
    return classifier

def get_classifier_23class(input_dim):
    
    input1 = Input(input_dim)
    out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input1)
    out = BatchNormalization()(out)
    out = Dense(64, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(32, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(23, activation= 'softmax', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)

    # Create model
    classifier = Model(input1, out)
    
    return classifier


def discrete_classification(EMB_DIM=1024) :
    '''
    Single task classification from pre-trained embeddings.
    '''

    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(EMB_DIM)

    # Classifier model
    cls_model = get_classifier_10class(EMB_DIM)
    
    cls_out = cls_model(S1_instance1_inp)
    
    model = Model(S1_instance1_inp, cls_out)
    
    model.compile(optimizer= 'Adam', loss= mean_squared_loss)

    # print(model.summary())

    return model

def discrete_classification9(EMB_DIM=1024) :
    '''
    Single task classification from pre-trained embeddings.
    '''

    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(EMB_DIM)

    # Classifier model
    cls_model = get_classifier_9class(EMB_DIM)
    
    cls_out = cls_model(S1_instance1_inp)
    
    model = Model(S1_instance1_inp, cls_out)
    
    model.compile(optimizer= 'Adam', loss= mean_squared_loss)

    # print(model.summary())

    return model

# Regression model for corners prediction - xmin, xmax, ymin, ymax
def corner_predictions4(input_dim) :
    
    input1 = Input(input_dim)
    out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input1)
    out = BatchNormalization()(out)
    out = Dense(64, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(32, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(16, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(4, kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Activation('sigmoid')(out)

    box_predictor = Model(input1, out)
    
    return box_predictor

# Regression model for corners prediction - xmin, xmax, ymin, ymax
def corner_predictions2(input_dim) :
    
    input1 = Input(input_dim)
    out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input1)
    out = BatchNormalization()(out)
    out = Dense(64, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(32, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(16, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(2, kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = Activation('sigmoid')(out)

    box_predictor = Model(input1, out)
    
    return box_predictor


    
# Classifier: Outputs both classification logits and attention weights
def corner_predictions4_task_head_weighted_softmax(input_dim):
    '''
    Regressor that outputs both sigmoid regression values and attention weights.
    '''
    input_reg = Input(input_dim)

    # Generate layers for regression
    reg_features = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input_reg)
    reg_features = BatchNormalization()(reg_features)
    reg_features = Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(reg_features)
    reg_features = BatchNormalization()(reg_features)
    reg_features = Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(reg_features)
    reg_features = BatchNormalization()(reg_features)
    reg_features = Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(reg_features)
    reg_features = BatchNormalization()(reg_features)
    
    reg_out = Dense(4, activation = 'sigmoid', kernel_initializer= tf.keras.initializers.GlorotNormal())(reg_features)

    # Attention weights
    attention_weights = Dense(3, activation='softmax', kernel_initializer='zeros', name='attention_weights')(reg_features)

    # Combined Model
    regressor = Model(input_reg, [reg_out, attention_weights], name='regressor_with_attention')
    return regressor

# Classifier: Outputs both classification logits and attention weights
def corner_predictions4_task_head_weighted_sigmoid(input_dim):
    '''
    Regressor that outputs both sigmoid regression values and attention weights.
    '''
    input_reg = Input(input_dim)

    # Generate layers for regression
    reg_features = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input_reg)
    reg_features = BatchNormalization()(reg_features)
    reg_features = Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(reg_features)
    reg_features = BatchNormalization()(reg_features)
    reg_features = Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(reg_features)
    reg_features = BatchNormalization()(reg_features)
    reg_features = Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(reg_features)
    reg_features = BatchNormalization()(reg_features)
    
    reg_out = Dense(4, activation = 'sigmoid', kernel_initializer= tf.keras.initializers.GlorotNormal())(reg_features)

    # Attention weights
    attention_weights = Dense(3, activation='sigmoid', kernel_initializer='zeros', name='attention_weights')(reg_features)

    # Combined Model
    regressor = Model(input_reg, [reg_out, attention_weights], name='regressor_with_attention')
    return regressor

def corner_predictions4_task_head_weighted_mtl(input_dim):
    '''
    Regressor that outputs both sigmoid regression values and the 
    second-last layer (features before the final dense layers).
    '''
    input_reg = Input(input_dim)

    # Generate layers for regression
    reg_features = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input_reg)
    reg_features = BatchNormalization()(reg_features)
    reg_features = Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(reg_features)
    reg_features = BatchNormalization()(reg_features)
    reg_features = Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(reg_features)
    reg_features = BatchNormalization()(reg_features)
    reg_features = Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(reg_features)
    reg_features = BatchNormalization()(reg_features)

    # Second-last layer output
    second_last_layer = reg_features

    # Regression output
    reg_out = Dense(4, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotNormal())(reg_features)

    # Combined Model
    regressor = Model(input_reg, [second_last_layer, reg_out], name='regressor')
    return regressor



def corner_predictions4_encoder_weighted(input_dim):
    
    input1 = Input(input_dim)
    # input2 = input1[:, :-3]
    input2 = Lambda(lambda x: x[:, :-3], name='slice_input')(input1)
    
    # Generate layers for regression
    reg_features = Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(input2)
    reg_features = BatchNormalization()(reg_features)
    reg_features = Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(reg_features)
    reg_features = BatchNormalization()(reg_features)
    reg_features = Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(reg_features)
    reg_features = BatchNormalization()(reg_features)
    reg_features = Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.GlorotNormal())(reg_features)
    reg_features = BatchNormalization()(reg_features)
    
    reg_out = Dense(4, activation = 'sigmoid', kernel_initializer= tf.keras.initializers.GlorotNormal())(reg_features)

    # Combined Model
    regressor = Model(input1, reg_out, name='regressor_with_attention')
    return regressor


def continuous_regression4(EMB_DIM=1024) :
    '''
    Single task classification from pre-trained embeddings.
    '''

    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(EMB_DIM)

    # Classifier model
    reg_model = corner_predictions4(EMB_DIM)
    
    reg_out = reg_model(S1_instance1_inp)
    
    model = Model(S1_instance1_inp, reg_out)
    
    model.compile(optimizer= 'Adam', loss= mean_squared_loss)

    # print(model.summary())

    return model

def continuous_regression2(EMB_DIM=1024) :
    '''
    Single task classification from pre-trained embeddings.
    '''

    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(EMB_DIM)

    # Classifier model
    reg_model = corner_predictions2(EMB_DIM)
    
    reg_out = reg_model(S1_instance1_inp)
    
    model = Model(S1_instance1_inp, reg_out)
    
    model.compile(optimizer= 'Adam', loss= mean_squared_loss)

    # print(model.summary())

    return model


def get_DC_CB(input_dim):
    
    input1 = Input(input_dim)
    out = Dense(128, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(input1)
    out = BatchNormalization()(out)
    out = Dense(64, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(32, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    out = BatchNormalization()(out)
    out = Dense(16, activation= 'relu', kernel_initializer= tf.keras.initializers.GlorotNormal())(out)
    shared_layers = BatchNormalization()(out)
    
    
    cls_out = Dense(3, activation= 'softmax', kernel_initializer= tf.keras.initializers.GlorotNormal())(shared_layers)
    reg_out = Dense(4, activation='sigmoid', kernel_initializer= tf.keras.initializers.GlorotNormal())(shared_layers)

    # Create model
    DB_CB = Model(input1, [cls_out, reg_out])
    
    return DB_CB



def DB_CB(EMB_DIM=1024):
    '''
    Multitask discrete box classification and continuous box regression
    '''
    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(EMB_DIM)

    # Classifier model
    cls_model = get_classifier_3class(EMB_DIM)
    
    cls_out = cls_model(S1_instance1_inp)

    # Regression model
    reg_model = corner_predictions4(EMB_DIM)
    
    reg_out = reg_model(S1_instance1_inp)

    
    model = Model(S1_instance1_inp, [cls_out, reg_out])
    
    model.compile(optimizer= 'Adam', loss = [mean_squared_loss, mean_squared_loss])


    # print(model.summary())

    return model

def DB_DC(EMB_DIM=1024):
    '''
    Multitask discrete class classification and continuous box regression
    '''
    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(EMB_DIM)

    # Classifier model
    cls_model_box = get_classifier_3class(EMB_DIM)
    
    cls_out_box = cls_model_box(S1_instance1_inp)

    # Classifier model
    cls_model_cls = get_classifier_3class(EMB_DIM)
    
    cls_out_cls = cls_model_cls(S1_instance1_inp)

    
    model = Model(S1_instance1_inp, [cls_out_box, cls_out_cls])
    
    model.compile(optimizer= 'Adam', loss = [mean_squared_loss, mean_squared_loss])


    # print(model.summary())

    return model

def DCB_CB(EMB_DIM=1024):
    '''
    Multitask discrete box and class classification and continuous box regression
    '''
    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(EMB_DIM)

    # Classifier model
    cls_model_class = get_classifier_3class(EMB_DIM)
    
    cls_out_class = cls_model_class(S1_instance1_inp)
    
    # Classifier model
    cls_model_box = get_classifier_3class(EMB_DIM)
    
    cls_out_box = cls_model_box(S1_instance1_inp)

    # Regression model
    reg_model = corner_predictions4(EMB_DIM)
    
    reg_out = reg_model(S1_instance1_inp)

    
    model = Model(S1_instance1_inp, [cls_out_class, cls_out_box, reg_out])
    
    model.compile(optimizer= 'Adam', loss = [mean_squared_loss, mean_squared_loss, mean_squared_loss])


    # print(model.summary())

    return model

def DC_CB(EMB_DIM=1024):
    '''
    Multitask discrete class classification and continuous box regression
    '''
    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(EMB_DIM)

    # Classifier model
    cls_model_class = get_classifier_3class(EMB_DIM)
    
    cls_out_class = cls_model_class(S1_instance1_inp)
    

    # Regression model
    reg_model = corner_predictions4(EMB_DIM)
    
    reg_out = reg_model(S1_instance1_inp)

    
    model = Model(S1_instance1_inp, [cls_out_class, reg_out])
    
    model.compile(optimizer= 'Adam', loss = [mean_squared_loss, mean_squared_loss])


    # print(model.summary())

    return model

def regressor_triplet_taskhead_weighted_softmax(S1_DIM, EMB_DIM=1024):
    '''
    End-to-end triplet loss + classifier with attention weights from the task head.
    '''
    # Encoder: Only outputs embeddings
    input_encoder = Input(S1_DIM)
    output_encoder = encoder(input_encoder, EMB_DIM)  # Update encoder to return only embeddings
    model_encoder = Model(input_encoder, output_encoder, name='encoder')

    task_head = corner_predictions4_task_head_weighted_softmax(EMB_DIM)

    # Input for the complete model
    input_regressor = Input(S1_DIM)
    
    # Pass through encoder
    encoded = model_encoder(input_regressor)
    
    # Pass through classifier to get both outputs
    reg_out, attention_weights = task_head(encoded)
    
    emb_weights = concatenate([encoded, attention_weights], axis=-1)
    

    # End-to-End Model
    e2e_regressor = Model(input_regressor, [emb_weights, reg_out], name='e2e_regressor')
    print(e2e_regressor.summary())

    return e2e_regressor
    
def DC_CB_sigmoid(S1_DIM, EMB_DIM=1024):
    '''
    Multitask discrete class classification and continuous box regression
    '''
    # Encoder: Only outputs embeddings
    input_encoder = Input(S1_DIM)
    output_encoder = encoder(input_encoder, EMB_DIM)  # Update encoder to return only embeddings
    model_encoder = Model(input_encoder, output_encoder, name='encoder')
    
    # Input for the complete model
    input_DC_CB = Input(S1_DIM)
    
    # Pass through encoder
    encoded = model_encoder(input_DC_CB)

    # Classifier model
    cls_model_class = get_classifier_3class_task_head_mtl(EMB_DIM)
    
    # Extract outputs from classifier model
    cls_second_last_layer, cls_out_class = cls_model_class(encoded)
    
    # Regression model
    reg_model = corner_predictions4_task_head_weighted_mtl(EMB_DIM)
    
    # Extract outputs from regression model
    reg_second_last_layer, reg_out = reg_model(encoded)

    # Concatenate second_last_layer outputs for combined attention weights
    combined_second_last_layer = tf.keras.layers.Concatenate()([cls_second_last_layer, reg_second_last_layer])
    combined_attention_weights = Dense(3, activation='sigmoid', kernel_initializer='zeros', name='combined_attention_weights')(combined_second_last_layer)
    
    emb_weights = concatenate([encoded, combined_attention_weights], axis=-1)

    # Final model
    model = Model(input_DC_CB, [emb_weights, cls_out_class, reg_out])
    
    model.compile(optimizer='Adam', loss=[mean_squared_loss, mean_squared_loss, mean_squared_loss])

    return model

def DC_CB_softmax(S1_DIM, EMB_DIM=1024):
    '''
    Multitask discrete class classification and continuous box regression
    '''
    # Encoder: Only outputs embeddings
    input_encoder = Input(S1_DIM)
    output_encoder = encoder(input_encoder, EMB_DIM)  # Update encoder to return only embeddings
    model_encoder = Model(input_encoder, output_encoder, name='encoder')
    
    # Input for the complete model
    input_DC_CB = Input(S1_DIM)
    
    # Pass through encoder
    encoded = model_encoder(input_DC_CB)

    # Classifier model
    cls_model_class = get_classifier_3class_task_head_mtl(EMB_DIM)
    
    # Extract outputs from classifier model
    cls_second_last_layer, cls_out_class = cls_model_class(encoded)
    
    # Regression model
    reg_model = corner_predictions4_task_head_weighted_mtl(EMB_DIM)
    
    # Extract outputs from regression model
    reg_second_last_layer, reg_out = reg_model(encoded)

    # Concatenate second_last_layer outputs for combined attention weights
    combined_second_last_layer = tf.keras.layers.Concatenate()([cls_second_last_layer, reg_second_last_layer])
    combined_attention_weights = Dense(3, activation='softmax', kernel_initializer='zeros', name='combined_attention_weights')(combined_second_last_layer)
    
    emb_weights = concatenate([encoded, combined_attention_weights], axis=-1)

    # Final model
    model = Model(input_DC_CB, [emb_weights, cls_out_class, reg_out])
    
    model.compile(optimizer='Adam', loss=[mean_squared_loss, mean_squared_loss, mean_squared_loss])

    return model



    
def image_decoder_mm(input):
    x = Dense(512, kernel_initializer= tf.keras.initializers.GlorotNormal())(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)    

    x = Reshape((1, 1, 512))(x)

    x = UpSampling2D((3,3))(x)
    previous_block_activation = x

    filters = [128, 128, 64, 64, 32]
    upsample_factors = [3, 3, 3, 2, 2]
    for i in range(5) :    

        # Block i
        x = Conv2DTranspose(filters[i], (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 1x1 to 2x2
        x = ReLU()(x)    
        x = Conv2DTranspose(filters[i], (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 2x2 to 6x6
        x = ReLU()(x)

        x = UpSampling2D((upsample_factors[i], upsample_factors[i]))(x)

        # Project residual
        residual = UpSampling2D(upsample_factors[i])(previous_block_activation)
        residual = Conv2D(filters[i], 1, padding="same")(residual)
        x = Add()([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside the next residual

    x = Cropping2D(((0,24), (0, 24)))(x)

    x = Conv2D(32, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    # x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(16, (2, 2), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    x = ReLU()(x)

    x = Conv2D(8, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33
    x = ReLU()(x)

    x = Conv2D(3, (3, 3), kernel_initializer=tf.keras.initializers.GlorotNormal(), strides=(1, 1), padding='same')(x)  # 35x35 to 33x33

    output = Activation('sigmoid')(x)
    
    return output

def modality_selection_attention(input1, input2, identifier) :
    # attention per modality, same weight for all embeddings
    
    mask1 = Dense(1, use_bias= False, kernel_initializer= tf.keras.initializers.GlorotNormal())(input1)
    mask2 = Dense(1, use_bias= False, kernel_initializer= tf.keras.initializers.GlorotNormal())(input2)

    mask = concatenate([mask1, mask2], axis= -1)    

    soft_mask = Activation('softmax', name= 'attn_'+identifier)(mask)   
    

    mask1_soft = 1 + soft_mask[:,0]
    
    mask2_soft = 1 + soft_mask[:,1]
    
    out1_normalized = Multiply()([input1, mask1_soft])
    out2_normalized = Multiply()([input2, mask2_soft])

    out = concatenate([out1_normalized, out2_normalized], axis= -1)
    
    return out

def modality_selection_attention2(input1, EMB_DIM, identifier):
    # Attention per dimension of embedding

    ### With Squeeze and Excitation
    
    # Concatenate input
    #merged_input = concatenate([input1, input2], axis=-1)
    merged_input = input1

    # Squeeze step
    reduction = 4
    REDUCED_DIM = int((EMB_DIM) / reduction)
    squeeze = Dense(REDUCED_DIM, activation= 'relu', use_bias= False)(merged_input)

    # Excitation step
    #excite = Dense(EMB_DIM*2, activation= 'softmax', name= 'attn_' + identifier, use_bias= False)(squeeze)
    excite = Dense(EMB_DIM, activation= 'sigmoid', name= 'attn_' + identifier, use_bias= False, kernel_initializer=Ones())(squeeze)
    
    # Use the attention weights to scale the inputs
    out1_normalized = Multiply()([input1, excite])
    
    #out2_normalized = Multiply()([input2, excite[:, EMB_DIM:]])

    # Concatenate the scaled inputs
    # out = concatenate([out1_normalized, out2_normalized], axis=-1)
    out = out1_normalized

    '''
    ### Without Squeeze and Excitation

    merged_input = concatenate([input1, input2], axis=-1)
    mask = Dense(EMB_DIM, activation= 'softmax', name= 'attn_' + identifier, use_bias=False, kernel_initializer=tf.keras.initializers.GlorotNormal())(merged_input)

    # Apply softmax over the concatenated attention weights
    soft_mask = Activation('softmax', name='attn_' + identifier)(mask)

    # Use the attention weights to scale the inputs
    out1_normalized = Multiply()([input1, soft_mask[:, :EMB_DIM]])
    out2_normalized = Multiply()([input2, soft_mask[:, EMB_DIM:]])

    # Concatenate the scaled inputs
    out = concatenate([out1_normalized, out2_normalized], axis=-1)
    '''
    return out

def getTriplet(S1_DIM, EMB_DIM=256) :
    '''Triplet Network only
    
    '''
    
    input_modality1 = Input(S1_DIM)    
    output_modality1 = encoder(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1)
    
    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)

    S1_instance1 = EncoderModality1(S1_instance1_inp)

    # box_predictor = corner_predictions(EMB_DIM)
    
    #new decoder for mask
    # decoded_mask = mask_decoder(S1_instance1)
    
    # corners1 = box_predictor(S1_instance1)
    
    # Classifier + BBox model
    model = Model(S1_instance1_inp, S1_instance1)
    
    model.compile(optimizer= 'Adam', loss= mean_squared_loss)

    # print(model.summary())

    return model


def getTripletMTL(S1_DIM, EMB_DIM=256) :
    '''
    Class Label Triplet for Multi-Task
    '''
    # Build encoder
    input_modality1 = Input(S1_DIM)
    output_modality1 = encoder(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1)

    # Get instance of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)


    classifer = get_classifier3(EMB_DIM)
    classifier_output = classifer(S1_instance1)
    
    #new decoder for mask
    S1_decoder_inp = Input(EMB_DIM)
    decoder = mask_decoder(S1_decoder_inp)
    decoded_mask1 = decoder(S1_instance1)

    # Classifier + BBox model
    model = Model(S1_instance1_inp, [S1_instance1, decoded_mask1, classifier_output])
    
    model.compile(optimizer= 'Adam', loss= [mean_squared_loss, mean_squared_loss, mean_squared_loss])

    # print(model.summary())

    return model

def getMTL(S1_DIM, EMB_DIM=256) :
    '''
    MTL without triplet loss
    '''
    
    # Build encoder
    input_modality1 = Input(S1_DIM)
    output_modality1 = encoder(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1)

    # Get instance of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)


    classifer = get_classifier3(EMB_DIM)
    classifier_output = classifer(S1_instance1)
    
    #new decoder for mask
    S1_decoder_inp = Input(EMB_DIM)
    decoder = mask_decoder(S1_decoder_inp)
    decoded_mask1 = decoder(S1_instance1)


    # Classifier + BBox model
    model = Model(S1_instance1_inp, [decoded_mask1, classifier_output])
    
    model.compile(optimizer= 'Adam', loss= [mean_squared_loss, mean_squared_loss])

    # print(model.summary())

    return model

def getMTL_separate(EMB_DIM=32) :
    '''
    MTL without triplet loss
    '''

    # Build encoder
    input1 = Input(2*EMB_DIM)

    classifer = get_classifier3(2*EMB_DIM)
    classifier_output = classifer(input1)
    
    #new decoder for mask
    S1_decoder_inp = Input(2*EMB_DIM)
    decoder = mask_decoder(S1_decoder_inp)
    decoded_mask1 = decoder(input1)


    # Classifier + BBox model
    model = Model(input1, [decoded_mask1, classifier_output])
    
    model.compile(optimizer= 'Adam', loss= [mean_squared_loss, mean_squared_loss])

    # print(model.summary())

    return model

def getMTL_separate_learned(EMB_DIM=64) :
    '''
    MTL without triplet loss
    '''

    # Build encoder
    input1 = Input(EMB_DIM*2)
    
    out = Dense(EMB_DIM, kernel_initializer= tf.keras.initializers.GlorotNormal())(input1)

    classifer = get_classifier3(EMB_DIM)
    classifier_output = classifer(out)
    
    #new decoder for mask
    S1_decoder_inp = Input(EMB_DIM)
    decoder = mask_decoder(S1_decoder_inp)
    decoded_mask1 = decoder(out)


    # Classifier + BBox model
    model = Model(input1, [decoded_mask1, classifier_output])
    
    model.compile(optimizer= 'Adam', loss= [mean_squared_loss, mean_squared_loss])

    # print(model.summary())

    return model

# def getCOMMANetMTL_ModalitySelection(S1_DIM, S2_DIM, EMB_DIM=256) :
    
#     # Build encoder
#     input_modality1 = Input(S1_DIM)
#     output_modality1 = DenseNetDilation_HSI(input_modality1, EMB_DIM)
#     EncoderModality1 = Model(input_modality1, output_modality1)

#     # input_modality2 = Input(S2_DIM)    
#     # output_modality2 = DenseNetDilation_S2_DIM5(input_modality2, EMB_DIM)
#     # EncoderModality2 = Model(input_modality2, output_modality2)

#     # Get 3 instances of the Modality 1 encoder
#     S1_instance1_inp = Input(S1_DIM)
#     # S1_instance2_inp = Input(S1_DIM)
#     # S1_instance3_inp = Input(S1_DIM)

#     S1_instance1 = EncoderModality1(S1_instance1_inp)
#     # S1_instance2 = EncoderModality1(S1_instance2_inp)
#     # S1_instance3 = EncoderModality1(S1_instance3_inp)

#     #print(EncoderModality1.summary())

#     # # Get 1 instance of the Modality 2 encoder
#     # S2_instance1_inp = Input(S2_DIM)
#     # S2_instance1 = EncoderModality2(S2_instance1_inp)
    
#     #print(S1_instance1.shape, S2_instance1.shape)

#     # Concatenate the embeddings of 3 instances of modality1 and 1 instance of modality 2 for triplet loss
#     # classifier_fused_representation = modality_selection_attention2(S1_instance1, S2_instance1, EMB_DIM, identifier = 'classifier')
    
#     classifier_fused_representation = modality_selection_attention2(S1_instance1, EMB_DIM, identifier = 'classifier')
    
    
#     #concat_emb = concatenate([S1_instance1, S2_instance1], axis=1)
#     concat_emb = classifier_fused_representation
    
    
#     decoded = Dense(3*EMB_DIM, activation= 'relu')(concat_emb)
#     decoded = Reshape((1, 1, 3*EMB_DIM))(decoded)
#     decoded = UpSampling2D((3, 3))(decoded)
#     decoded = Conv2D(128, (3,3), activation= 'relu', padding= 'same')(decoded)
#     decoded = UpSampling2D((2, 2))(decoded)
#     decoded = Conv2D(64, (3,3), activation= 'relu', padding= 'same')(decoded)
#     decoded = Conv2D(32, (3,3), activation= 'relu', padding= 'same')(decoded)
#     decoded = UpSampling2D((2, 2))(decoded)
    
#     decoded = Conv2D(16, (2,2), activation= 'relu', padding= 'valid')(decoded)
#     decoded = Conv2D(10, (1,1), activation= 'softmax', padding= 'same')(decoded)
#     #decoded = SoftmaxWithMin()(decoded)    

#     box_fused_representation = modality_selection_attention2(S1_instance1, EMB_DIM, identifier= 'bbox')
#     #fused_representation = concatenate([S1_instance1, S2_instance1], axis= 1)

#     # classifer = get_classifier10(EMB_DIM*2)
#     # classifier_output = classifer(fused_representation)

#     box_predictor = corner_predictions(EMB_DIM)
#     #corners1 = box_predictor(fused_representation)
#     corners1 = box_predictor(box_fused_representation)
    
#     # pred_cropped = DynamicCropping2D()([decoded, corners1])
#     pred_flatten = Flatten()(decoded)
    
#     concat_corners_cropped = concatenate([corners1, pred_flatten], axis= -1)
    
#     # Classifier + BBox model
#     model = Model([S1_instance1_inp], [concat_emb, corners1, concat_corners_cropped])
    
#     model.compile(optimizer= 'Adam', loss= [mean_squared_loss, mean_squared_loss, conditional_patch_matching_loss])

#     # print(model.summary())

#     return model

def getBox(S1_DIM, EMB_DIM=256) :
    '''
    No triplet loss. Single task box.
    '''
    
    input_modality1 = Input(S1_DIM)    
    output_modality1 = encoder(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1, name='encoder')
    
    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)
    
    S1_decoder_inp = Input(EMB_DIM)
    decoder = mask_decoder(S1_decoder_inp)
    decoded_mask1 = decoder(S1_instance1)
    # corners1 = box_predictor(S1_instance1)
    
#     # Classifier + BBox model
#     model = Model(S1_instance1_inp, [corners1, decoded_mask])
    
#     model.compile(optimizer= 'Adam', loss= [iou_loss, 'binary_crossentropy'])
    
    # Classifier + BBox model
    model = Model(S1_instance1_inp, decoded_mask1)
    
    model.compile(optimizer= 'Adam', loss= 'binary_crossentropy')

    # print(model.summary())

    return model


def getTripletBox_mask_only(S1_DIM, EMB_DIM=256):
    '''
    Class label triplet loss. Single task box.
    '''
    input_modality1 = Input(S1_DIM)    
    output_modality1 = encoder(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1)

    # Input and encoding
    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)

    # Decoder for mask
    S1_decoder_inp = Input(EMB_DIM)
    decoder = mask_decoder(S1_decoder_inp)
    decoded_mask1 = decoder(S1_instance1)
    

    # Updated model
    model = Model(S1_instance1_inp, [S1_instance1, decoded_mask1])
    
    model.compile(optimizer='Adam', loss=[mean_squared_loss, 'binary_crossentropy'])

    return model


def getTripletCls(S1_DIM, EMB_DIM=256) :
    '''
    Class label triplet loss. Single task classification.
    '''

    input_modality1 = Input(S1_DIM)    
    output_modality1 = encoder(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1)
    #print(EncoderModality1.summary())

    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)

    S1_instance1 = EncoderModality1(S1_instance1_inp)

    
    # Classifier model
    cls_model = get_classifier3(EMB_DIM)
    
    cls_out = cls_model(S1_instance1)
    #print(cls_model.summary())
    
    model = Model(S1_instance1_inp, [S1_instance1, cls_out])
    
    model.compile(optimizer= 'Adam', loss= [mean_squared_loss, mean_squared_loss])

    # print(model.summary())

    return model



def getCls(S1_DIM, EMB_DIM=256) :
    '''
    No triplet loss. Single task classification.
    '''
    input_modality1 = Input(S1_DIM)    
    output_modality1 = encoder(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1)
    #print(EncoderModality1.summary())

    # Get 3 instances of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)

    
    # Classifier model
    cls_model = get_classifier3(EMB_DIM)
    
    cls_out = cls_model(S1_instance1)
    #print(cls_model.summary())
    
    model = Model(S1_instance1_inp, cls_out)
    
    model.compile(optimizer= 'Adam', loss= mean_squared_loss)

    # print(model.summary())

    return model

def get_continuous_box_euclidean(S1_DIM, EMB_DIM=64):
    
    input_modality1 = Input(S1_DIM)
    output_modality1 = encoder(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1, name = 'encoder')

    
    # Get 2 instance of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)
    
    S1_instance2_inp = Input(S1_DIM)
    S1_instance2 = EncoderModality1(S1_instance2_inp)
    
    # Decoder for mask
    S1_decoder_inp = Input(EMB_DIM)
    decoder = mask_decoder(S1_decoder_inp)

    decoded_mask1 = decoder(S1_instance1)
    decoded_mask2 = decoder(S1_instance2)
    
    # Compute Euclidean distance between embeddings
    euclidean_distance = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(S1_instance1 - S1_instance2), axis=-1), 1e-8))


    model = Model([S1_instance1_inp, S1_instance2_inp], [euclidean_distance, decoded_mask1, decoded_mask2])
    
    model.compile(optimizer = 'Adam', loss = ['mse', mean_squared_loss, mean_squared_loss])
    
    
    return model

def get_continuous_box_cosine(S1_DIM, EMB_DIM=64):
    
    input_modality1 = Input(S1_DIM)
    output_modality1 = encoder(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1, name = 'encoder')

    
    # Get 2 instance of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)
    
    S1_instance2_inp = Input(S1_DIM)
    S1_instance2 = EncoderModality1(S1_instance2_inp)
    
    # Decoder for mask
    S1_decoder_inp = Input(EMB_DIM)
    decoder = mask_decoder(S1_decoder_inp)

    decoded_mask1 = decoder(S1_instance1)
    decoded_mask2 = decoder(S1_instance2)
    
    dot_product = Dot(axes=-1, normalize=True)([S1_instance1, S1_instance2])
    
    cosine_distance = 1 - dot_product
    
    model = Model([S1_instance1_inp, S1_instance2_inp], [cosine_distance, decoded_mask1, decoded_mask2])
    
    model.compile(optimizer = 'Adam', loss = ['mse', mean_squared_loss, mean_squared_loss])
    
    
    return model


def get_continuous_box_cosine_online(S1_DIM, EMB_DIM=64):
    
    input_modality1 = Input(S1_DIM)
    output_modality1 = encoder(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1, name = 'encoder')

    
    # Get 2 instance of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)

    
    # Decoder for mask
    S1_decoder_inp = Input(EMB_DIM)
    decoder = mask_decoder(S1_decoder_inp)

    decoded_mask1 = decoder(S1_instance1)

    
    model = Model(S1_instance1_inp, [S1_instance1, decoded_mask1])
    
    model.compile(optimizer = 'Adam', loss = ['mse', 'mse'])
    
    
    return model



def get_continuous_box_features_cosine_online(S1_DIM, EMB_DIM=64):
    
    input_modality1 = Input(S1_DIM)
    output_modality1 = encoder(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1, name = 'encoder')

    
    # Get 2 instance of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)

    
    # Decoder for mask
    S1_decoder_inp = Input(EMB_DIM)
    decoder = mask_decoder(S1_decoder_inp)
    feature_predictor = feature_predictions(EMB_DIM)

    decoded_mask1 = decoder(S1_instance1)
    
    features1 = feature_predictor(S1_instance1)
    
    model = Model(S1_instance1_inp, [S1_instance1, decoded_mask1, features1])
    
    model.compile(optimizer = 'Adam', loss = ['mse', 'mse', 'mse'])
    
    
    return model


def getCOMMANetMTL(S1_DIM, S2_DIM, EMB_DIM=32) :
    
    # Build encoder
    input_modality1 = Input(S1_DIM)
    output_modality1 = encoder(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1)

    input_modality2 = Input(S2_DIM)    
    output_modality2 = encoder(input_modality2, EMB_DIM)
    EncoderModality2 = Model(input_modality2, output_modality2)

    # Get 1 instance of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)

    # Get 1 instance of the Modality 2 encoder
    S2_instance1_inp = Input(S2_DIM)
    S2_instance1 = EncoderModality2(S2_instance1_inp)
    
    # Concatenate the embeddings of instances of modality1 and modality 2 for triplet loss
    concat_emb = concatenate([S1_instance1, S2_instance1], axis=1)
    
    classifer = get_classifier3(EMB_DIM*2)
    classifier_output = classifer(concat_emb)

    box_predictor = corner_predictions(EMB_DIM*2)
    
    corners1 = box_predictor(concat_emb)
    
    #new decoder for mask
    decoded_mask = mask_decoder(concat_emb)

    # Classifier + BBox model
    model = Model([S1_instance1_inp, S2_instance1_inp], [concat_emb, corners1, decoded_mask, classifier_output])
    
    model.compile(optimizer= 'Adam', loss= [mean_squared_loss, mean_squared_loss, mean_squared_loss, mean_squared_loss])
    return model

def get_MM_MTL(S1_DIM, S2_DIM, EMB_DIM=32) :
    
    # Build encoder
    input_modality1 = Input(S1_DIM)
    output_modality1 = DenseNetDilation_S2_DIM5(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1)

    input_modality2 = Input(S2_DIM)    
    output_modality2 = DenseNetDilation_S2_DIM5(input_modality2, EMB_DIM)
    EncoderModality2 = Model(input_modality2, output_modality2)

    # Get 1 instance of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)

    # Get 1 instance of the Modality 2 encoder
    S2_instance1_inp = Input(S2_DIM)
    S2_instance1 = EncoderModality2(S2_instance1_inp)
    
    # Concatenate the embeddings of instances of modality1 and modality 2 for triplet loss
    concat_emb = concatenate([S1_instance1, S2_instance1], axis=1)
    
    classifer = get_classifier3(EMB_DIM*2)
    classifier_output = classifer(concat_emb)

    box_predictor = corner_predictions(EMB_DIM*2)
    
    corners1 = box_predictor(concat_emb)
    
    #new decoder for mask
    decoded_mask = mask_decoder(concat_emb)

    # Classifier + BBox model
    model = Model([S1_instance1_inp, S2_instance1_inp], [corners1, decoded_mask, classifier_output])
    
    model.compile(optimizer= 'Adam', loss= [mean_squared_loss, mean_squared_loss, mean_squared_loss])
    return model
    
def getCOMMANetBox(S1_DIM, S2_DIM, EMB_DIM=32) :
    
     # Build encoder
    input_modality1 = Input(S1_DIM)
    output_modality1 = DenseNetDilation_S2_DIM5(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1)

    input_modality2 = Input(S2_DIM)    
    output_modality2 = DenseNetDilation_S2_DIM5(input_modality2, EMB_DIM)
    EncoderModality2 = Model(input_modality2, output_modality2)

    # print(EncoderModality1.summary())

    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)

    # Get 1 instance of the Modality 2 encoder
    S2_instance1_inp = Input(S2_DIM)
    S2_instance1 = EncoderModality2(S2_instance1_inp)

    concat_emb = concatenate([S1_instance1, S2_instance1], axis= 1)

    box_predictor = corner_predictions(EMB_DIM*2)
    corners1 = box_predictor(concat_emb)
    
    #new decoder for mask
    decoded_mask = mask_decoder(concat_emb)
    
    # Classifier + BBox model
    model = Model([S1_instance1_inp, S2_instance1_inp], [concat_emb, corners1, decoded_mask])
    
    model.compile(optimizer= 'Adam', loss= [mean_squared_loss, iou_loss, mean_squared_loss])

    # print(model.summary())

    return model

def get_MM_Box(S1_DIM, S2_DIM, EMB_DIM=32) :
    
     # Build encoder
    input_modality1 = Input(S1_DIM)
    output_modality1 = DenseNetDilation_S2_DIM5(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1)

    input_modality2 = Input(S2_DIM)    
    output_modality2 = DenseNetDilation_S2_DIM5(input_modality2, EMB_DIM)
    EncoderModality2 = Model(input_modality2, output_modality2)

    # print(EncoderModality1.summary())

    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)

    # Get 1 instance of the Modality 2 encoder
    S2_instance1_inp = Input(S2_DIM)
    S2_instance1 = EncoderModality2(S2_instance1_inp)

    concat_emb = concatenate([S1_instance1, S2_instance1], axis= 1)

    box_predictor = corner_predictions(EMB_DIM*2)
    corners1 = box_predictor(concat_emb)
    
    #new decoder for mask
    decoded_mask = mask_decoder(concat_emb)
    
    # Classifier + BBox model
    model = Model([S1_instance1_inp, S2_instance1_inp], [corners1, decoded_mask])
    
    model.compile(optimizer= 'Adam', loss= [iou_loss, mean_squared_loss])

    # print(model.summary())

    return model

def getCOMMANetCls(S1_DIM, S2_DIM, EMB_DIM=32) :
    
    # Build encoder
    input_modality1 = Input(S1_DIM)
    output_modality1 = DenseNetDilation_S2_DIM5(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1)

    input_modality2 = Input(S2_DIM)    
    output_modality2 = DenseNetDilation_S2_DIM5(input_modality2, EMB_DIM)
    EncoderModality2 = Model(input_modality2, output_modality2)

    # Get 1 instance of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)

    # Get 1 instance of the Modality 2 encoder
    S2_instance1_inp = Input(S2_DIM)
    S2_instance1 = EncoderModality2(S2_instance1_inp)

    # Concatenate the embeddings of instances of modality1 and modality 2 for triplet loss
    concat_emb = concatenate([S1_instance1, S2_instance1], axis=1)
    
    # Classifier model
    cls_model = get_classifier3(EMB_DIM*2)
    
    cls_out = cls_model(concat_emb)
    #print(cls_model.summary())
    
    model = Model([S1_instance1_inp, S2_instance1_inp], [concat_emb, cls_out])
    
    model.compile(optimizer= 'Adam', loss= [mean_squared_loss, mean_squared_loss])

    return model

def get_MM_Cls(S1_DIM, S2_DIM, EMB_DIM=32) :
    
    # Build encoder
    input_modality1 = Input(S1_DIM)
    output_modality1 = DenseNetDilation_S2_DIM5(input_modality1, EMB_DIM)
    EncoderModality1 = Model(input_modality1, output_modality1)

    input_modality2 = Input(S2_DIM)    
    output_modality2 = DenseNetDilation_S2_DIM5(input_modality2, EMB_DIM)
    EncoderModality2 = Model(input_modality2, output_modality2)

    # Get 1 instance of the Modality 1 encoder
    S1_instance1_inp = Input(S1_DIM)
    S1_instance1 = EncoderModality1(S1_instance1_inp)

    # Get 1 instance of the Modality 2 encoder
    S2_instance1_inp = Input(S2_DIM)
    S2_instance1 = EncoderModality2(S2_instance1_inp)

    # Concatenate the embeddings of instances of modality1 and modality 2 for triplet loss
    concat_emb = concatenate([S1_instance1, S2_instance1], axis=1)
    
    # Classifier model
    cls_model = get_classifier3(EMB_DIM*2)
    
    cls_out = cls_model(concat_emb)
    #print(cls_model.summary())
    
    model = Model([S1_instance1_inp, S2_instance1_inp], cls_out)
    
    model.compile(optimizer= 'Adam', loss = mean_squared_loss)

    return model


