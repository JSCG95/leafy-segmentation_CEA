# -*- coding: utf-8 -*-
"""
IMPLEMENTATION UNET trained on Google Colab
Model: lettuce_unet_test.hdf5

"""

from simple_unet_model import simple_unet_model
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.utils import normalize
import os
import cv2
import numpy as np
from patchify import patchify, unpatchify


######################################################################################################################


def unet_segment(model_filename,img_filename):

    def get_model():
        return simple_unet_model(320, 320, 3)
    
    model = get_model()
    
    #model.load_weights('mitochondria_gpu_tf1.4.hdf5')
    model.load_weights(model_filename)
    #Apply a trained model on large image
    large_image = cv2.imread(img_filename)
    
    
    #This will split the image into small images of shape [3,3]
    #This will split the image into small images of shape [3,3]
    patches = patchify(large_image, (320, 320,3), step=320)  #Step=256 for 256 patches means no overlap
    print(patches.shape)
    
    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            print(i,j)
    ### HERE I CHANGED single patch input for single patch norm       
            single_patch = patches[i,j,:,:]
            single_patch_norm = normalize(np.array(single_patch), axis=1)
            
    
    #Predict and threshold for values above 0.5 probability
            single_patch_prediction = (model.predict(single_patch_norm)[0,:,:,0] > 0.5).astype(np.uint8)
            predicted_patches.append(single_patch_prediction)
    
    predicted_patches = np.array(predicted_patches)
    
    
    
    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 320,320))
    reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape[0:2])
    
    return reconstructed_image
#plt.imsave('/content/drive/MyDrive/Deep_Learning/Sreeni_UNET/masks_UNET/segm_t02_rgb_2.jpg', reconstructed_image, cmap='gray')

#plt.hist(reconstructed_image.flatten())  #Threshold everything above 0

#final_prediction = (reconstructed_image > 0.01).astype(np.uint8)
#plt.imshow(final_prediction)







