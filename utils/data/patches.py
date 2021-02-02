import tensorflow as tf
import numpy as np

def get_patched_dataset(train_images, 
                        train_labels, 
                        test_images, 
                        test_labels,
                        p_size = (1,4,4,1),
                        s_size = (1,4,4,1),
                        rate=(1,1,1,1),
                        padding='VALID',
                        augmentation=False):
    """
        This function returns the training and testing set in patch form 

        train_images (np.array) training images 
        train_labels (np.array) training labels 
        test_images  (np.array) test images 
        test_labels  (np.array) test labels 
        p_size (list) patch size
        s_size (list) stride size 
        rate (list) subsampling rate after getting patches 
        padding (str) ...
    """
    train_patches, train_labels_p = get_patches(train_images,
                                                train_labels,
                                                p_size,
                                                s_size,
                                                rate,
                                                padding)

    test_patches, test_labels_p = get_patches(test_images,
                                              test_labels,
                                              p_size,
                                              s_size,
                                              rate,
                                              padding)
    return train_patches,train_labels_p,test_patches,test_labels_p


def get_patches(x, 
                y, 
                p_size, 
                s_size, 
                rate,
                padding):
    """
        this function gets reformated image patches with the reshaped labels 

        x (np.array) images 
        y (np.array) labels 
        p_size (list) patch size
        s_size (list) stride size 
        rate (list) subsampling rate after getting patches 
    """
    x_out = tf.image.extract_patches(images=x,
                                     sizes=p_size,
                                     strides=s_size,
                                     rates=rate,
                                     padding=padding)

    x_patches = np.reshape(x_out,(x_out.shape[0]*x_out.shape[1]*x_out.shape[2],
                                  p_size[1],
                                  p_size[2],
                                  x.shape[-1]))


    y_patches= np.array([[label]*x_out.shape[1]*x_out.shape[2] for label in y]).flatten()

    return x_patches,y_patches
