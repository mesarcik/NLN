import tensorflow as tf
import numpy as np

def get_patched_dataset(train_images, 
                        train_labels, 
                        test_images, 
                        test_labels,
                        test_masks=None,
                        p_size=(1,4,4,1), 
                        s_size=(1,4,4,1),
                        rate=(1,1,1,1),
                        padding='VALID',
                        central_crop=False):
    """
        This function returns the training and testing set in patch form. 
        Note: If test_masks is specified then labels generated from the masks, 
              otherwise labels are based off the original labels. 

        train_images (np.array) training images 
        train_labels (np.array) training labels 
        test_images  (np.array) test images 
        test_labels  (np.array) test labels 
        p_size (list) patch size
        s_size (list) stride size 
        rate (list) subsampling rate after getting patches 
        padding (str) ...
    """
    if central_crop:
        train_images = tf.image.central_crop(train_images,0.7).numpy()
        test_images =  tf.image.central_crop(test_images,0.7).numpy()
        if test_masks is not None: 
            test_masks = np.expand_dims(test_masks,axis=-1)
            test_masks =   tf.image.central_crop(test_masks,0.7).numpy()[...,0]

    train_patches, train_labels_p = get_patches(train_images,
                                                train_labels,
                                                p_size,
                                                s_size,
                                                rate,
                                                padding)

    if test_masks is not None:
        # Test labels based off the results in the mask patches
        test_masks = np.expand_dims(test_masks,axis=-1)
        test_patches, test_labels_bool = get_patches(test_images,
                                                     test_masks,
                                                     p_size,
                                                     s_size,
                                                     rate,
                                                     padding)


        lbls = np.unique(test_labels)
        test_labels_p = np.empty(len(test_labels_bool), dtype='<U13')


        for i,label in enumerate(test_labels_bool):
            if label: test_labels_p[i] = lbls[0]
            else: test_labels_p[i] = lbls[1]


        test_masks_patches, _ = get_patches(test_masks,
                                           test_labels,
                                           p_size,
                                           s_size,
                                           rate,
                                           padding)

        return train_patches,train_labels_p,test_patches,test_labels_p,test_masks_patches[...,0]

    else: #test_masks is None 
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
        This function gets reformated image patches with the reshaped labels
        Note: If y is the mask, then we perform logic to get labels from patches 

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
    if x.shape[:3] == y.shape[:3]:
        # mask is supplied as y
        y_out = tf.image.extract_patches(images=y,
                                         sizes=p_size,
                                         strides=s_size,
                                         rates=rate,
                                         padding=padding)

        y_patches = np.reshape(y_out,(y_out.shape[0]*y_out.shape[1]*y_out.shape[2],
                                      p_size[1],
                                      p_size[2],
                                      y.shape[-1]))
        y_labels = np.array([int(np.sum(patch)>10) for patch in y_patches])

        # For each patch is there at least 1 pixel of anomalies

    else:
        y_labels = np.array([[label]*x_out.shape[1]*x_out.shape[2] for label in y]).flatten()

    return x_patches,y_labels
