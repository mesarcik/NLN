import tensorflow as tf
import numpy as np
from .defaults import sizes

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

def reconstruct(patches,args, labels=None):
    """
        Reconstructs the original training/testing images from the patches 
        NOTE: does not work on patches where stride!=patch_size or when data has been central cropped
        
        patches (np.array) array of patches generated by get_patches() 
        args (Namespace): the argumenets from cmd_args
        labels (np.array) array of labels of arranged according to patches

    """
    t = patches.transpose(0,2,1,3)
    n_patches = sizes[str(args.anomaly_class)]//args.patch_x
    recon = np.empty([patches.shape[0]//n_patches**2, args.patch_x*n_patches,args.patch_y*n_patches,patches.shape[-1]])

    start, counter, indx, b  = 0, 0, 0, []

    for i in range(n_patches, patches.shape[0], n_patches):
        b.append(np.reshape(np.stack(t[start:i,...],axis=0),(n_patches*args.patch_x,args.patch_x, patches.shape[-1])))
        start = i
        counter +=1
        if counter == n_patches:
            recon[indx,...] = np.hstack(b)
            indx+=1
            counter, b = 0, []

    if labels is not None:
        start, end, labels_recon = 0, args.patch_x**2, []

        for i in range(0, labels.shape[0], n_patches**2):
            if args.anomaly_class in labels[start:end]:
                labels_recon.append(str(args.anomaly_class))
            else:
                labels_recon.append('non_anomalous')

            start = end
            end += args.patch_x**2 
        return recon, np.array(labels_recon)

    else:
        return recon

