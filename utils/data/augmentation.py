import tensorflow as tf
import numpy as np

#TODO this wont work as it needs to do it on a per image basis!


def random_rotation(test_images, test_mask=None):
    """
        Applies random discrete rotation based augmentations to the test data and to their masks (if applicable)
        The discrete rotations are [0,90,180,270,360] degrees

        Parameters
        ----------
        test_images (np.array) 
        mask_images (optional, np.array) 
        
        Returns
        -------
        np.array, (optional) np.array

    """
    k = np.random.randomint(0,5)
    if np.random.randint(0,2):# probability 0.5 for up_down
        test_images = tf.image.rot90(test_images,k)
        if test_mask is not None:
            test_mask = tf.image.rot90(test_mask,k)
            return test_images, test_mask
        else:
            return test_images


def random_crop(test_images, crop_size, test_mask=None):
    """
        Applies random crop based augmentations to the test data and to their masks (if applicable)

        Parameters
        ----------
        test_images (np.array) 
        crop_size (list-like)
        mask_images (optional, np.array) 
        
        Returns
        -------
        np.array, (optional) np.array

        Raises:
        -------
        ValueError: If the shape of `image` is incompatible with the `test_mask` 

    """
    assert_op += assert(test_images.shape[1] > crop_size[0], ValueError, 
                        'X dimension of crop must be greater than X of crop')
    assert_op += assert(test_images.shape[2] > crop_size[1], ValueError, 
                        'Y dimension of crop must be greater than Y of crop')

    x_offset = np.random.randint(0, test_images.shape[1] -  crop_size[0] + 1)
    y_offset = np.random.randint(0, test_images.shape[2] -  crop_size[1] + 1)


    if np.random.randint(0,2):# probability 0.5 for up_down
        test_images = tf.image.crop_to_bounding_box(test_images,
                                                    x_offset,
                                                    y_offset,
                                                    crop_size[0],
                                                    crop_size[1])
        if test_mask is not None:
            test_mask = tf.image.crop_to_bounding_box(test_mask,
                                                        x_offset,
                                                        y_offset,
                                                        crop_size[0],
                                                        crop_size[1])
            return test_images, test_mask
        else:
            return test_images



