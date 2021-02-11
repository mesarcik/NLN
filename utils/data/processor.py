import copy 
from sklearn.preprocessing import MinMaxScaler 
from skimage import transform 

def process(data):
    """
        Scales data between 0 and 1 on a per image basis

        data (np.array) is either the test or training data

    """
    output = copy.deepcopy(data).astype('float32')
    for i,image in enumerate(data):
        x,y,z = image.shape
        output[i,...] = MinMaxScaler(feature_range=(0,1)
                                      ).fit_transform(image.reshape([x*y,z])).reshape([x,y,z])
    return output

def resize(data, dim):
    """
        Overloaded method for resizing input image

        data (np.array)  3D matrix containing image data (#images,X,Y,RGB/G)
        dim  (tuple) Tuple with 4 entires (#images, X, Y, RGB)

    """
    return transform.resize(data,(data.shape[0], dim[0], dim[1], dim[2]), anti_aliasing=False)

