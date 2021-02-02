import copy 
from sklearn.preprocessing import MinMaxScaler 

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
