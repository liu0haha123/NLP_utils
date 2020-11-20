import numpy as np
import pickle
import torch
# exceptions
class MCBException(Exception):
    """base class for mcb exceptions"""

class SampleSizeException(MCBException):
    """raise when sample size of two features does not match"""

def _raise_sample_size_exception():
    raise SampleSizeException("size of samples does not match")


def _count_sketch_torch(d,h,s,v):
    """

    :param d:
    :param h: list shape = (v,) 随机生成的0到d-1
    :param s: list shape = (v,) 随机生成的1和-1
    :param v: 输出的向量
    :return:
    """
    cs_vector  = torch.zeros(d).float()

    for dim_num, _ in enumerate(v):
        cs_vector[h[dim_num]] += s[dim_num] * v[dim_num]

    return cs_vector


def _count_sketch(d, h, s, v):
    """
    count sketch
    argument:
        - h : list
            shape : (dimension of v, )
            a list of integers between 0 to (d-1) that are randomly placed
       - s : list
            shape : (dimension of v, )
            a list of -1 or 1 that is randomly placed
        - v : array
            the vector you want to transform by count sketch
    return:
        - cs_vector : ndarray
            v transformed by count sketch
    """

    cs_vector = np.zeros(d).astype("float64")

    for dim_num, _ in enumerate(v):
        cs_vector[h[dim_num]] += s[dim_num] * v[dim_num]

    return cs_vector


def _count_sketch_init(feature_dims, d):
    """
    for variables used in count sketch
    argument:
        - feature_dims : list
            dimensions of features
        - d : int
            output dimension
    return:
        - h : list
            shape : (feature_dim, d)
            list of integer between 0 to (d-1), that are randomly chosen
        - s : list
            shape : (feature_dim, d)
            list of -1 or 1, that are randomly chosen
    """
    h = [None, None]
    s = [None, None]

    for vec_num in range(2): # NOTE: we are only thinking about two modalities
        h[vec_num] = np.random.randint(0, d-1, size=(feature_dims[vec_num], ))
        s[vec_num] = (np.floor(np.random.uniform(0, 2, size=(feature_dims[vec_num], ))) * 2 - 1).astype("int64")

    return h, s

def _count_sketch_init_torch(feature_dims, d):
    """
    for variables used in count sketch
    argument:
        - feature_dims : list
            dimensions of features
        - d : int
            output dimension
    return:
        - h : list
            shape : (feature_dim, d)
            list of integer between 0 to (d-1), that are randomly chosen
        - s : list
            shape : (feature_dim, d)
            list of -1 or 1, that are randomly chosen
    """
    h = [None, None]
    s = [None, None]

    for vec_num in range(2): # NOTE: we are only thinking about two modalities
        h[vec_num] = torch.Tensor(torch.randint(0,d-1,size=(feature_dims[vec_num],)))
        s[vec_num] = (torch.floor(torch.Tensor(torch.rand(size=(feature_dims[vec_num],))).uniform_(0,2))*2-1).long()

    return h, s


def mcb(features1, features2, d: int = 16000, save=False, filename="mcb_feature.pickle"):
    """
    tranform two vectors of samples to one, using MCB
    argument:
        - features1 : ndarray
            shape : (sample size, dimension)
            sample size must match features2
        - features2 : ndarray
            shape : (sample size, dimension)
            sample size must match features2
        - d : integer (default:16000)
            output dimension
        - save : bool (default:False)
            True : saves variable using pickle
        - filename : str (default:"mcb_feature.pickle")
            filename for saving the variable
    return:
        - mcb_features : ndarray
            shape : (sample size, d)
            feature vectors, extracted from two feature vectors with mcb
    """
    # sample size check
    if features1.shape[0] != features2.shape[0]:
        _raise_sample_size_exception()

    # count sketch
    h, s = _count_sketch_init([features1.shape[1], features2.shape[1]], d)

    sketch_features1 = []
    sketch_features2 = []

    for v0, v1 in zip(features1, features2):
        sketch_features1.append(_count_sketch(d, h[0], s[0], v0))
        sketch_features2.append(_count_sketch(d, h[1], s[1], v1))

    # fft
    fft_features1 = []
    fft_features2 = []

    for v0, v1 in zip(sketch_features1, sketch_features2):
        fft_features1.append(np.fft.fft(v0))
        fft_features2.append(np.fft.fft(v1))

    # element-wise product
    ewp_features = np.multiply(fft_features1, fft_features2)

    # ifft
    ifft_features = np.fft.ifft(ewp_features)

    # cast to float (only taking the real part from complex matrix)
    mcb_features = np.real(ifft_features)

    # TODO : add element-wise sqrt and l2 normalization

    if save:
        try:
            with open(filename, "wb") as fout:
                pickle.dump(mcb_features, fout)
        except Exception as e:
            raise e

    return mcb_features


def mcb(features1, features2, d: int = 16000, save=False, filename="mcb_feature.pickle"):
    """
    tranform two vectors of samples to one, using MCB
    argument:
        - features1 : ndarray
            shape : (sample size, dimension)
            sample size must match features2
        - features2 : ndarray
            shape : (sample size, dimension)
            sample size must match features2
        - d : integer (default:16000)
            output dimension
        - save : bool (default:False)
            True : saves variable using pickle
        - filename : str (default:"mcb_feature.pickle")
            filename for saving the variable
    return:
        - mcb_features : ndarray
            shape : (sample size, d)
            feature vectors, extracted from two feature vectors with mcb
    """
    # sample size check
    if features1.shape[0] != features2.shape[0]:
        _raise_sample_size_exception()

    # count sketch
    h, s = _count_sketch_init([features1.shape[1], features2.shape[1]], d)

    sketch_features1 = []
    sketch_features2 = []

    for v0, v1 in zip(features1, features2):
        sketch_features1.append(_count_sketch(d, h[0], s[0], v0))
        sketch_features2.append(_count_sketch(d, h[1], s[1], v1))

    # fft
    fft_features1 = []
    fft_features2 = []

    for v0, v1 in zip(sketch_features1, sketch_features2):
        fft_features1.append(np.fft.fft(v0))
        fft_features2.append(np.fft.fft(v1))

    # element-wise product
    ewp_features = np.multiply(fft_features1, fft_features2)

    # ifft
    ifft_features = np.fft.ifft(ewp_features)

    # cast to float (only taking the real part from complex matrix)
    mcb_features = np.real(ifft_features)

    # TODO : add element-wise sqrt and l2 normalization

    if save:
        try:
            with open(filename, "wb") as fout:
                pickle.dump(mcb_features, fout)
        except Exception as e:
            raise e

    return mcb_features

def mcb_torch(features1, features2, d: int = 16000, save=False, filename="mcb_feature.pickle"):
    """
    tranform two vectors of samples to one, using MCB
    argument:
        - features1 : ndarray
            shape : (sample size, dimension)
            sample size must match features2
        - features2 : ndarray
            shape : (sample size, dimension)
            sample size must match features2
        - d : integer (default:16000)
            output dimension
        - save : bool (default:False)
            True : saves variable using pickle
        - filename : str (default:"mcb_feature.pickle")
            filename for saving the variable
    return:
        - mcb_features : ndarray
            shape : (sample size, d)
            feature vectors, extracted from two feature vectors with mcb
    """
    # sample size check
    if features1.shape[0] != features2.shape[0]:
        _raise_sample_size_exception()

    # count sketch
    h, s = _count_sketch_init([features1.shape[1], features2.shape[1]], d)

    sketch_features1 = []
    sketch_features2 = []

    for v0, v1 in zip(features1, features2):
        sketch_features1.append(_count_sketch(d, h[0], s[0], v0))
        sketch_features2.append(_count_sketch(d, h[1], s[1], v1))

    # fft
    fft_features1 = []
    fft_features2 = []

    for v0, v1 in zip(sketch_features1, sketch_features2):
        fft_features1.append(torch.fft(v0))
        fft_features2.append(torch.fft(v1))

    # element-wise product
    ewp_features = torch.matmul(fft_features1,fft_features2)
    #ewp_features = np.multiply(fft_features1, fft_features2)

    # ifft
    #ifft_features = np.fft.ifft(ewp_features)
    ifft_features = torch.ifft(ewp_features)

    # cast to float (only taking the real part from complex matrix)
    #mcb_features = np.real(ifft_features)
    mcb_features = torch.real(ifft_features)
    # TODO : add element-wise sqrt and l2 normalization

    if save:
        try:
            with open(filename, "wb") as fout:
                pickle.dump(mcb_features, fout)
        except Exception as e:
            raise e

    return mcb_features