""" @PETER HALLDESTAM, 2020
    
    All tested loss functions.
    
"""
from tensorflow.keras.backend import square
from tensorflow.keras.backend import sqrt
from tensorflow.keras.backend import log


DELTA = 1

## REGRESSION

def squared_error(px, py, pz, px_, py_, pz_):
    """
    Standard loss function for regression. Also known as MSE

    """
    return square(px - px_) + square(py - py_) + square(pz - pz_)


def absolute_error(px, py, pz, px_, py_, pz_):
    """
    Pretty much the same as square_error with an additional square root
    
    """
    return sqrt(squared_error(px, py, pz, px_, py_, pz_))


## OBS. NOT USING KERAS STUFF...
def huber_loss(px, py, pz, px_, py_, pz_):
    """
    Sort of a combination of squared_error and absolute_error. It is less 
    sensitive to outliers in data than the squared_error loss.
    
    """
    error = squared_error(px, py, pz, px_, py_, pz_)
    if error <= DELTA**2:
        return .5*error
    else:
        return DELTA*(sqrt(error) - .5*DELTA)
    
    
def pseudo_huber_loss(px, py, pz, px_, py_, pz_):
    """
    Used as a smooth approximation of the Huber loss function
    
    """
    return DELTA**2*(sqrt(1 + squared_error(px, py, pz, px_, py_, pz_)/DELTA**2) - 1)
    


## BINARY CLASSIFICATION

def binary_cross_entropy(b, b_):
    """
    Standard loss function for logistic regression. Expects values b_ in {0, 1}

    """
    return - b_*log(b) - (1-b_)*log(1-b) 


def hinge_loss(b, b_):
    """
    This one not only penalizes the wrong predictions but also the right 
    predictions that are not confident. (OBS!!) Expects values b_ in {-1, 1}
    
    """
    return max(0, 1 - b*b_) 


## OBS. NOT USING KERAS STUFF...
def modified_huber_loss(b, b_):
    bb_ = b*b_
    if bb_ >= -1:
        return hinge_loss(b, b_)
    else:
        return - 4*bb_