""" @PETER HALLDESTAM, 2020"""
    
from tensorflow import transpose
from tensorflow.keras.backend import dot
from tensorflow.keras.backend import mean
from tensorflow.keras.backend import min as Kmin
from tensorflow.keras.backend import sum as Ksum

from utils.tensors import get_identity_tensor
from utils.tensors import get_permutation_tensor
from utils.tensors import get_shift_tensor

from loss_function.loss_functions import squared_error
from loss_function.loss_functions import absolute_error
from loss_function.loss_functions import huber_loss
from loss_function.loss_functions import pseudo_huber_loss

REGRESSION_LOSS = {'squared':   squared_error,
                   'absolute':  absolute_error,
                   'huber':     huber_loss,
                   'pseudo':    pseudo_huber_loss}

from loss_function.loss_functions import binary_cross_entropy
from loss_function.loss_functions import hinge_loss
from loss_function.loss_functions import modified_huber_loss

CLASSIFICATION_LOSS = {'cross_entropy': binary_cross_entropy,
                       'hinge':         hinge_loss,
                       'mod_huber':     modified_huber_loss}


class LossFunction(object):
    """
    The LossFunction object is used both in training and while matching the 
    combinations in the eventual prediction (see get_permutation_match in utils).
    To initiate a LossFunction only the maximum multiplicity of the data set is
    needed, but is customizable with additional keyword arguments.
    
    Args:
        max_mult :                  The maximum multiplicity of data set.
        
        permutation :               Returns the minumum loss amongst all
                                    combinations of network outputs with 
                                    corresponding labels.
                                    
        classification :            Used when training with an additional 
                                    classification node.
                                    
        regression_loss :           Loss function used in the momentum 
                                    regression with {'squared', 'absolute',
                                    'huber', 'pseudo'}.
                                    
        classification_loss :       Loss function used in the logistic 
                                    regression with {'cross_entropy', 'hinge',
                                    'mod_huber'}
                                    
        classification_weight :     weighting factor between the regression
                                    and classification loss                       
    """
    def __init__(self, max_mult,
                 permutation=True,
                 classification=False,
                 regression_loss='squared',
                 classification_loss='cross_entropy',
                 classification_weight=1):
                 
        self.max_mult = max_mult
        self.permutation = permutation
        self.classification = classification
        self.classification_weight = classification_weight
        
        self.identity_tensor = None
        self.permutation_tensor = None
        self.shift_tensor = None        
        
        if classification:
            self.shift_tensor = get_shift_tensor(max_mult)

        if permutation:
            if classification:    
                self.permutation_tensor = get_permutation_tensor(max_mult, m=4)
                self.identity_tensor = get_identity_tensor(max_mult, m=4)    
            else:
                self.permutation_tensor = get_permutation_tensor(max_mult, m=3)
                self.identity_tensor = get_identity_tensor(max_mult, m=3)    

        if regression_loss not in REGRESSION_LOSS:
            raise KeyError('regression_loss={} does not exist!'.format(regression_loss))
        if classification_loss not in CLASSIFICATION_LOSS:
            raise KeyError('classification_loss={} does not exist!'.format(classification_loss))

        self.rloss = REGRESSION_LOSS[regression_loss]
        self.closs = CLASSIFICATION_LOSS[classification_loss]
                    
    
    def get(self, train=True):
        if self.classification:
            return lambda y, y_: self.closs_function(y, y_, train)
        else:
            return lambda y, y_: self.rloss_function(y, y_, train)
        
        
    def rloss_function(self, y, y_, train):
        if self.permutation:    
            y = transpose(dot(y, self.identity_tensor), perm=[1,0,2])
            y_ = transpose(dot(y_, self.permutation_tensor), perm=[1,0,2])

        px, px_ = y[...,0::3], y_[...,0::3]
        py, py_ = y[...,1::3], y_[...,1::3]
        pz, pz_ = y[...,2::3], y_[...,2::3]
        loss = self.rloss(px, py, pz, px_, py_, pz_)
        
        if self.permutation:            
            if train:
                return mean(Kmin(Ksum(loss, axis=2), axis=0))
            else:
                return Ksum(loss, axis=2)
        else:
            return mean(Ksum(loss, axis=1))

    
    def closs_function(self, y, y_, train):
        y_ = dot(y, self.shift_tensor)
        if self.permutation:    
            y = transpose(dot(y, self.identity_tensor), perm=[1,0,2])
            y_ = transpose(dot(y_, self.permutation_tensor), perm=[1,0,2])

        b, b_ = y[...,0::4], y_[...,0::4] 
        px, px_ = y[...,1::4], y_[...,1::4]
        py, py_ = y[...,2::4], y_[...,2::4]
        pz, pz_ = y[...,3::4], y_[...,3::4]
        loss = self.rloss(px, py, pz, px_, py_, pz_) + self.classification*self.closs(b, b_)
        if self.permutation:            
            if train:
                return mean(Kmin(Ksum(loss, axis=2), axis=0))
            else:
                return Ksum(loss, axis=2)
        else:
            return mean(Ksum(loss, axis=1))
           