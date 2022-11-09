"""Calculates a cutoff performance threshold, below which a model stops training
"""

from tensorflow.keras.callbacks import Callback

class TerminateOnThreshold(Callback):
    '''Adaptive Cutoff Threshold (ACT)
    
    Keras Callback that terminates training if a given :code:`val_sparse_categorical_accuracy`
    dynamic threshold is not reached after :math:`\\epsilon` epochs.
    The termination threshold has a logarithmic nature where the threshold
    increases by a decaying factor.
    
    Attributes:
        beta (float): threshold coefficient (captures the leniency of the calculated threshold)
        monitor (str): the optimizer metric type to monitor and calculate ACT on
        n_classes (int): number of classes
        zeta (float): diminishing factor; a positive, non-zero factor that controls how steeply the function horizontally asymptotes at :math:`y = 1.0` (i.e 100% accuracy)
    '''

    def __init__(self, 
                monitor='val_sparse_categorical_accuracy', 
                threshold_multiplier=0.25,
                diminishing_factor=0.25,
                n_classes = None):
        '''Initialize threshold-based termination callback 
        
        Args:
            monitor (str, optional): the optimizer metric type to monitor and calculate ACT on
            threshold_multiplier (float, optional): threshold coefficient (captures the leniency of the calculated threshold)
            diminishing_factor (float, optional): iminishing factor; a positive, non-zero factor that controls how steeply the function horizontally asymptotes at y = 1.0 (i.e 100% accuracy)
            n_classes (None, optional): number of classes / output neurons
        '''
        
        super(TerminateOnThreshold, self).__init__()

        self.monitor = monitor
        self.beta = threshold_multiplier
        self.zeta = diminishing_factor
        self.n_classes = n_classes


    def get_threshold(self, epoch):
        '''Calculates the termination threshold given the current epoch 
        
            .. math:: ΔThreshold = ß(1 - \\frac{1}{n})

            .. math:: 

                Threshold_{base} = \\frac{1}{n} + ΔThreshold &= \\frac{1}{n} + ß(1 - \\frac{1}{n}) \\
                                                              
                                                        &= \\frac{(1 + ßn - ß)}{n}
            
            .. math:: Threshold_{base} \\Rightarrow (\\frac{1}{n},\\: 1) \\; ; \\text{horizontal asymptote at} \\; Threshold_{base} = 1

            :math:`ΔThreshold` decays as the number of classes decreases
            
            --------------

            To account for the expected increase in accuracy over the number
            of epochs :math:`ε` , a growth factor :math:`g` is added to the base threshold:

            .. math:: g = (1 - Threshold_{base}) - \\frac{1}{\\frac{1}{1-Threshold_{base}} + ζ(ε - 1)}
            
            .. math:: Threshold_{adaptive} = Threshold_{base} + g

            .. math:: g \\Rightarrow [Threshold_{base}, 1) \\; ; \\text{horizontal asymptote at} \\; g = 1

        Args:
            epoch (int): current epoch
        
        Returns:
            float: calculated cutoff threshold
        '''

        baseline = 1.0 / self.n_classes     # baseline (random) val_acc
        complement_baseline = 1 - baseline
        delta_threshold = complement_baseline * self.beta
        base_threshold = baseline + delta_threshold
        ''' 
        n_classes = 10, threshold_multiplier = 0.15
        yields .325 acc threshold for epoch 1 
        '''

        # epoch-based decaying increase in val_acc threshold
        complement_threshold = 1 - base_threshold    # the increase factor's upper limit
        growth_denom = (1.0 / complement_threshold) + self.zeta * (epoch - 1)
        growth_factor = complement_threshold - 1.0 / growth_denom

        calculated_threshold = base_threshold + growth_factor
        ''' 
            Same settings as before yields:
            epoch 1 = .325000
            epoch 2 = .422459, 
            epoch 3 = .495327,
            epoch 4 = .551867,
            epoch 5 = .597014
        '''
        
        return calculated_threshold


    def on_epoch_end(self, epoch, logs=None):
        '''Called by Keras backend after each epoch during :code:`.fit()` & :code:`.evaluate()` 
        
        Args:
            epoch (int): current epoch
            logs (None, optional): contains all the monitors (or metrics) used by the optimizer in the training and evaluation contexts
        '''

        logs = logs or {}

        if self.model is None:
            return

        if self.n_classes is None:
            self.n_classes = self.model.layers[-1].output_shape[1]

        threshold = self.get_threshold(epoch + 1)

        if self.monitor in logs:
            val_acc = logs[self.monitor]
            if val_acc < threshold:
                # threshold not met, terminate
                print(f'\nEpoch {(epoch + 1)}: Accuracy ({val_acc}) has not reached the baseline threshold {threshold}, terminating training... \n')
                self.model.stop_training = True

