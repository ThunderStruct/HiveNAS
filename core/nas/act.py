from tensorflow.keras.callbacks import Callback

class TerminateOnThreshold(Callback):
    """
        Adaptive Cutoff Threshold (ACT)

        Keras Callback that terminates training if a given 'sparse_categorical_accuracy'
        dynamic threshold is not reached after n_epochs.
        The termination threshold has a logarithmic nature where the threshold
        increases by a decaying factor.
    """

    def __init__(self, 
                monitor='val_sparse_categorical_accuracy', 
                threshold_multiplier=0.25,
                diminishing_factor=0.25,
                n_classes = None):
        ''' Initialize threshold-based termination callback '''
        
        super(TerminateOnThreshold, self).__init__()

        self.monitor = monitor
        self.beta = threshold_multiplier
        self.zeta = diminishing_factor
        self.n_classes = n_classes

    def get_threshold(self, epoch):
        ''' Calculates val_acc termination threshold given the current epoch '''
        '''
            ΔThreshold = ß(1 - (1 / n))
            Threshold_base = (1 / n) + ΔThreshold = (1 / n) + ß(1 - (1 / n))
                                                  = (1 + ßn - ß) / n

            Range of Threshold_base = (1 / n, 1) ; horizontal asymptote at 1
            ΔThreshold decays as the number of classes decreases
            
            --------------

            To account for the expected increase in accuracy over the number
            of epochs ε, a growth_factor is added to the base threshold:

            growth_factor = (1 - Threshold_base) - (1 / (1 / 1-Threshold_base) + ζ(ε - 1))
            
            Threshold_adaptive = Threshold_base + growth_factor

            Range of growth_factor = [Threshold_base, 1) ; horizontal asymptote at 1
        '''

        baseline = 1.0 / self.n_classes     # baseline (random) val_acc
        complement_baseline = 1 - baseline
        delta_threshold = complement_baseline * self.beta
        base_threshold = baseline + delta_threshold
        ''' n_classes = 10, threshold_multiplier = 0.15 '''
        ''' yields .325 acc threshold for epoch 1 '''

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
        ''' Called by Keras backend after each epoch during .fit() & .evaluate() '''

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

