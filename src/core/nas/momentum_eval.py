from tensorflow.keras.callbacks import Callback

class MomentumAugmentation(Callback):
    ''' Calculates the momentum's moving average of the parent model '''

    def __init__(self, monitor='val_sparse_categorical_accuracy'):
        ''' Initialize MA '''

        super(MomentumAugmentation, self).__init__()
        self.monitor = monitor

    
    def get_momentum(self, epoch, acc):
        ''' Calculates the momentums based on the given accuracies and epochs '''

        if epoch < 2:
            # momentum = acc at ε < 3
            return (acc, acc)

        delta_1 = acc - self.model.momentum[epoch - 1][0]
        delta_2 = self.model.momentum[epoch - 1][0] - self.model.momentum[epoch - 2][0]

        if delta_2 == 0.0:
            # avoid division by 0
            # if previous 2 accuracies are somehow exactly the same (very unlikely) => 0 momentum
            return (acc, 0.0)

        current_momentum = delta_1 / delta_2

        return (acc, current_momentum)

    
    def on_epoch_end(self, epoch, logs=None):
        ''' Called by Keras backend after each epoch during .fit() & .evaluate() '''

        logs = logs or {}

        if self.model is None:
            return

        if not hasattr(self.model, 'momentum'):
            self.model.momentum = {}

        if self.monitor in logs:
            val_acc = logs[self.monitor]
            
            self.model.momentum[epoch] = self.get_momentum(epoch, val_acc)
            
    