"""Cell-based operations definition
"""

from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.layers import SeparableConv2D, Dense, Dropout, Activation
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, ReLU
from tensorflow.keras.layers import concatenate, GlobalAveragePooling2D

class OperationCells:
    '''Wrapper for all Search Space cells,
    each group of operations (cell) is defined as a function that \
    returns a function. This is to conform to Keras' functional models' design pattern.
    
    i.e: :code:`ConvBnReLU(3, 32, 1)(input_layer)`. This ensures that cells defined here \
    and primitive Keras layers are interchangeable and behave the same way in the \
    remainder of the code.
    '''
    
    ''' Fixed parameters (empirically deduced) '''
    __BN_AXIS = 3
    __BN_MOMENTUM = 0.9
    __BN_EPSILON = 1e-5

    __CONV_L2_REG = 0.0001

    @staticmethod
    def ConvBnReLU(conv_kern=3, conv_filt=32, conv_stride=1):
        '''Cell consisting of :code:`Conv2D` | :code:`BatchNormalization` | :code:`ReLU`

        Args:
            conv_kern (int, optional): the convolution's kernel size
            conv_filter (int, optional): the number of filters in the conv layer
            conv_stride (int, optional): the convolution's stride value

        Returns:
            func: returns a functional template that accepts the input layer
        '''

        def retfunc(input_layer):
            '''Conforming to Keras' functional model design pattern

            Args:
                input_layer (:class:`tensorflow.keras.layers.Layer`): the functional input layer to the cell

            Returns:
                :class:`tensorflow.keras.layers.Layer`: output functional layer group
            '''

            conv_layer = Conv2D(conv_filt, 
                                kernel_size=conv_kern, 
                                padding='same', 
                                kernel_regularizer=l2(OperationCells.__CONV_L2_REG))(input_layer)
            bn_layer = BatchNormalization(axis=OperationCells.__BN_AXIS,
                                        momentum=OperationCells.__BN_MOMENTUM,
                                        epsilon=OperationCells.__BN_EPSILON)(conv_layer)
            relu_layer = Activation('relu')(bn_layer)

            return relu_layer

        return retfunc


    @staticmethod
    def ConvBnReLUAvgPool(conv_kern=3, conv_filt=32, conv_stride=1,
                          pool_size=2, pool_stride=2):
        '''Cell consisting of :code:`Conv2D` | :code:`BatchNormalization` | :code:`ReLU` | :code:`AveragePooling2D`

        Args:
            conv_kern (int, optional): the convolution's kernel size
            conv_filter (int, optional): the number of filters in the conv layer
            conv_stride (int, optional): the convolution's stride value
            pool_size (int, optional): the average pooling kernel size
            pool_stride (int, optional): the average pooling's stride value

        Returns:
            func: returns a functional template that accepts the input layer
        '''
        
        def retfunc(input_layer):
            '''Conforming to Keras' functional model design pattern

            Args:
                input_layer (:class:`tensorflow.keras.layers.Layer`): the functional input layer to the cell

            Returns:
                :class:`tensorflow.keras.layers.Layer`: output functional layer group
            '''

            conv_bn_relu = OperationCells.ConvBnReLU(conv_kern, conv_filt, conv_stride)(input_layer)
            pooling_layer = AveragePooling2D(pool_size=pool_size, strides=pool_stride, padding='same')(conv_bn_relu)

            return pooling_layer

        return retfunc

    
    @staticmethod
    def ResidualConvBnReLU(conv_kern=3, conv_filt=32, conv_stride=1, 
                           reg_identity=True, reg_filters=32,
                           block_count=2):
        '''Residual cell consisting of :code:`block_count` :math:`\\times` (:code:`Conv2D` | :code:`BatchNormalization` | :code:`ReLU`)

        Args:
            conv_kern (int, optional): the convolution's kernel size
            conv_filter (int, optional): the number of filters in the conv layer
            conv_stride (int, optional): the convolution's stride value
            reg_identity (bool, optional): specifies whether an initial \
            regularizing identity layer is added
            block_count (int, optional): number of (:code:`Conv2D` | :code:`BatchNormalization` | :code:`ReLU`) \
            cells are used in the residual block

        Returns:
            func: returns a functional template that accepts the input layer
        '''
        
        def retfunc(input_layer):
            '''Conforming to Keras' functional model design pattern

            Args:
                input_layer (:class:`tensorflow.keras.layers.Layer`): the functional input layer to the cell

            Returns:
                :class:`tensorflow.keras.layers.Layer`: output functional layer group
            '''

            in_layer = input_layer

            if reg_identity:
                # identity layer (Conv2D 1x1)
                in_layer = Conv2D(filters=reg_filters, kernel_size=1, strides=1, 
                                  padding='same', 
                                  kernel_regularizer=l2(OperationCells.__CONV_L2_REG))(input_layer)

            block = OperationCells.ConvBnReLU(conv_kern, conv_filt, conv_stride)(in_layer)
            for _ in range(block_count-1):
                block = OperationCells.ConvBnReLU(conv_kern, conv_filt, conv_stride)(block)

            return concatenate([input_layer, block])

        return retfunc

