import tensorflow as tf
from tensorflow.keras import layers

kernel_init = tf.keras.initializers.GlorotNormal()

class Conv2D(layers.Conv2D):
    def __init__(self, kernel_size = 3, padding = 'same', **kwargs):
        super(Conv2D, self).__init__(
            kernel_size = kernel_size,
            padding = padding,
            kernel_initializer = kernel_init,
            bias_initializer = tf.keras.initializers.Zeros(),
            **kwargs
        )

class Conv2DBlock(layers.Layer):
    def __init__(self, filters, batchnorm = True, activate = True, **kwargs):
        super(Conv2DBlock, self).__init__()

        self.conv = Conv2D(filters = filters, **kwargs)
        # I forgot to set use_bias to False...
        # you please set it to False if you want to save some parameters
        # because batchnorm right after conv layer is gonna make the biases obsolete
        self.batchnorm = layers.BatchNormalization() if batchnorm else None
        self.activate = layers.PReLU(shared_axes = [1, 2]) if activate else None
        
    def call(self, inputs):
        x = self.conv(inputs)
        if self.batchnorm:
            x = self.batchnorm(x)
        if self.activate:
            x = self.activate(x)
        return x

class PixelShuffleUpSampling(layers.Layer):
    def __init__(self, filters, scale, **kwargs):
        super(PixelShuffleUpSampling, self).__init__(**kwargs)

        self.conv1 = Conv2DBlock(filters = filters, batchnorm = False, activate = False)
        self.upsample = layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale))
        self.prelu = layers.PReLU(shared_axes = [1, 2])
    
    def call(self, x):
        x = self.conv1(x)
        x = self.upsample(x)
        x = self.prelu(x)
        return x

class ResidualDenseBlock(layers.Layer):
    def __init__(self, filters = 64):
        super(ResidualDenseBlock, self).__init__()

        self.conv1 = Conv2DBlock(filters = filters // 2)
        self.conv2 = Conv2DBlock(filters = filters // 2)
        self.conv3 = Conv2DBlock(filters = filters, activate = False)

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(tf.concat([x1, inputs], 3))
        outputs = self.conv3(tf.concat([x2, x1], 3))
        
        return outputs + inputs

class RRDBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super(RRDBlock, self).__init__(**kwargs)

        self.rdb_1 = ResidualDenseBlock(filters)
        self.rdb_2 = ResidualDenseBlock(filters)
        self.rdb_3 = ResidualDenseBlock(filters)

        self.rrdb_inputs_scales = tf.Variable(
            tf.constant(value = 1.0, dtype = tf.float32, shape = [1, 1, 1, filters]),
            name = f'{self.name}_rrdb_inputs_scales',
            trainable = True
        )
        self.rrdb_outputs_scales = tf.Variable(
            tf.constant(value = 0.5, dtype = tf.float32, shape = [1, 1, 1, filters]),
            name = f'{self.name}_rrdb_outputs_scales',
            trainable = True
        )

    def call(self, inputs):
        x1 = self.rdb_1(inputs)
        x2 = self.rdb_2(x1)
        outputs = self.rdb_3(x2)

        return (self.rrdb_inputs_scales * inputs) + (self.rrdb_outputs_scales * outputs)