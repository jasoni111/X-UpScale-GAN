import tensorflow as tf

from suppixel import SubpixelConv2D


class encoder_seperate_layers(tf.keras.layers.Layer):
    def __init__(self):
        super(encoder_seperate_layers,self).__init__()
        pass

    def build(self, input_shape):
        self.layers = []

        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        # 64,64,3
        self.layers.append(  tf.keras.layers.Conv2D(32, (5,5), strides=(2,2), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        # 32,32,32   
        self.layers.append(  tf.keras.layers.Conv2D(64, (5,5), strides=(2,2), padding='same', kernel_initializer=init) )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        # 16,16,64

    def call(self, inputs):
        # print(inputs.shape)
        for layer in self.layers:
            inputs = layer(inputs)
            # print(inputs.shape)
        return inputs

class encoder_shared_layers(tf.keras.layers.Layer):

    def __init__(self):
        super(encoder_shared_layers,self).__init__()
        pass

    def build(self, input_shape):
        self.layers = []
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        # 16,16,64
        self.layers.append(  tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        # 8,8,128
        self.layers.append(  tf.keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init) )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        # 4,4,256
        self.layers.append(  tf.keras.layers.Conv2D(1024, (4,4), strides=(1,1), padding='valid', kernel_initializer=init) )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        # 1,1,1024       
        self.layers.append(  tf.keras.layers.Conv2D(1024, (1,1), strides=(1,1), padding='valid', kernel_initializer=init) )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        self.layers.append(tf.keras.layers.Activation("linear" , dtype='float32') )
        # 1,1,1024  


    def call(self, inputs):
        # print("~~~~~~~~shared")
        print(inputs.shape)
        for layer in self.layers:
            inputs = layer(inputs)
            # print(inputs.shape)
        # print("~~~~~~~~")
        return inputs

class decoder_shared_layers(tf.keras.layers.Layer):

    def __init__(self):
        super(decoder_shared_layers,self).__init__()
        pass

    def build(self, input_shape):
        self.layers = []
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        self.layers.append(  tf.keras.layers.Conv2DTranspose(512,kernel_size=(1,1),strides=(4,4), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )

        self.layers.append(  tf.keras.layers.Conv2DTranspose(256,kernel_size=(4,4),strides=(2,2), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )

    def call(self, inputs):
        print("decoder_shared_layers~~~~~~~~~~~~~~~~")
        for layer in self.layers:
            inputs = layer(inputs)
            print(inputs.shape)
        print("###############")
        return inputs

class decoder_seperate_layers(tf.keras.layers.Layer):

    def __init__(self):
        super(decoder_seperate_layers,self).__init__()
        pass

    def build(self, input_shape):
        self.layers = []
        # 8 8 256
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        self.layers.append(  tf.keras.layers.Conv2DTranspose(256,kernel_size=(4,4),strides=(2,2), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        # 16 16 256
        self.layers.append(  SubpixelConv2D(input_shape,2) )
        # 32 32 64
        self.layers.append(  tf.keras.layers.Conv2D(256, (3,3), padding='same', kernel_initializer=init) )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        # 32 32 256
        self.layers.append(  SubpixelConv2D(input_shape,2) )
        self.layers.append(  tf.keras.layers.Conv2D(256, (3,3), padding='same', kernel_initializer=init) )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )

        self.layers.append(  tf.keras.layers.Conv2D(3, (1,1), padding='same', kernel_initializer=init) )
        self.layers.append(tf.keras.layers.Activation("tanh" , dtype='float32') )

    def call(self, inputs):
        print("~~~decode_seperate",inputs.shape)
        for layer in self.layers:
            inputs = layer(inputs)
            print(inputs.shape)
        print("###")
        return inputs







