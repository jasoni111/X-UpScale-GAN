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
        # self.layers.append(  tf.keras.layers.LayerNormalization() )
        # 32,32,32   
        self.layers.append(  tf.keras.layers.Conv2D(64, (5,5), strides=(2,2), padding='same', kernel_initializer=init) )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        # self.layers.append(  tf.keras.layers.LayerNormalization() )

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
        # self.layers.append(  tf.keras.layers.LayerNormalization() )

        # 8,8,128
        self.layers.append(  tf.keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init) )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        # self.layers.append(  tf.keras.layers.LayerNormalization() )

        # 4,4,256
        self.layers.append(  tf.keras.layers.Conv2D(1024, (4,4), strides=(1,1), padding='valid', kernel_initializer=init) )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )

        # 1,1,1024       
        self.layers.append(  tf.keras.layers.Dense(1024, kernel_initializer=init) )
        self.layers.append(tf.keras.layers.Activation("linear" , dtype='float32') )
        # 1,1,1024  


    def call(self, inputs):
        print("~~~~~~~~shared")
        print(inputs.shape)
        for layer in self.layers:
            inputs = layer(inputs)
            print(inputs.shape)
        print("######share")
        return inputs

class VAE_encoder_shared_layers(tf.keras.layers.Layer):

    def __init__(self):
        super(VAE_encoder_shared_layers,self).__init__()
        pass

    def build(self, input_shape):
        self.layers = []
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        # 16,16,64
        self.layers.append(  tf.keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init))
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        # self.layers.append(  tf.keras.layers.LayerNormalization() )

        # 8,8,128
        self.layers.append(  tf.keras.layers.Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init) )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        # self.layers.append(  tf.keras.layers.LayerNormalization() )
        # 4,4,256
        self.layers.append(  tf.keras.layers.Conv2D(1024, (4,4), strides=(1,1), padding='valid', kernel_initializer=init) )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )

        self.z_mean_l  = tf.keras.layers.Dense(1024, kernel_initializer=init)
        self.z_log_var_l = tf.keras.layers.Dense(1024, kernel_initializer=init)
        self.to_32=tf.keras.layers.Activation("linear" , dtype='float32')
        # 1,1,1024       
        # self.layers.append(  tf.keras.layers.Conv2D(1024, (1,1), strides=(1,1), padding='valid', kernel_initializer=init) )
        # self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        # self.layers.append(tf.keras.layers.Activation("linear" , dtype='float32') )
        # 1,1,1024  


    def call(self, inputs):
        # print("~~~~~~~~shared")
        print(inputs.shape)
        for layer in self.layers:
            inputs = layer(inputs)
        
        z_mean = self.to_32(self.z_mean_l(inputs))
        z_log_var = self.to_32(self.z_log_var_l(inputs))
        normal = tf.random.normal(z_log_var.shape)
        print('z_mean',z_mean.shape)
        print('z_log_var',z_log_var.shape)
        print('normal',normal.shape)

        z = z_mean + tf.math.exp(0.5*z_log_var)*normal
        # print(inputs.shape)
        # print("~~~~~~~~")
        return z_mean,z_log_var,z

class decoder_shared_layers(tf.keras.layers.Layer):

    def __init__(self):
        super(decoder_shared_layers,self).__init__()
        pass

    def build(self, input_shape):
        self.layers = []
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        self.layers.append(  tf.keras.layers.Conv2DTranspose(512,kernel_size=(2,2),strides=(4,4), padding='same', kernel_initializer=init))
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
        self.layers.append(   lambda x:tf.nn.depth_to_space(x, 2) )
        # tf.nn.depth_to_space(x, scale)
        # 32 32 64
        self.layers.append(  tf.keras.layers.Conv2D(256, (3,3), padding='same', kernel_initializer=init) )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        # 32 32 256
        self.layers.append(  lambda x:tf.nn.depth_to_space(x, 2) )
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








