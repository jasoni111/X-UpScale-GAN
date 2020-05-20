# reference: https://stackoverflow.com/questions/56841166/how-to-implement-gradient-reversal-layer-in-tf-2-0
import tensorflow as tf

class C_dann(tf.keras.layers.Layer):
    def __init__(self):
        super(C_dann,self).__init__()
        pass

    def build(self, input_shape):
        self.layers = []

        @tf.custom_gradient
        def grad_reverse(x):
            y = tf.identity(x)
            def custom_grad(dy):
                return -dy
            return y, custom_grad

        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        self.layers.append(  grad_reverse  ) 
        self.layers.append(  tf.keras.layers.Conv2D(1024, (1,1) , (1,1) , padding='VALID')  )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        self.layers.append(  tf.keras.layers.Conv2D(1024, (1,1) , (1,1) , padding='VALID')  )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        self.layers.append(  tf.keras.layers.Conv2D(1024, (1,1) , (1,1) , padding='VALID')  )
        self.layers.append(  tf.keras.layers.LeakyReLU(alpha=0.2) )
        self.layers.append(  tf.keras.layers.Conv2D(1, (1,1) , (1,1) , padding='VALID')  )
        self.layers.append(tf.keras.layers.Activation("linear" , dtype='float32') )


    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs




