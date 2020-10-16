import tensorflow as tf


class EncoderMultiChannel(tf.keras.Model):

    def __init__(self, encoders):
        super(EncoderMultiChannel, self).__init__()
        self.encoders=encoders

    def call(self,x):
        outputs_fc=[]
        for i_channel,x_channel in enumerate(x):
            specific_encoder=self.encoders[i_channel]
            outputs_fc.append(specific_encoder(x_channel))
        return outputs_fc


