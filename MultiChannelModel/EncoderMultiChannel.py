import tensorflow as tf
from Logging import Logging

class EncoderMultiChannel(tf.keras.Model):

    def __init__(self, encoders,batch_size):
        super(EncoderMultiChannel, self).__init__()
        self.log=Logging()
        self.encoders=encoders
        self.batch_size=encoders[0].batch_sz
        self.enc_units=encoders[0].enc_units

    def call(self,x,hiddens):
        self.log.debug("Forward Pass through SimpleDecoder")
        outputs_fc=[]
        for i_channel,x_channel in enumerate(x):
            specific_encoder=self.encoders[i_channel]
            outputs_fc.append(specific_encoder(x_channel,hiddens[i_channel]))

        outputs_fc=tf.stack(outputs_fc,axis=1)
        outputs_fc=tf.reshape(outputs_fc,[self.batch_size,len(self.encoders),2,self.enc_units])
        outputs_fc=tf.transpose(outputs_fc,[0,2,1,3])
        outputs_fc=tf.reshape(outputs_fc,[-1])
        return outputs_fc

    def initialize_hidden_state(self):
        hiddens=[]
        for encoder in self.encoders:
            hiddens.append(tf.zeros((self.batch_size, self.enc_units)))
        return hiddens


