from MultiChannelModel.DecoderMultiChannel import DecoderMultiChannel
from MultiChannelModel.SimpleEncoder import SimpleEncoder
from MultiChannelModel.EncoderMultiChannel import EncoderMultiChannel
from MultiChannelModel.VaeMultiChannel import VaeMultiChannel
import tensorflow as tf
from MultiChannelModel.SimpleEncoder import SimpleEncoder
from MultiChannelModel.multichannel_config import multichannel_config
from math import ceil
from Logging import Logging
class TrainMultiChannel:

    def __init__(self,dataset):
        self.dataset=dataset
        self.log = Logging()
        self.encoders=self.initialize_encoders()

    def initialize_encoders(self):
        encoders=[]
        embedding_size_factor=multichannel_config['embedding_size_factor']
        batch_size=multichannel_config['batch_size']
        latent_dim=multichannel_config['latent_dim']


        for i,channel in enumerate(self.dataset.channel_names):
            vocab_size=len(self.dataset.word_indexes[i].keys())
            gru_units=multichannel_config['gru_units']
            embedding_dim=max(5,ceil(embedding_size_factor*vocab_size))
            self.log.info('channel '+channel+' has '+str(embedding_dim)+'d embedding dims')
            enc_channel=SimpleEncoder(vocab_size,embedding_dim,gru_units,batch_size,latent_dim)
            encoders.append(enc_channel)

        self.log.info("encoders initialized")
        return encoders

    @staticmethod
    @tf.function
    def train_step(inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = encoder.trainable_variables + decoder.trainable_variables

        gradients = tape.gradient(loss, variables)

        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss