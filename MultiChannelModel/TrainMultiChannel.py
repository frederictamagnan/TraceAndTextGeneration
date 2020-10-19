from MultiChannelModel.SimpleDecoder import SimpleDecoder
from MultiChannelModel.SimpleEncoder import SimpleEncoder
from MultiChannelModel.EncoderMultiChannel import EncoderMultiChannel
from MultiChannelModel.SimpleVae import SimpleVae
import tensorflow as tf
from MultiChannelModel.SimpleEncoder import SimpleEncoder
from MultiChannelModel.multichannel_config import multichannel_config
from math import ceil
from Logging import Logging
from Dataset import Dataset
from general_config import general_config
import time
class TrainMultiChannel:

    def __init__(self,dataset):
        self.dataset=dataset
        self.log = Logging()


        self.embedding_size_factor = multichannel_config['embedding_size_factor']
        self.batch_size = general_config['batch_size']
        self.latent_dim = multichannel_config['latent_dim']
        self.encoders = self.initialize_encoders()
        self.encoderMultiChannel = EncoderMultiChannel(self.encoders, batch_size=self.batch_size)
        self.steps_per_epoch=self.dataset.len//self.batch_size
    def initialize_encoders(self):
        encoders=[]



        for i,channel in enumerate(self.dataset.channel_names):
            vocab_size=len(self.dataset.word_indexes[i].keys())
            gru_units=multichannel_config['gru_units']
            embedding_dim=max(5,ceil(self.embedding_size_factor*vocab_size))
            self.log.info('channel '+channel+' has '+str(embedding_dim)+'d embedding dims')
            enc_channel=SimpleEncoder(vocab_size,embedding_dim,gru_units,self.batch_size,self.latent_dim)
            encoders.append(enc_channel)

        self.log.info("encoders instanciated")
        return encoders


    @tf.function
    def train_step(self,inputs,encoder_hiddens):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoderMultiChannel(inputs, encoder_hiddens)

            dec_hidden = enc_hidden

        #     dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
        #
        #     # Teacher forcing - feeding the target as the next input
        #     for t in range(1, targ.shape[1]):
        #         # passing enc_output to the decoder
        #         predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
        #
        #         loss += loss_function(targ[:, t], predictions)
        #
        #         # using teacher forcing
        #         dec_input = tf.expand_dims(targ[:, t], 1)
        #
        # batch_loss = (loss / int(targ.shape[1]))
        #
        # variables = encoder.trainable_variables + decoder.trainable_variables
        #
        # gradients = tape.gradient(loss, variables)
        #
        # optimizer.apply_gradients(zip(gradients, variables))
        #
        # return batch_loss
    def train(self):

        for epoch in range(multichannel_config['epochs']):
            start = time.time()

            enc_hidden = self.encoderMultiChannel.initialize_hidden_state()
            total_loss = 0

            for (batch, input_from_all_channels) in enumerate(self.dataset.batched_dataset.take(self.steps_per_epoch)):
                batch_loss = self.train_step(input_from_all_channels, enc_hidden)
                total_loss += batch_loss


if __name__=='__main__':
    dataset = Dataset(limit_rows=1000)
    tmc=TrainMultiChannel(dataset=dataset)
    tmc.train()