import tensorflow as tf
from Logging import Logging

class SimpleEncoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz,latent_dim):
    super(SimpleEncoder, self).__init__()
    self.log = Logging()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.latent_dim=latent_dim
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                       recurrent_initializer='glorot_uniform')
    self.fc=tf.keras.layers.Dense(2*self.latent_dim)

  def call(self, x, hidden):
    self.log.debug("Forward Pass through SimpleEncoder")
    self.log.debug("Input shape of Simple Encoder is :\n"+"X :"+str(x.shape)+"\n hidden :"+str(hidden.shape) )
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    output_fc=self.fc(state)
    self.log.debug("Output shape of Simple Encoder is :\n"+str(output_fc))
    return output_fc

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))