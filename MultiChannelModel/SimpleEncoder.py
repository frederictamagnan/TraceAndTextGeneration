import tensorflow as tf


class SimpleEncoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz,latent_dim):
    super(SimpleEncoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                       recurrent_initializer='glorot_uniform')
    self.fc=tf.keras.layers.Dense(latent_dim + latent_dim)

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    output_fc=self.fc(output)
    return output_fc

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))