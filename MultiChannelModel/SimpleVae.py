import tensorflow as tf

class SimpleVae(tf.keras.Model):


  def __init__(self, latent_dim,encoder):
    super(SimpleVae, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = encoder
    # self.decoder=decoder

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean



  # def decode(self, z, apply_sigmoid=False):
  #   logits = self.decoder(z)
  #   if apply_sigmoid:
  #     probs = tf.sigmoid(logits)
  #     return probs
  #   return logits

  