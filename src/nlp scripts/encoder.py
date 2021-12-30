"""Encoder Architecture"""
"""Better explanation is done at the documentation section of this project"""
"""score = FC(tanh(FC(EO) + FC(H)))

attention weights = softmax(score, axis = 1).

context vector = sum(attention weights * EO, axis = 1).

embedding output = The input to the decoder X is passed through an embedding layer.

merged vector = concat(embedding output, context vector)

This merged vector is then given to the GRU
"""

"""We may do this from scratch using , but for now let's use the existing tensorflow library because the code
provided by tensorflow is more optimized and clean"""

from preprocess import*
from create_dataset import*
from tokenspad import*
from train_test_creation import*

class Encoder(tf.keras.Model):
    """init method for constructor"""
    def __init__(self, input_vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()

        self.batch_size = batch_size
        self.enc_units = enc_units

        # Embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(input_vocab_size,
                                                   embedding_dim)

        # The GRU RNN layer preprocess those vector sequentially

        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       # return the sequence and state,
                                       return_sequences = True,
                                       return_state = True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):

        return tf.zeros((self.batch_size, self.enc_units))
         # object of the Encoder class 
encoder = Encoder(vocab_sent_size, embedding_dim, units, BATCH_SIZE)

# simple input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

""" Print the shapes  """
#print(sample_hidden.shape)
#print(sample_output.shape)

























