"""Neural Machine Translation communication system.
load the files"""

from preprocess import*
from create_dataset import*
from tokenspad import*
from train_test_creation import*
from encoder import*
from encodeFunction import*

"""Let's build decode class to decode the input sequences
of the messages"""


"""The decoder's job is to generate predictions for the next output token.

The decoder receives the complete encoder output.
It uses an RNN to keep track of what it has generated so far.
It uses its RNN output as the query to the attention over the encoder's output, producing the context vector.
It combines the RNN output and the context vector using Equation 3 (below) to generate the "attention vector".
It generates logit predictions for the next token based on the "attention vector".

comment source : Tensorflow docs"""

"""Here is the Decoder class and its initializer.
The initializer creates all the necessary layers."""


class Decoder(tf.keras.layers.Layer):

    """Initializer for different layers"""

    def __init__(self, input_vocab_size, embedding_dim, dec_units, batch_size):
        super(Decoder, self).__init__()

        self.batch_size = batch_size        #batch_size
        self.dec_units = dec_units          #decoder units

        """Embedding layer converts token IDs to vectors"""
        self.embedding = tf.keras.layers.Embedding(input_vocab_size,
                                                       embedding_dim)


        """Gru is helpful for long dependences of the text"""
        """RNN keeps track of what's been generated so far."""
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences = True,
                                       return_state = True,
                                       recurrent_initializer='glorot_uniform')
        # fully connected layer
        self.fc = tf.keras.layers.Dense(input_vocab_size)

        """The RNN output will be the query for the attention layer."""

        self.attention = BahdanauAttention(self.dec_units)



    def call(self, x, hidden, enc_output):

        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)
        # concate with context vector
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis = -1)
        # passing the concatenated vector to the GRU

        output, state  = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights




"""Object of the Decoder class and further testing"""

decoder = Decoder(vocab_received_size, embedding_dim, units, BATCH_SIZE)


sample_decoder_output, a, b = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

# Decoder output shape: (batch_size, vocab size)
#print(sample_decoder_output.shape)














