# BahdanauAttention Model
"""The encoder returns its internal state so that its state 
can be used to initialize the decoder.

"""
from preprocess import*
from create_dataset import*
from tokenspad import*
from train_test_creation import*
from encoder import*

class BahdanauAttention(tf.keras.layers.Layer):
    """Initialization function"""

    def __init__(self, units):
        # super class
        super(BahdanauAttention, self).__init__()
        """weights"""
        self.W1  = tf.keras.layers.Dense(units)
        self.W2  = tf.keras.layers.Dense(units)
        self.V   = tf.keras.layers.Dense(1)

    def call(self, query, values):
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape = (batch_size, max_length. 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        #attention weights shape
        attention_weights = tf.nn.softmax(score, axis =1)
        # context vector shape 
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# driver code for testing

# create object of the class BahdanauAttention

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)


#print(attention_result.shape)
