# creating the source and target tokens
# defining max length
# sent and received variable is imported from create_dataset
from preprocess import*
from create_dataset import*

"""In max_length() function, On passing the tensor as the argument, we get the max length 
of the tensor"""

def max_length(tensor):
    return max(len(t) for t in tensor)


"""create the source and the target token and post pad them each"""

sent_message_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = '')
sent_message_tokenizer.fit_on_texts(sent)
sent_tensor = sent_message_tokenizer.texts_to_sequences(sent)
sent_tensor = tf.keras.preprocessing.sequence.pad_sequences(sent_tensor, padding='post')       #post padding the tensors


""" create the same vector representation for target sentence
make sure to post pad the tokens."""

"""On printing the sent_tensor / received_tensor
we get vector of the sentence tokens.
sent_tensor[1] = [1, 7, 2, 0]"""

received_message_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = '')
received_message_tokenizer.fit_on_texts(received)                  #tokenize on received message
received_tensor = received_message_tokenizer.texts_to_sequences(received)
received_tensor = tf.keras.preprocessing.sequence.pad_sequences(received_tensor, padding='post')



# limit the size of the dataset to experiment faster

max_target_length= max(len(t) for t in  received_tensor)
max_source_length= max(len(t) for t in  sent_tensor)
