"""Neural Machine Translatron communication system
ut a diagnosis or a procedure, based on his status as a health care worker at a hospital.
translate.py helps translate the sent message to the receiver
and vice versa."""

"""The input to the decoder at each time step is its previous predictions
along with the hidden state and the encoder output.

We stop predicting when the model predicts the end token.

We store the attention weights for every time step."""

"""The encoder output is calculated only once for one input."""

# load the files
from preprocess import*
from create_dataset import*
from tokenspad import*
from train_test_creation import*
from encoder import*
from encodeFunction import*
from decoder import*
from training import*

"""Evaluation function"""
def evaluate(sentence):
  attention_plot = np.zeros((max_target_length, max_source_length))

  sentence = preprocess_data(sentence)
  #print(sentence)
  #print(source_sentence_tokenizer.word_index)

  inputs = [sent_message_tokenizer.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_source_length,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([received_message_tokenizer.word_index['start_']], 0)

  for t in range(max_target_length):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += received_message_tokenizer.index_word[predicted_id] + ' '

    if received_message_tokenizer.index_word[predicted_id] == '_end':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot

"""Plotting function"""
# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()


"""Let's translate """
def translate(sentence):
  result, sentence, attention_plot = evaluate(sentence)
  attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
  plot_attention(attention_plot, sentence.split(' '), result.split(' '))

  return result



"""Translate the model"""

def mycheck():

   trained_model =  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
   return trained_model

