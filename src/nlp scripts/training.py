"""Neural Machine Translation communication system"""
from preprocess import*
from create_dataset import*
from tokenspad import*
from train_test_creation import*
from encoder import*
from encodeFunction import*
from decoder import*

"""
Training
Now that you have all the model components, it's time to start training the model.
You'll need:

A loss function and optimizer to perform the optimization.
A training step function defining how to update the model for each sent/received batch.
A training loop to drive the training and save checkpoints
"""


"""Define the optimizer and the loss function"""

optimizer = tf.keras.optimizers.Adam() # Adam optimizer

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits = True, reduction='none'
)




def loss_function(real, predicted):

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, predicted)

    mask = tf.cast(mask, dtype=loss_.dtype)

    loss_ *=mask

    return tf.reduce_mean(loss_)

"""Checkpoints """

checkpoint_dir = 'training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(optimizer = optimizer,
                                 encoder = encoder,
                                 decoder = decoder)
"""
-  Pass the input through the encoder which return encoder output and the encoder hidden state.
-  The encoder output, encoder hidden state and the decoder input (which is the start token) is passed to the decoder.
-  The decoder returns the predictions and the decoder hidden state.
-  The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
-  Use teacher forcing to decide the next input to the decoder.
-  Teacher forcing is the technique where the target word is passed as the next input to the decoder.
-  The final step is to calculate the gradients and apply it to the optimizer and backpropagate.
"""
@tf.function

def train_steps(sent, received, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(sent, enc_hidden)

        # decoder hidden = encode_hidden
        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([received_message_tokenizer.word_index['start_']] * BATCH_SIZE, 1)

        # Tracher forcing, feeding the target as the next input

        for t in range(1, received.shape[1]):
            # passing output to the decoder

            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(received[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(received[:, t], 1)

    batch_loss = (loss / int(received.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

'''
EPOCHS = 100

for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = encoder.initialize_hidden_state()
    total_loss  = 0

    for (batch, (sent, received)) in enumerate(dataset.take(steps_per_epoch)):

        batch_loss = train_steps(sent, received, enc_hidden)
        total_loss += batch_loss

        if batch % 100 ==0:
            print("Epoch {} Batch {} Loss {}".format(epoch + 1, batch, batch_loss.numpy()))
    # saving the checkpoint() on every two epoch

    if (epoch + 1)% 2 ==0:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print("Epoch {} Loss {:.4f}".format(epoch + 1, total_loss/steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    '''
