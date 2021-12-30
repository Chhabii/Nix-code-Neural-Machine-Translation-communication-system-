"""Train test split
We use sklearn model_selection function
for this case, we use test_size = 0.2.
we can vary this while testing the accuracy"""
from preprocess import*
from create_dataset import*
from tokenspad import*

sent_train_tensor, sent_test_tensor, received_train_tensor, received_test_tensor = train_test_split(sent_tensor, received_tensor, test_size = 0.2)

# create train and validation split on 80-20 ratio

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(sent_tensor, received_tensor, test_size = 0.2)

"""Feel free to play around the variables.
print each variable's length, data type for better understanding"""
#print(len(input_tensor_val))


"""Write function to convert langauge to their respective tensors"""

def lang_to_tensor(lang, tensor):
    for t in tensor:
        if t!=0:
            print("%d ----> %s" %(t, lang.index_word[t]))

"""Try uncommenting the below comment to see index to word mapping"""
#print("sent message: index --> word")
#lang_to_tensor(sent_message_tokenizer, sent_train_tensor[10])
#print()
#lang_to_tensor(received_message_tokenizer, received_train_tensor[10])


"""Let's create the tensorflow dataset"""

BUFFER_SIZE = len(sent_train_tensor)
BATCH_SIZE = 64
steps_per_epoch = len(sent_train_tensor) // BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_sent_size = len(sent_message_tokenizer.word_index)+1
vocab_received_size = len(received_message_tokenizer.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((sent_train_tensor, received_train_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder = True)


"""Example testing """

example_input_batch, example_target_batch = next(iter(dataset))

# example_input_batch.shape, example_target_batch.shape

#print(example_target_batch.shape)









