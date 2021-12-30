# Neural Machine Translation 
# Communication System

# load the libraries
from __future__ import absolute_import, division, print_function, unicode_literals


import os
import time
import io
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re # regular expression
import unicodedata
# Train and Test split
from sklearn.model_selection import train_test_split

# load the data location
data_location = "data/spa.txt"

"""Data that contains sender's message and receiver's message is an argument that is passed
inside the read_me function. The column name is nade as Sender and Receiver."""
def read_me(path):
    # prepare the data
    data = pd.read_table(data_location, names = ["sent", "received"])
    return data.sample(10)
#print(read_me(data_location))
# Use regex to clean the data
"""Cleaning the data is the foremost process.
Let's clean and preprocess the data."""
def preprocess_data(message):
    num_dig = str.maketrans('','',message)
    message = message.lower()                               # Lowercase message
    message = re.sub("  +",' ', message)
    message = re.sub("'", '',message)
    message = message.strip()
    message = re.sub(r"([?.!,¿])", r" \1 ", message)
    message = message.rstrip().strip()
    message = 'start_ '+message+' _end'

    return message

