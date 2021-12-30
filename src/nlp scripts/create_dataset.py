# Neural Machine Translation communication system
from preprocess import*
"""
Cleaning the messages for better data processing.
Making two langauge pairs

"""
def create_dataset(location, sample_size):
    """Location is the path of the data and
    sample_size is the total size we are taking
    from entire dataset. Simply, size of the sample."""

    lines = io.open(location, encoding='UTF-8').read().strip().split('\n')
    # play around the function
    # print(lines)
    """ lan_pair is the pair of two langauges.
    For example: Nepali:English / French:German"""

    lan_pair = [[preprocess_data(w) for w in l.split('\t')] for l in lines[:sample_size]]
    return zip(*lan_pair)


sample_size = 50000
sent, received = create_dataset(data_location, sample_size)
#print(sent[-1])

def max_length(tensor):
  return max(len(t) for t in tensor)
