import re
import numpy as np

# clean up given string
def clean_text(raw): 
    # Remove link
    raw = re.sub(r'http\S+', '', raw)
    # Remove unexpected artifacts
    raw = re.sub(r'â€¦', '', raw)
    raw = re.sub(r'…', '', raw)
    raw = re.sub(r'â€™', "'", raw)
    raw = re.sub(r'â€˜', "'", raw)
    raw = re.sub(r'\$q\$', "'", raw)
    raw = re.sub(r'&amp;', "and", raw)
    # remove non valid characters
    raw = re.sub('[^A-Za-z0-9#@ ]+', "", raw)
    return(raw)

# create onehot representation of the label
def get_onehot(arr, num_class):
    return np.eye(num_class)[np.array(arr).reshape(-1)]