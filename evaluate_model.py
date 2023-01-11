""" 
      Author  : Suresh Pokharel
      Email   : sureshp@mtu.edu
"""

import numpy as np
import pandas as pd
from Bio import SeqIO
from keras import backend as K
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model


"""
define file paths and other parameters
"""

model_path = 'models/LMSuccSite.h5'
win_size = 33
cutoff_threshold = 0.5


def get_input_for_embedding(fasta_file):
    encodings = []
    
    # define universe of possible input values
    alphabet = 'ARNDCQEGHILKMFPSTWYV-'
    
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
        data = seq_record.seq
        for char in data:
            if char not in alphabet:
                return
        integer_encoded = [char_to_int[char] for char in data]
        encodings.append(integer_encoded)
    encodings = np.array(encodings)
    return encodings


# load test data
test_positive_pt5 = pd.read_csv("data/test/features/test_positive_ProtT5-XL-UniRef50.csv", header = None).iloc[:,2:]
test_negative_pt5 = pd.read_csv("data/test/features/test_negative_ProtT5-XL-UniRef50.csv", header = None).iloc[:,2:]

# create labels
test_positive_labels = np.ones(test_positive_pt5.shape[0])
test_negative_labels = np.zeros(test_negative_pt5.shape[0])

# stack positive and negative data together
X_test_pt5 = np.vstack((test_positive_pt5,test_negative_pt5))
y_test = np.concatenate((test_positive_labels, test_negative_labels), axis = 0)

# shuffle X and y together
# X_train_pt5, y_train_pt5 = shuffle(X_train_pt5, y_train_pt5)
# X_test_pt5, y_test_pt5 = shuffle(X_test_pt5, y_test_pt5)

# convert sequences to integer encoding, for embedding
test_positive_embedding = get_input_for_embedding('data/test/fasta/test_positive_sites.fasta')
test_negative_embedding = get_input_for_embedding('data/test/fasta/test_negative_sites.fasta')

# stack positive and negative data together
X_test_embedding = np.vstack((test_positive_embedding,test_negative_embedding))

# load saved model
combined_model = load_model(model_path)

# predict test data
y_pred = combined_model.predict([X_test_embedding,X_test_pt5]).reshape(y_test.shape[0],)
y_pred = (y_pred > cutoff_threshold)
y_pred = [int(i) for i in y_pred]
y_test = np.array(y_test)
y_pred = np.array(y_pred)

# calculate performance metrics
cm = confusion_matrix(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

sn = cm[1][1]/(cm[1][1]+cm[1][0])
sp = cm[0][0]/(cm[0][0]+cm[0][1])

print("\n %s, %s, %s, %s, %s \n" %(str(acc), str(mcc), str(sn), str(sp), cm))
