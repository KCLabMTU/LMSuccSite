""" 
      Author  : Suresh Pokharel
      Email   : sureshp@mtu.edu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from Bio import SeqIO
from keras import backend as K
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten, Input,
                                     LeakyReLU, MaxPooling1D, Reshape)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm


def extract_one_windows_position(sequence,site,window_size=33):
    '''
    Description: Extract a window from the given string at given position of given size
                (Need to test more conditions, optimizations)
    Parameters:
        sequence (str):
        site:
        window_size(int):
    Returns:
        string: a window/section
    '''
    
    half_window = int((window_size-1)/2)
    
    # if window is greater than seq length, make the sequence long by introducing virtual amino acids
    # To avoid different conditions for virtual amino acids, add half window everywhere
    sequence = "X" * half_window + sequence + "X" * half_window
    
    section = sequence[site - 1 : site + 2 * half_window]
    return section


def get_input_for_embedding(window):
    # define universe of possible input values
    alphabet = 'ARNDCQEGHILKMFPSTWYVX'
    
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    
    for char in window:
        if char not in alphabet:
            return
    integer_encoded = np.array([char_to_int[char] for char in window])
    return integer_encoded


def get_protT5_features(sequence):
    # pass
    dummy = np.array([np.ones(1024)]*len(sequence))
    return dummy

# load test sequence
fasta_file = "input/sequence.fasta"

# create results dataframe
results_df = pd.DataFrame(columns = ['prot_desc', 'position','site_residue', 'probability', 'prediction'])

for seq_record in tqdm(SeqIO.parse(fasta_file, "fasta")):
    prot_id = seq_record.id
    sequence = seq_record.seq
    
    positive_predicted = []
    negative_predicted = []
    
    # extract protT5 for full sequence and store in temporary dataframe 
    pt5_all = get_protT5_features(sequence)
    
    # generate embedding features and window for each amino acid in sequence
    for index, amino_acid in enumerate(sequence):
        
        # check if AA is 'S' or 'T'
        if amino_acid in ['S', 'T']:
            site = index + 1
            
            # extract window
            window = extract_one_windows_position(sequence, site)
            
            # extract embedding_encoding
            X_test_embedding = get_input_for_embedding(window)
            
            # get ProtT5 features extracted above
            X_test_pt5 = pt5_all[index]
            
            # load model
            combined_model = load_model('models/LMSuccSite.h5')
                        
            y_pred = combined_model.predict([X_test_embedding.reshape(1,33), np.array(X_test_pt5.reshape(1,1024))], verbose = 0)[0][0]
            
            # append results to results_df
            results_df.loc[len(results_df)] = [prot_id, site, amino_acid, y_pred, int(y_pred>0.5)]

# Export results 
print('Saving results ...')
results_df.to_csv("output/results.csv", index = False)
print('Results saved to output/results.csv')