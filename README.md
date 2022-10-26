# LMSuccSite
### Introduction
Improving Protein Succinylation Sites Prediction Using Features Extracted from Protein Language Model

### Install libraries
`pip install -r requirements.txt`

### Model evaluation using the existing benchmark independent test set
Please run the `evaluate_model.py` script.

### To run `LMSuccSite` model on your own sequences 
In order to predict succinylation site using your own sequence, you need to have two inputs:
1. Clone the LMSuccSite Repo
 `git clone git@github.com:KCLabMTU/LMSuccSite.git`
2. Copy sequences to `input/sequence.fasta`
3. Install transformers 
`pip install -q SentencePiece transformers`
4. Run `predict.py`
5. Find results inside `output` folder


### To train the model using our training set (To be updated)


### To train the model using your own dataset (To be updated)


## Extracting ProtT5 Features (To be updated)



## Citation
Pokharel, S., Pratyush, P., Heinzinger, M. et al. Improving protein succinylation sites prediction using embeddings from protein language model. Sci Rep 12, 16933 (2022). https://doi.org/10.1038/s41598-022-21366-2

Link: https://rdcu.be/cXFfM


## Contact
Please send an email to `sureshp@mtu.edu` (CC: `dbkc@mtu.edu`) for any kind of queries and discussions.
