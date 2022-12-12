## LMSuccSite
Improving Protein Succinylation Sites Prediction Using Features Extracted from Protein Language Model

## Authors
Suresh Pokharel<sup>1</sup>, Pawel Pratyush<sup>1</sup>, Michael Heinzinger<sup>2</sup>, Robert H. Newman<sup>3</sup>, Dukka B KC<sup>1*</sup>
<br>
<sup>1</sup>Department of Computer Science, Michigan Technological University, Houghton, MI, USA
<br>
<sup>2</sup>Department of Informatics, Bioinformatics and Computational Biology - i12, TUM (Technical University of Munich), Boltzmannstr. 3, 85748, Garching/Munich, Germany
<br>
<sup>3</sup>Department of Biology, College of Science and Technology, North Carolina A&T State University, Greensboro, NC, USA
<br><br>
<sup>*</sup> Corresponding Author: dbkc@mtu.edu
## Installation
Clone the repository: `git clone git@github.com:KCLabMTU/LMSuccSite.git` or download `https://github.com/KCLabMTU/LMSuccSite`
### Install Libraries
Python version: `3.9.7`

Install from requirement.txt: 
<code>
pip install -r requirements.txt
</code>

Required libraries and versions: 
<code>
Bio==1.5.2
keras==2.9.0
matplotlib==3.5.1
numpy==1.23.5
pandas==1.5.0
requests==2.27.1
scikit_learn==1.2.0
seaborn==0.11.2
tensorflow==2.9.1
torch==1.11.0
tqdm==4.63.0
transformers==4.18.0
xgboost==1.5.0
</code>

### Install Transformers
<code>
pip install -q SentencePiece transformers
</code>

### Model evaluation using the existing benchmark independent test set
Please run the `evaluate_model.py` script.
To evaluate our model on the independent test set, we have already placed the test sequences and corresponding ProtT5 features in `data/test/` folder. Once you install the requirements, run the following command:
<br>
<code>
 python evaluate_model.py
</code>

### To run `LMSuccSite` model on your own sequences 

In order to predict succinylation site using your own sequence, you need to have two inputs:
1. Copy sequences you want to predict to `input/sequence.fasta`
2. Run `python predict.py`
3. Find results inside `output` folder


### Training and other experiments
1. Find training data at `data/train/` folder
2. Find all the codes and models related to training at `training_experiments` folder (To be updated).


## Citation
Pokharel, S., Pratyush, P., Heinzinger, M. et al. Improving protein succinylation sites prediction using embeddings from protein language model. Sci Rep 12, 16933 (2022). https://doi.org/10.1038/s41598-022-21366-2

Link: https://rdcu.be/cXFfM


## Contact
Please send an email to sureshp@mtu.edu (CC: dbkc@mtu.edu) for any kind of queries and discussions.
