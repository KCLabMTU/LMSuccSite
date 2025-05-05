<div align="center">
  
## LMSuccSite
Improving Protein Succinylation Sites Prediction Using Features Extracted from Protein Language Model

<p align="center">
<a href="https://www.python.org/"><img alt="python" src="https://img.shields.io/badge/Python-3.9.7-blue.svg"/></a>
<a href="https://biopython.org/"><img alt="Bio" src="https://img.shields.io/badge/Bio-1.5.2-brightgreen.svg"/></a>
<a href="https://keras.io/"><img alt="Keras" src="https://img.shields.io/badge/Keras-2.9.0-red.svg"/></a>
<a href="https://matplotlib.org/"><img alt="matplotlib" src="https://img.shields.io/badge/matplotlib-3.5.1-blueviolet.svg"/></a>
<a href="https://numpy.org/"><img alt="numpy" src="https://img.shields.io/badge/numpy-1.23.5-red.svg"/></a>
<a href="https://pandas.pydata.org/"><img alt="pandas" src="https://img.shields.io/badge/pandas-1.5.0-yellow.svg"/></a>
<a href="https://requests.readthedocs.io/en/latest/"><img alt="Requests" src="https://img.shields.io/badge/requests-2.27.1-blue.svg"/></a>
<a href="https://scikit-learn.org/"><img alt="scikit_learn" src="https://img.shields.io/badge/scikit_learn-1.2.0-blue.svg"/></a>
<a href="https://seaborn.pydata.org/"><img alt="seaborn" src="https://img.shields.io/badge/seaborn-0.11.2-lightgrey.svg"/></a>
<a href="https://www.tensorflow.org/"><img alt="tensorflow" src="https://img.shields.io/badge/TensorFlow-2.9.1-orange.svg"/></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.11.0-orange.svg"/></a>
<a href="https://tqdm.github.io/"><img alt="tqdm" src="https://img.shields.io/badge/tqdm-4.63.0-blue.svg"/></a>
<a href="https://huggingface.co/transformers/"><img alt="Transformers" src="https://img.shields.io/badge/Transformers-4.20.1-yellow.svg"/></a>
<a href="https://xgboost.readthedocs.io/en/stable/"><img alt="XGBoost" src="https://img.shields.io/badge/xgboost-1.5.0-blueviolet.svg"/></a><br>
<a href="https://github.com/KCLabMTU/LMSuccSite/commits/main"><img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/KCLabMTU/LMSuccSite.svg?style=flat&color=blue"></a>
<a href="https://github.com/KCLabMTU/LMSuccSite/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/KCLabMTU/LMSuccSite.svg?style=flat&color=blue"></a>
<a href="https://github.com/KCLabMTU/LMSuccSite/pulls"><img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/KCLabMTU/LMSuccSite.svg?style=flat&color=blue"></a>
</p>

</div>

## Web Server
<a href="http://kcdukkalab.org/LMSuccSite/" target="_blank">http://kcdukkalab.org/LMSuccSite/</a>  (Not working at the moment, we are working to fix issues)

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

If git is installed on your machine, clone the repository by entering this command into the terminal: 
```shell
git clone git@github.com:KCLabMTU/LMSuccSite.git
```
 or download the repository as a zip file by clicking [here](https://github.com/KCLabMTU/LMSuccSite/archive/refs/heads/main.zip)
 
### Install Libraries

Python version: `3.9.7`

To install the required libraries, run the following command:

```shell
pip install -r requirements.txt
```

Required libraries and versions: <br>
<code>Bio==1.5.2</code><br>
<code>keras==2.9.0</code><br>
<code>matplotlib==3.5.1</code><br>
<code>numpy==1.23.5</code><br>
<code>pandas==1.5.0</code><br>
<code>protobuf==3.20.*</code><br>
<code>requests==2.27.1</code><br>
<code>scikit_learn==1.2.0</code><br>
<code>seaborn==0.11.2</code><br>
<code>tensorflow==2.9.1</code><br>
<code>torch==1.11.0</code><br>
<code>tqdm==4.63.0</code><br>
<code>transformers==4.18.0</code><br>
<code>xgboost==1.5.0</code><br>


### Install Transformers
<code> pip install -q SentencePiece transformers</code>

### Model evaluation using the existing benchmark independent test set
Please run the `evaluate_model.py` script.
To evaluate our model on the independent test set, we have already placed the test sequences and corresponding ProtT5 features in `data/test/` folder. Once you install the requirements, run the following command:
<br>
```shell
python evaluate_model.py
```

### To run `LMSuccSite` model on your own sequences 

In order to predict succinylation site using your own sequence, you need to have two inputs:
1. Copy sequences you want to predict to `input/sequence.fasta`
2. Run `python predict.py`
3. Find results inside `output` folder


### Training and other experiments
1. Find training data at `data/train/` folder
2. Find all the codes and models related to training at `training codes` folder.


## Citation
Pokharel, S., Pratyush, P., Heinzinger, M. et al. Improving protein succinylation sites prediction using embeddings from protein language model. Sci Rep 12, 16933 (2022). https://doi.org/10.1038/s41598-022-21366-2

Link: https://rdcu.be/cXFfM


## Contact
Please send an email to sureshp@mtu.edu (CC: dbkc@mtu.edu, ppratyush@mtu.edu) for any kind of queries and discussions.

Additional files can be found at https://drive.google.com/drive/folders/1gzRzxoNI3LTWuU24qiBB-vu1t6-AsGW4?usp=drive_link
