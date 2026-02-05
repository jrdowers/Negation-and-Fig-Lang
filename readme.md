## Datasets

`Fig-QA Dataset calc.xlsx` - main Fig-QA dataset with all annotations

`Fig-QA parsed2.csv` - used in extracting tense and concreteness annotation - output from `Concreteness tense.ipynb`

`negation sentences.xlsx` - literal negation dataset

`concreteness_category_lists2.csv` - used in calculating high/low concreteness categories - output from `test models PCA.ipynb`

`figqa_lemmatised.csv` - lemmatised version of the dataset

`Fig-QA Dataset copy.csv` - csv copy of the dataset used to run tests

`negsents_lem.csv` - lemmatised version of the literal negation dataset

`negsents copy.csv` - csv copy of the literal negation dataset used to run tests

`Fig-QA Dataset for PCA.xlsx` - Fig-QA dataset with correct paraphrase column used to run PCA and other analysis on the paraphrases

`LICENSE` - license for Fig-QA Dataset

## Code

`dm_functions_figqa-Copy.py` - functions used to run embedding models

`DataManipulation.ipynb` - used to create lemmatised datasets

`test embedding models.ipynb` - run embedding models and sentence encoders on dataset - 
To run this file models must be downloaded to the `Models` folder

https://github.com/tmikolov/word2vec

https://github.com/stanfordnlp/GloVe/releases/tag/1.2

https://github.com/facebookresearch/InferSent

https://github.com/francois-meyer/lexical-ambiguity-dms/tree/master

`test LLM openai.ipynb` - run models using openai API

`test LLM Together AI.ipynb` - run models using Together AI API

`test LLM local.ipynb` - run LLMs locally

`Concreteness tense.ipynb` - generates concreteness and tense labels - outputs `Fig-QA parsed2.csv`

`test models PCA.ipynb` - runs PCA on sentence embeddings and generates PCA plots; calculations for conccreteness/tense analysis

`error intervals and plots.ipynb` - calculates binomial confidence intervals and generates results plots

`stats.ipynb` - calculates p-values and exports to excel

## Results

`Results all Nov24` - contains all results

`test_model_(all/litneg/human)_summary.xlsx` - all results compiled and accuracies calculated and summarised in table




