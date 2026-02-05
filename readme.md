## Datasets

`Fig-QA Dataset calc.xlsx` - main Fig-QA dataset (train set) with all annotations - includes calculations

`Fig-QA Dataset copy.csv` - csv version of Fig-QA (train set) with annotations

`Fig-QA Dataset calc copy.xlsx` - annotated Fig-QA test set

`negation sentences.xlsx` - literal negation dataset

`negsents copy.csv` - csv version of the literal negation dataset

`Fig-QA parsed.csv` - used in extracting tense and concreteness annotation - output from `Concreteness tense.ipynb`

`Fig-QA Dataset for PCA.xlsx` - Fig-QA dataset with correct paraphrase column used to run PCA and other analysis on the paraphrases

`concreteness_category_lists.xlsx` and `concreteness_category_lists_test.xlsx` - used in calculating high/low concreteness categories - output from `test models PCA.ipynb`

`figqa_lemmatised.csv` - lemmatised version of the dataset

`negsents_lem.csv` - lemmatised version of the literal negation dataset

`LICENSE` - license for Fig-QA Dataset

## Code

`dm_functions_figqa.py` - functions used to run embedding models

`DataManipulation.ipynb` - used to create lemmatised datasets

`Concreteness tense.ipynb` - generates concreteness and tense labels - outputs `Fig-QA parsed.csv`

`test embedding models.ipynb` - run embedding models and sentence encoders on dataset - 
To run this file models must be downloaded to the `Models` folder

https://github.com/tmikolov/word2vec

https://github.com/stanfordnlp/GloVe/releases/tag/1.2

https://github.com/facebookresearch/InferSent

https://github.com/francois-meyer/lexical-ambiguity-dms/tree/master

`test LLM openai.ipynb` - run models using openai API

`test LLM Together AI.ipynb` - run models using Together AI API

`test LLM local.ipynb` - run LLMs locally

`test models PCA.ipynb` - runs PCA on sentence embeddings and generates PCA plots; calculations for concreteness/tense analysis

`error intervals and plots.ipynb` - calculates confidence intervals and generates results plots

`stats.ipynb` - calculates p-values and exports to excel

## Results

`Results all Nov24` - contains all results

`test_model_(all/test/litneg/human)_summary.xlsx` - results compiled and accuracies summarised in table

`Nov25_counts-for-errors_combined copied.xlsx` - all results combined for calculation of confidence intervals

`results with errors output totals combined copied.csv` - all results with confidence intervals - used to generate plots

`stats_out.csv` - calculated p-values from `stats.ipynb`

`stats_corrected_reorder.xlsx` - p-values with multiple comparisons correction

`pca_accuracies_full_sbertall.csv`, `pca_accuracies_negpairs_sbertall.csv`, `pca_concproportions-all_sbertall.csv`, `pca_negproportions-all_sbertall.csv` - results from PCA analysis



