#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
import torch
from math import log


# In[2]:


# Composition methods
def compose_add(embeddings, words):
    matrix = embeddings[words[0]]
    for word in words[1:]:
        #if word == "an":
        #    word = "a"
        matrix = matrix + embeddings[word]
    matrix = matrix / len(words)
    # Normalise matrix; ignore if vector
    if isinstance(matrix[0], list):
        matrix = matrix/matrix.trace()
    return(matrix)

def compose_mult(embeddings, words):
    matrix = embeddings[words[0]]
    for word in words[1:]:
        #if word == "an":
        #    word = "a"
        matrix = matrix * embeddings[word]
    if isinstance(matrix[0], list):
        matrix = matrix/matrix.trace()
    return(matrix)

def compose_tensor(embeddings, sequence1, sequence2,):
    if len(sequence1) == 2:
        verb1_vector = embeddings[sequence1[POS_TAGS[dataset_name].index("verb")]]
        verb1_tensor = np.outer(verb1_vector, verb1_vector)
        noun1_vector = embeddings[sequence1[POS_TAGS[dataset_name].index("noun")]]
        vector1 = verb1_tensor @ noun1_vector
       
    elif len(sequence1) == 3:
        subj1_vector = embeddings[sequence1[0]]
        verb1_vector = embeddings[sequence1[1]]
        verb1_tensor = np.outer(verb1_vector, verb1_vector)
        obj1_vector = embeddings[sequence1[2]]
        sub_obj_matrix1 = np.outer(subj1_vector, obj1_vector)
        matrix1 = verb1_tensor * sub_obj_matrix1
        vector1 = matrix1.flatten()

    elif len(sequence1) == 5:
        # Compose adj-subj verb adj-obj for sequence 1
        subj_adj1_vector = embeddings[sequence1[0]]
        subj_adj1_tensor = np.outer(subj_adj1_vector, subj_adj1_vector)
        subj1_vector = embeddings[sequence1[1]]
        subj_np1_vector = subj_adj1_tensor @ subj1_vector

        verb1_vector = embeddings[sequence1[2]]
        verb1_tensor = np.outer(verb1_vector, verb1_vector)

        obj_adj1_vector = embeddings[sequence1[3]]
        obj_adj1_tensor = np.outer(obj_adj1_vector, obj_adj1_vector)
        obj1_vector = embeddings[sequence1[4]]
        obj_np1_vector = obj_adj1_tensor @ obj1_vector

        sub_obj_matrix1 = np.outer(subj_np1_vector, obj_np1_vector)
        matrix1 = verb1_tensor * sub_obj_matrix1
        vector1 = matrix1.flatten()
        
def get_dm_sqrt(dm):
    eigvals, eigvecs = np.linalg.eigh(dm)
    eigvals[eigvals <= 0] = 0
    dm_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T # element-wise multiplication in bracket achieves matrix multiplication with diagonal matrix
    return dm_sqrt
    
def bmult(operator_dm, input_dm):
    operator_dm_sqrt = get_dm_sqrt(operator_dm) # ugly OOP?
    result = operator_dm_sqrt @ input_dm @ operator_dm_sqrt
    result = result / result.trace()
    return result
    
def kmult(operator_dm, input_dm):
    eigvals, eigvecs = np.linalg.eigh(operator_dm)
    eigvals[eigvals <= 0] = 0
    result = np.zeros(shape=operator_dm.shape)
    for i in range(operator_dm.shape[0]):
        if eigvals[i] > 0:
            eigvec_outer = np.outer(eigvecs.T[i, :], eigvecs.T[i, :])
            result += eigvals[i] * (eigvec_outer @ input_dm @ eigvec_outer)
    result = result / result.trace()
    return result
    
def fuzz_phaser(embeddings, sequence, composer, operator='noun', pos=None, prep='A'):
    
    # Specify fuzz or phaser
    if composer == 'fuzz' or composer == 'fuzz_simple':
        composer_func = kmult
    else:
        composer_func = bmult

    # Specify verb or noun as operator in composition
    def verb_compose(verb_dm, noun_dm, adj=False):
        return composer_func(operator_dm=verb_dm, input_dm=noun_dm)

    def noun_compose(verb_dm, noun_dm, adj=False):
        # if adj = True: composing adj-noun, not verb-noun so adj is operator. 
        if adj == False:
            return composer_func(operator_dm=noun_dm, input_dm=verb_dm)
        else:
            return composer_func(operator_dm=verb_dm, input_dm=noun_dm)
        
    if operator == 'verb':
        compose_func = verb_compose
    else:
        compose_func = noun_compose
    
    # treat prepositions as either adverbs or adjectives: compose with verb or with noun
    pos = pos.replace('P', prep)

    # Compose sentence: SVAJO as (((VA)(JO))S)
    
    if pos == '?':
        return None

    if len(sequence) == 2:
        if pos == 'SV':
            Vdm = embeddings[sequence[1]]
            Ndm = embeddings[sequence[0]]
        elif pos == 'VO':
            Vdm = embeddings[sequence[0]]
            Ndm = embeddings[sequence[1]]
            
        else:
            print(sequence)
            print(pos)

        result = compose_func(Vdm, Ndm)


    elif len(sequence) == 3:

        if pos in ('SVO','SVA','SVJ'):
            Sdm = embeddings[sequence[0]]
            Vdm = embeddings[sequence[1]]
            Odm = embeddings[sequence[2]]
            VOdm = compose_func(Vdm, Odm)
            result = compose_func(VOdm, Sdm)

        elif pos == 'VAO':
            Vdm = embeddings[sequence[0]]
            Adm = embeddings[sequence[1]]
            Odm = embeddings[sequence[2]]
            VAdm = compose_func(Vdm, Adm)
            result = compose_func(VAdm, Odm)

        elif pos == 'VJO':
            Vdm = embeddings[sequence[0]]
            Jdm = embeddings[sequence[1]]
            Odm = embeddings[sequence[2]]
            JOdm = compose_func(Jdm, Odm, True)
            result = compose_func(Vdm, JOdm)

    elif len(sequence) == 4:

        if pos == 'SAVO':
            Sdm = embeddings[sequence[0]]
            Adm = embeddings[sequence[1]]
            Vdm = embeddings[sequence[2]]
            Odm = embeddings[sequence[3]]
            VAdm = compose_func(Vdm, Adm)
            VAOdm = compose_func(VAdm, Odm)
            result = compose_func(VAOdm, Sdm)

        elif pos in ['SVAO','SVAA','SVAJ']:
            Sdm = embeddings[sequence[0]]
            Vdm = embeddings[sequence[1]]
            Adm = embeddings[sequence[2]]
            Odm = embeddings[sequence[3]]
            VAdm = compose_func(Vdm, Adm)
            VAOdm = compose_func(VAdm, Odm)
            result = compose_func(VAOdm, Sdm)

        elif pos == 'SVJO':
            Sdm = embeddings[sequence[0]]
            Vdm = embeddings[sequence[1]]
            Jdm = embeddings[sequence[2]]
            Odm = embeddings[sequence[3]]
            JOdm = compose_func(Jdm, Odm, True)
            VJOdm = compose_func(Vdm, JOdm)
            result = compose_func(VJOdm, Sdm)

        elif pos == 'VAAO':
            Vdm = embeddings[sequence[0]]
            Adm = embeddings[sequence[1]]
            Pdm = embeddings[sequence[2]]
            Odm = embeddings[sequence[3]]
            VAdm = compose_func(Vdm, Adm)
            VAPdm = compose_func(VAdm, Pdm)
            result = compose_func(VAPdm, Odm)

        elif pos == 'VAJO':
            Vdm = embeddings[sequence[0]]
            Adm = embeddings[sequence[1]]
            Jdm = embeddings[sequence[2]]
            Odm = embeddings[sequence[3]]
            VAdm = compose_func(Vdm, Adm)
            JOdm = compose_func(Jdm, Odm, True)
            result = compose_func(VAdm, JOdm)
            
        elif pos == 'VJJO':
            Vdm = embeddings[sequence[0]]
            Pdm = embeddings[sequence[1]]
            Jdm = embeddings[sequence[2]]
            Odm = embeddings[sequence[3]]
            JOdm = compose_func(Jdm, Odm, True)
            PJOdm = compose_func(Pdm, JOdm, True)
            result = compose_func(Vdm, PJOdm)

    elif len(sequence) == 5:

        if pos == 'SVAAO':
            Sdm = embeddings[sequence[0]]
            Vdm = embeddings[sequence[1]]
            Adm = embeddings[sequence[2]]
            Pdm = embeddings[sequence[3]]
            Odm = embeddings[sequence[4]]
            VAdm = compose_func(Vdm, Adm)
            VAPdm = compose_func(VAdm, Pdm)
            VAPOdm = compose_func(VAPdm, Odm)
            result = compose_func(VAPOdm, Sdm)
            
        if pos == 'SVAJO':
            Sdm = embeddings[sequence[0]]
            Vdm = embeddings[sequence[1]]
            Adm = embeddings[sequence[2]]
            Jdm = embeddings[sequence[3]]
            Odm = embeddings[sequence[4]]
            VAdm = compose_func(Vdm, Adm)
            JOdm = compose_func(Jdm, Odm, True)
            VAJOdm = compose_func(VAdm, JOdm)
            result = compose_func(VAJOdm, Sdm)
            
            
    try:
        result = result/result.trace()
        return result
    except UnboundLocalError:
        print(pos, sequence)


# In[3]:


# Compose and score
def compose_and_score(embeddings, sent1, sent2, method, pos=None, pos1=None, pos2=None, operator='noun', prep='A'):
    
    words1 = [word for word in sent1.lower().replace('_', ' ').replace(".","").split(' ') if word != '']
    words2 = [word for word in sent2.lower().replace('_', ' ').replace(".","").split(' ') if word != '']
    
    if method == 'add':
        sent1 = compose_add(embeddings, words1)
        sent2 = compose_add(embeddings, words2)
        

    if method == 'mult':
        sent1 = compose_mult(embeddings, words1)
        sent2 = compose_mult(embeddings, words2)
        
    
    if method == 'fuzz_simple':
        sent1 = fuzz_phaser(embeddings, words1, 'fuzz', operator, pos, prep)
        sent2 = fuzz_phaser(embeddings, words2, 'fuzz', operator, pos, prep)
        
        
    if method == 'phaser_simple':
        sent1 = fuzz_phaser(embeddings, words1, 'phaser', operator, pos, prep)
        sent2 = fuzz_phaser(embeddings, words2, 'phaser', operator, pos, prep)
        
        
    if method == 'fuzz':
        sent1 = fuzz_phaser(embeddings, words1, 'fuzz', operator, pos1, prep)
        sent2 = fuzz_phaser(embeddings, words2, 'fuzz', operator, pos2, prep)
        
    
    if method == 'phaser':
        sent1 = fuzz_phaser(embeddings, words1, 'phaser', operator, pos1, prep)
        sent2 = fuzz_phaser(embeddings, words2, 'phaser', operator, pos2, prep)
        
        
    # Flatten matrices before calculating cosine similarity
    if type(sent1) == list and type(sent2) == list:
        return cosine_similarity([sent1, sent2])[0,1]
    
    elif type(sent1) == None or type(sent2) == None:
        print('None appended')
        return None
    
    elif type(sent1) == np.ndarray and type(sent2) == np.ndarray:
        return cosine_similarity([sent1.flatten(), sent2.flatten()])[0,1]
    
def score_words(embeddings, word1, word2):
    if type(embeddings[word1]) == list:
        return cosine_similarity([embeddings[word1], embeddings[word2]])[0,1]
    else:
        return cosine_similarity([embeddings[word1].flatten(), embeddings[word2].flatten()])[0,1]
    
def score_sents(model, sent1, sent2):
    return cosine_similarity([model.encode([sent1], tokenize=False), model.encode([sent2], tokenize=False)])[0,1]


# In[5]:


def import_embeddings(filename, matrix=False):
        
    # Import txt file to dictionary of vectors    
    vectors = {}
    file = open(filename)
    for line in file.readlines():
        # Separate the numbers in each line
        vectors_str = line.split(' ')
        # Convert str to float
        floats = [float(x) for x in vectors_str[1:]]
        vectors[vectors_str[0]] = np.array(floats)

    if matrix==False:
        return vectors

    # Reshape vector to matrix
    if matrix==True:
        embeddings = {}
        for word in vectors:
            vector = vectors[word]
            DM_list = []
            for n in range(17):
                # Split into list of lists of length 17
                DM_list.append(vector[17*n:(17*(n+1))])
                # convert to array
                DM = np.array(DM_list)
            embeddings[word] = DM
        return embeddings


# In[4]:


# print words not in embeddings
def debug_vocab(embeddings, sent1, sent2):
    errors = set()
    words1 = sent1.lower().replace('_', ' ').split(' ')
    words2 = sent2.lower().replace('_', ' ').split(' ')
    for word in words1+words2:
        try:
            test = embeddings[word]
        except KeyError:
            errors.add(word)
    return errors


def embedding_compose_test(embeddings, sentence_data):
    errors = 0
    error_words = []
    for i, row in sentence_data.iterrows():
        for j in (1,2):
            try:
                compose_and_score(embeddings, row[0], row[j], 'add')
            except KeyError:
                errors += 1
                error_words.extend(debug_vocab(embeddings, row[2], row[4]))
    print('{} errors'.format(errors))
    print(error_words)

def embedding_test(embeddings, sentence_data):
    errors = 0
    error_words = []
    for i, row in sentence_data.iterrows():
        sent = row[1]+" "+row[2]+" "+row[3]
        words1 = [word for word in sent.lower().replace('.','').split(' ') if word != '']
        for word in words1:
            try:
                etest = embeddings[word]
            except KeyError:
                errors += 1
                error_words.append((word,i))
    print('{} errors'.format(errors))
    print(error_words)
# In[ ]:


def vne(matrix):
    eig_values = np.linalg.eig(matrix)[0]
    entropy = 0
    for i in eig_values:
        if i > 0:
            x = -i*log(i)
            entropy += x
    return entropy

