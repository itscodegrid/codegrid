
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from gensim.models import Word2Vec
from config import bcolors
from config import RAW_DATASET_FOLDER, TRAINING_DATASETS, DATA_FILENAME, CODE_WITHOUT_COMMENTS_KEY, CODE_RAW_TOKENS_KEY, INNITIAL_PROCESSED_DATASET_FOLDER
from config import SKIP_INITIAL_PROCESS, W2V_VECTOR_DIM, NUM_WORKERS, W2V_WINDOWS_SIZE, W2V_MIN_TOKEN_LEN, NUM_EPOCHS, W2V_TRAINING_SEED, W2V_MODEL_NAME, COLOR_MODEL_NAME, MODELS_FOLDER
from config import SPACE_TOKEN, LINE_BREAK_TOKEN, TOKEN_SPLIT_PATTERN, UNK_TOKEN
from utils import remove_comments, beautify_code, generate_colors, tokenize_code
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from statistics import mean as average

import pickle
import re
import pandas as pd

tqdm.pandas()


def beautify_and_tokenize_code(code):
    code_no_comment = remove_comments(code)
    beautiful_code = beautify_code(code_no_comment, method='astyle')
    tokens = tokenize_code(beautiful_code)
    return beautiful_code, tokens

def build_color_vectorizer(corpus):
    vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r'\S+')
    X = vectorizer.fit_transform(corpus)
    tokens_tifids = dict()
    tfidfs = vectorizer.fit_transform(corpus)
    tokens_reorder = vectorizer.get_feature_names_out()
    tfidfs = tfidfs.toarray()
    colors = generate_colors(len(tokens_reorder) + 1)
    sorted_colors = [c for _, c in sorted(zip(colors[2], colors[0]), reverse=True)]
    model = dict()
    for i, token in enumerate(tokens_reorder):
        tokens_tifids[token] = average([tfidf[i] for tfidf in tfidfs])
        model[token] = sorted_colors[i]
    model[UNK_TOKEN] = sorted_colors[-1]
    return model

if __name__ == '__main__':
    all_data = []
    for ds in TRAINING_DATASETS:
        task_data = []
        ds_name = ds.replace('_', ' ')
        ds_name = ds_name.capitalize()
        print(bcolors.HEADER + f'Initial Processing of {ds_name} Data...' + bcolors.ENDC)
        processed_filename = INNITIAL_PROCESSED_DATASET_FOLDER / ds / DATA_FILENAME
        processed_filename.parent.mkdir(parents=True, exist_ok=True)
        if SKIP_INITIAL_PROCESS and processed_filename.is_file():
            print(bcolors.BOLD + f'\tLoading Data From Cache...' + bcolors.ENDC)
            data = pickle.load(open(processed_filename, mode='rb'))
        else:
            ds_filename = RAW_DATASET_FOLDER / ds / DATA_FILENAME
            data = pickle.load(open(ds_filename, mode='rb'))
            
            print(bcolors.BOLD + f'\tRemoving Comments and Tokenizing Source Code...' + bcolors.ENDC)
            data[[CODE_WITHOUT_COMMENTS_KEY, CODE_RAW_TOKENS_KEY]] = data.progress_apply(lambda row: beautify_and_tokenize_code(row['code']), axis=1, result_type='expand')
            
            print(bcolors.BOLD + f'\tSaving Preprocessed Data...' + bcolors.ENDC)
            data.to_pickle(processed_filename)

        task_data = data[CODE_RAW_TOKENS_KEY].tolist()
        all_data.extend(task_data)

        corpus = []
        for d in task_data:
            corpus.append(' '.join(d))

        print(bcolors.BOLD + '\tBuilding Color Vectorizer...' + bcolors.ENDC)
        color_model_path = MODELS_FOLDER / str(ds + COLOR_MODEL_NAME)
        if SKIP_INITIAL_PROCESS and color_model_path.is_file():
            pass
        else:
            token2color_model = build_color_vectorizer(corpus)
            pickle.dump(token2color_model, open(str(color_model_path), mode='wb'))

        print(bcolors.BOLD + '\tTraining Word2Vec Vectorizer Model...' + bcolors.ENDC)
        w2v_model_path = MODELS_FOLDER / str(ds + W2V_MODEL_NAME)
        if SKIP_INITIAL_PROCESS and w2v_model_path.is_file():
            pass
        else:
            w2v_model = Word2Vec(sentences=task_data, vector_size=W2V_VECTOR_DIM, window=W2V_WINDOWS_SIZE, min_count=W2V_MIN_TOKEN_LEN, workers=NUM_WORKERS, seed=W2V_TRAINING_SEED, epochs=NUM_EPOCHS)
            w2v_model.save(str(w2v_model_path))

