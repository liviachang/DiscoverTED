from __future__ import division
from datetime import datetime
from functools import partial as ftPartial
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from itertools import combinations, chain
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from scipy.stats import rankdata
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from stop_words import get_stop_words
from time import time
import cPickle as pickle
import numpy as np
import os
import pandas as pd
import random
import re
import sys
import textwrap
from scipy.stats import ttest_1samp

TALK_DATA_FN = '/Users/liviachang/Galvanize/capstone/data/talks_info_merged.csv'
USER_TALK_FN = '/Users/liviachang/Galvanize/capstone/data/train_users_info_transformed.csv'
TEST_USER_TALK_FN = '/Users/liviachang/Galvanize/capstone/data/test_users_info_transformed.csv'
RATING_MATRIX_FN = '/Users/liviachang/Galvanize/capstone/data/train_rating_matrix.csv'
TEST_RATING_MATRIX_FN = '/Users/liviachang/Galvanize/capstone/data/test_rating_matrix.csv'

TALK_DATA_SCRAPED_FN = '/Users/liviachang/Galvanize/capstone/data/talks_info_scraped.csv'
TALK_DATA_IDIAP_FN = '/Users/liviachang/TED/idiap/ted_talks-10-Sep-2012.json'
USER_DATA_IDIAP_FN = '/Users/liviachang/TED/idiap/ted_users-10-Sep-2012.json'

LDA_MODEL_FN = '/Users/liviachang/Galvanize/capstone/model/model_lda.pkl'
LDA_TOPICS_FN = '/Users/liviachang/Galvanize/capstone/model/data_lda_topics.pkl'
GROUP_DATA_FN = '/Users/liviachang/Galvanize/capstone/model/data_user_groups.pkl'

RATING_TYPES = ['Beautiful', 'Confusing', 'Courageous', 'Fascinating', \
  'Funny','Informative', 'Ingenious', 'Inspiring', 'Jaw-dropping', \
  'Longwinded', 'OK', 'Obnoxious', 'Persuasive', 'Unconvincing']

N_TOTAL_TOPICS = 10
N_GROUP_TOPICS = 2 ## N_DEEPER_TOPICS
N_REC_TOPICS = 2 ## N_WIDER_TOPICS
N_TALK_CANDIDATES = 5
N_TALKS_FOR_KWS = 15
N_TESTING_USERS = 1500

IS_PRINT_TIME = False

def print_time(msg, t1=None):
  t2 = time()

  t2_str = datetime.fromtimestamp(t2).strftime('%Y/%m/%d %H:%M:%S')
  if t1 is None:
    print '{}: {}'.format(t2_str, msg)
  else:
    print '{}: {} <== {:.0f} secs'.format(t2_str, msg, t2-t1)
  return t2

def load_talk_data(fn=TALK_DATA_FN, is_print=IS_PRINT_TIME):
  t1 = print_time('Loading talk data')

  tdf = pd.read_csv(fn)
  
  ## load talk ratings
  rating_cols = ['Beautiful', 'Confusing', 'Courageous', 'Fascinating', \
    'Funny','Informative', 'Ingenious', 'Inspiring', 'Jaw-dropping', \
    'Longwinded', 'OK', 'Obnoxious', 'Persuasive', 'Unconvincing']
  TK_ratings = tdf.copy()
  TK_ratings.tid = TK_ratings.tid.astype(str)
  TK_ratings = TK_ratings.set_index('tid')
  TK_ratings = TK_ratings.ix[:,rating_cols]

  ## load talk info
  info_cols = ['speaker', 'title', 'ted_event', 'description', 'keywords', 'related_themes']
  TK_info = tdf.copy()
  TK_info.tid = TK_info.tid.astype(str)
  TK_info = TK_info.set_index('tid')
  TK_info = TK_info.ix[:, info_cols]
  
  return TK_ratings, TK_info

def load_user_data(user_fn=USER_TALK_FN, rating_fn=RATING_MATRIX_FN):
  t1 = print_time('Loading user data')

  ## load the dataframe with each row as user-ftalk
  user_ftalk_df = pd.read_csv(user_fn)
  user_ftalk_df.tid = user_ftalk_df.tid.astype(int).astype(str)
  
  ## load the 0-1 matrix with rows=users and columns=talks
  ratings_mat = pd.read_csv(rating_fn)
  ratings_mat = ratings_mat.set_index('uid_idiap')
  
  return user_ftalk_df, ratings_mat

def load_ted_data():
  TK_ratings, TK_info = load_talk_data()
  U_ftalks, R_mat = load_user_data()

  return TK_ratings, TK_info, U_ftalks, R_mat

def load_LDA_topics_data():
  with open(LDA_TOPICS_FN) as f:
    TK_topics, TP_info = pickle.load(f)
  return TK_topics, TP_info

def load_LDA_model_data():
  with open(LDA_MODEL_FN) as f:
    token_mapper, LDA = pickle.load(f)
  return token_mapper, LDA

def load_group_data():
  with open(GROUP_DATA_FN) as f:
    G_rtopics, U_tscores, U_ftalks = pickle.load(f)
  return G_rtopics, U_tscores, U_ftalks
