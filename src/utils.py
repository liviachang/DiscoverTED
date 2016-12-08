## commonly used packages
from __future__ import division
from datetime import datetime
from functools import partial as ftPartial
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from itertools import combinations, chain
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from scipy.stats import rankdata
from scipy.stats import ttest_1samp
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
from stop_words import get_stop_words
from time import time
import cPickle as pickle
import dill
import graphlab as gl
import numpy as np
import os
import pandas as pd
import random
import re
import sys
import textwrap
from scipy.spatial.distance import cdist

## file paths
TALK_DATA_FN = '/Users/liviachang/Galvanize/capstone/data/talks_info_merged.csv'
USER_TALK_FN = '/Users/liviachang/Galvanize/capstone/data/train_users_info_transformed.csv'
TEST_USER_TALK_FN = '/Users/liviachang/Galvanize/capstone/data/test_users_info_transformed.csv'
RATING_MATRIX_FN = '/Users/liviachang/Galvanize/capstone/data/train_rating_matrix.csv'
TEST_RATING_MATRIX_FN = '/Users/liviachang/Galvanize/capstone/data/test_rating_matrix.csv'

TALK_DATA_SCRAPED_FN = '/Users/liviachang/Galvanize/capstone/data/talks_info_scraped.csv'
TALK_DATA_IDIAP_FN = '/Users/liviachang/TED/idiap/ted_talks-10-Sep-2012.json'
USER_DATA_IDIAP_FN = '/Users/liviachang/TED/idiap/ted_users-10-Sep-2012.json'

TOPIC_MODEL_LDA_FN = '/Users/liviachang/Galvanize/capstone/model/topic_model_lda.pkl'
RATING_TYPES = ['Beautiful', 'Confusing', 'Courageous', 'Fascinating', \
  'Funny','Informative', 'Ingenious', 'Inspiring', 'Jaw-dropping', \
  'Longwinded', 'OK', 'Obnoxious', 'Persuasive', 'Unconvincing']
INFO_COLS = ['speaker', 'title', 'ted_event', 'description', 'keywords', 'related_themes']

## recommender configurations
N_TOTAL_TOPICS = 10
N_GROUP_TOPICS = 2 ## N_DEEPER_TOPICS
N_REC_TOPICS = 2 ## N_WIDER_TOPICS
N_TALK_CANDIDATES = 5
N_TALKS_FOR_KWS = 15
N_TESTING_USERS = 1500

## widely used functions to test elapsed time
def print_time(msg, t1=None):
  t2 = time()

  t2_str = datetime.fromtimestamp(t2).strftime('%Y/%m/%d %H:%M:%S')
  if t1 is None:
    print '{}: {}'.format(t2_str, msg)
  else:
    print '{}: {} <== {:.0f} secs'.format(t2_str, msg, t2-t1)
  return t2

