from __future__ import division
from src.configs import *
import cPickle as pickle
from src.build_models import load_talk_data, get_topic_talks
from src.build_models import get_user_rtopics, get_user_rec_talks_per_user
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import textwrap
import re

def get_user_fav_ratings():
  rating_cols = ['Beautiful', 'Confusing', 'Courageous', 'Fascinating', \
    'Funny','Informative', 'Ingenious', 'Inspiring', 'Jaw-dropping', \
    'Longwinded', 'OK', 'Obnoxious', 'Persuasive', 'Unconvincing']

  rating_idx = range( len(rating_cols) )

  rating_dict = dict(zip(rating_idx, rating_cols) )

  for (idx, typ) in rating_dict.iteritems():
      print '{}: {}'.format(idx, typ)

  U_type_idx = raw_input('\nTypes you are interested in: ' + \
    '(say, \'4,5\' for \'Funny+Informative\'):  ')
  if U_type_idx == '':
    U_type_idx = '4,5'

  U_type_idx = U_type_idx.replace(' ', '').split(',')
  U_type_idx = map(int, U_type_idx)
  score = 1. / len(U_type_idx)

  U_irating = np.repeat(0., len(rating_cols))
  for uidx in U_type_idx:
    U_irating[uidx] = score

  print ', '.join([rating_dict[x] for x in U_type_idx])

  U_irating = pd.DataFrame([U_irating], columns=rating_cols, index=['New'])
  return U_irating

def get_user_fav_keywords():
  U_kws = raw_input('\nTopics you are interested in: (say, \'data science, finance\'):  ')

  if U_kws == '':
    U_kws = 'data science, technology, computer, economics, finance, market, investing'

  print U_kws

  return U_kws

def vectorize_topics(V, TK_info, n_talks=N_TALKS_FOR_KWS): ##FIXME tokenize, remove stop words...
  TP_talks = get_topic_talks(V, n_talks)
  topic_docs = []
  for topic_talks in TP_talks:
    talk_texts = TK_info.ix[list(topic_talks), ['keywords', 'description'] ]
    talk_texts = talk_texts.apply(lambda x: ' '.join(x.tolist()), axis=1)
    topic_docs.append(' '.join(list(talk_texts)) )
  
  TP_vec = TfidfVectorizer(stop_words='english')
  TP_tfidf = TP_vec.fit_transform(topic_docs).toarray()
  TP_vocabs = np.array(TP_vec.get_feature_names())

  TP_kws = get_topic_keywords(TP_tfidf, TP_vocabs)

  return topic_docs, TP_vec, TP_tfidf, TP_vocabs, TP_kws

def get_topic_keywords(TP_tfidf, vocabs, IS_PRINT=False):
  TP_kws = np.apply_along_axis(lambda x: vocabs[x.argsort()[::-1][:3]], 1, TP_tfidf)

  if IS_PRINT:
    print TP_kws

  return TP_kws

def get_user_gtopics_from_keywords(kws, TP_vec, TP_tfidf, n_topics=N_GROUP_TOPICS):
  U_tfidf = TP_vec.transform([kws])
  cos_sim = linear_kernel(U_tfidf, TP_tfidf)[0]
  gtopics = cos_sim.argsort()[::-1][:n_topics]
  return gtopics

def rec_for_new_user(TK_info, TK_ratings):
  new_fratings = get_user_fav_ratings()
  new_kws = get_user_fav_keywords()
  
  with open(BROADER_MODEL_NEW_USERS_FN) as f:
    V, G_rtopics = pickle.load(f)
  TP_talks = get_topic_talks(V, N_TALK_CANDIDATES)

  TP_docs, TP_vec, TP_tfidf, TP_vocabs, TP_kws = vectorize_topics(V, TK_info)
  new_gtopics = get_user_gtopics_from_keywords(new_kws, TP_vec, TP_tfidf)
  new_gtopics = str( sorted(new_gtopics) )
  new_rtopics = G_rtopics[new_gtopics]
  new_rtopics_talks = TP_talks[new_rtopics]
  new_rtalks = get_user_rec_talks_per_user( new_fratings, new_rtopics_talks, TK_ratings)
  return new_rtalks

def rec_for_existing_user(uid, TK_info):
  with open(BROADER_MODEL_EXISTING_USERS_FN) as f:
    U_rtalks = pickle.load(f)
  new_rtalks = U_rtalks[uid]
  return new_rtalks

def rec_talks(uid, TK_info, TK_ratings):
  if uid.lower() =='n':
    rec_tids = rec_for_new_user(TK_info, TK_ratings)
  else:
    rec_tids = rec_for_existing_user(uid, TK_info)

  LINE_LENGTH = 80
  for rtid in rec_tids:
    tt = TK_info.ix[rtid]
    print '\n====={}: {} (tid={})=====\n{}\n[keywords]\n{}\n[themes]\n{}'.format(\
      tt.speaker, tt.title, rtid,
      textwrap.fill(tt.description, LINE_LENGTH), \
      textwrap.fill(tt.keywords.replace('[','').replace(']',''), LINE_LENGTH),
      re.sub('\[|\]|u\'|\'|\"|u\"', '', tt.related_themes))
      #re.sub('\[|\]|u\'|\'|\"|u\"', '', tt.related_themes).split(', '))

if __name__ == '__main__':
  print 'Loading TED data'
  TK_ratings, TK_info = load_talk_data()
  
  msg = '\nPlease enter your UserID, or "n" (for a new user), or "Q" (for quit): '

  uid = raw_input(msg)
  while uid.lower() not in ['q', '']:
    rec_talks(uid, TK_info, TK_ratings)
    uid = raw_input(msg)



