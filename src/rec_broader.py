from __future__ import division
from src.configs import *
import cPickle as pickle
from src.build_models import load_ted_data
import numpy as np
import pandas as pd

def get_user_input_fav_ratings():
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

def get_user_input_keywords():
  U_kws = raw_input('\nTopics you are interested in: (say, \'data science, finance\'):  ')
  if U_kws == '':
    U_kws = 'data science, finance'
  return U_kws

def get_topic_docs(TP_talks, TK_info): ##FIXME tokenize, remove stop words...
  topic_docs = []
  for topic_talks in TP_talks:
    talk_texts = TK_info.ix[list(topic_talks), ['keywords', 'description'] ]
    talk_texts = talk_texts.apply(lambda x: ' '.join(x.tolist()), axis=1)
    topic_docs.append(' '.join(list(talk_texts)) )
  return topic_docs

def get_user_gtopics_from_keywords(kws, TP_docs, n_topics):
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.metrics.pairwise import linear_kernel
  vectorizer = TfidfVectorizer(stop_words='english') ## word counter
  TP_tfidf = vectorizer.fit_transform(TP_docs).toarray() ## shape = k_topics x n_vocab
  vocab = vectorizer.get_feature_names()
  U_tfidf = vectorizer.transform([kws])
  cos_sim = linear_kernel( U_tfidf, TP_tfidf )[0]
  gtopics = cos_sim.argsort()[::-1][:n_topics]
  return gtopics

def rec_for_new_user():
  U_fratings = get_user_input_fav_ratings()
  U_kws = get_user_input_keywords()

  TP_docs = get_topic_docs(TP_talks, TK_info)
  gtopics = str( get_user_gtopics_from_keywords(U_kws, TP_docs, N_GROUP_TOPICS))
  newU_gtopics = ('New', gtopics)


def rec_for_existing_user(uid):
  print TK_info.ix[U_rtalks[uid], :4]

def rec_talks(uid):
  if uid=='New':
    rec_for_new_user()
  else:
    rec_for_existing_user(uid)
  
if __name__ == '__main__':
#  print 'Loading the model to learn broader'
#  with open(BROADER_MODEL_FN) as f:
#    U, V, U_gtopics, G_rtopics, TP_talks, U_fratings, U_rtalks = pickle.load(f)
#  
#  print 'Loading TED data'
#  TK_ratings, TK_info, U_ftalks, R_mat = load_ted_data()
#
  uid = ['000fe7196ce60cdfa26e1e69364c85ee8aaf8931', 'New'][1]
  rec_talks(uid)



