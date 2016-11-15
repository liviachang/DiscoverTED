from __future__ import division
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import cPickle as pickle
from matplotlib import pyplot as plt
from scipy.stats import rankdata
from functools import partial as ftPartial
from collections import Counter
from time import time
from datetime import datetime
import os
#import graphlab as gl
#from mf import matrix_factorization

def load_rating_data(rating_fn):
  t1 = print_time('Loading the rating data')

  ## load the data
  R = pd.read_csv(rating_fn)
  R = R.set_index('uid_idiap')

  t2 = print_time('Loading the rating data', t1)
  print 'R.shape={}\n'.format(R.shape)
  return R

def build_nmf(rating_fn, k, IS_LOAD_UV=True):
  ## get model filename
  model_fn = rating_fn.replace('/data/rating_matrix', '/model/UV_matrix')

  if IS_LOAD_UV and os.path.isfile(model_fn):
    with open(model_fn) as f:
      U, V = pickle.load(f)
    return U, V
  else:
    t1 = print_time('Building NMF models with k={}'.format(k))
  
    ## load user_talk rating matrix
    ## R.shape = n_users x m_talks
    R = load_rating_data(rating_fn)

    ## get NMF R = U x V
    ## U.shape = n_users x k_topics
    ## V.shape = k_topics x m_talks
    nmf = NMF(n_components = k, random_state=319)
    nmf.fit(R.values)
    U = pd.DataFrame(nmf.transform(R.values), index=R.index)
    V = pd.DataFrame(nmf.components_, columns=R.columns)
    E = nmf.reconstruction_err_
    
    ## save the model (i.e. U and V matrix)
    with open(model_fn, 'wb') as f:
      pickle.dump( (U,V), f)

    t2 = print_time('Building NMF models with k={}'.format(k), t1)
    print 'R.shape={}, U.shape={}, V.shape={}, nmf_err={:.4f}\n'.format(\
      R.shape, U.shape, V.shape, E)
    return U, V

def get_topic_rankings_per_user(x):
  n_topcis = len(x)
  n_nonzero = sum(x>0.)
  ranks = n_topcis - rankdata(x) + 1
  ranks = [tmp if tmp<=n_nonzero else np.nan for tmp in ranks]
  return ranks

def get_ptopics_per_user(x, N_PEER_TOPICS):
  top_topics = sorted(np.where(x.values<=N_PEER_TOPICS)[0])
  return str(top_topics)

def get_ptopics(U, N_PEER_TOPICS):
  t1 = print_time('Getting {} peer topics for each user'.format(N_PEER_TOPICS))
  ## get the rankings of the topcis (i.e. latent featues) for each user based on U matrix
  ## U_ranks.shape = n_users x k_topics
  U_ranks = U.apply(get_topic_rankings_per_user, axis=1)
  
  ## get the top N_PEER_TOPICS topics for each user
  ## this is used to define peers
  tmpf = ftPartial(get_ptopics_per_user, N_PEER_TOPICS=N_PEER_TOPICS)
  U_ptopics = U_ranks.apply(tmpf, axis=1)

  t2 = print_time('Getting {} peer topics for each user'.format(N_PEER_TOPICS), t1)
  print 'U_ptopics.shape = {}\n'.format(U_ptopics.shape)

  return U_ranks, U_ptopics

def get_ptopic_peers(U_ptopics):
  t1 = print_time('Getting the mapping from ptopics to peers')
  ## find peers for each group with top N_PEER_TOPICS peer topics
  ## peers is a dictionary {peer_topics, user_ids}
  peers = {} 
  for peer_topic in U_ptopics.unique():
    user_idx = np.where( U_ptopics==peer_topic )
    user_ids = U_ptopics.index[user_idx]
    peers[peer_topic] = user_ids

  t2 = print_time('Getting the mapping from ptopics to peers', t1)
  
  psize = [len(v) for k,v in peers.iteritems()]
  print '# grps={}, # total users={}, (min, max) grp size=({}, {})\n'.format( \
    len(peers), sum(psize), min(psize), max(psize) )

  return peers


def find_peers(U, N_PEER_TOPICS=2):
  print '\nFinding peers based on latent features'

  ## for every user in U matrix, find top N_PEER_TOPICS as the peer_topics
  ## user_ptopics.shape = n_users x N_PEER_TOPICS
  user_ptopics = get_ptopics(U, N_PEER_TOPICS)
  ## mapper_ptopics_to_peers is a dict with (key, value) = (peer topics, user ids)
  mapper_ptopics_to_peers = get_ptopic_peers(user_ptopics)


  psize = [len(v) for k,v in peers.iteritems()]
  print '# grps={}, # users={}, min/max grp size={}, {}'.format( \
    len(peers), sum(psize), min(psize), max(psize) )
  #print 'most common size of groups:\n{}'.format(Counter(psize).most_common())

  return grps, peers, U_ranks

def get_rec_topic_per_user(cur_user, U_ranks, grps, peers):
  cur_grp = grps[cur_user]
  cur_top_topics = cur_grp.replace(r'[', '').replace(']', '').split(', ')

  ## if no latent featues to define peers for a given user
  ## define peers as all users
  if cur_top_topics==['']: 
    cur_peers = U_ranks.index
    cur_peers = cur_peers[cur_peers!=cur_user]
    U_new_ranks = U_ranks.ix[cur_peers,:]
    cur_avg_rank = U_new_ranks.apply(np.mean, axis=0)

  else:
    cur_top_topics = map(int, cur_top_topics)
    cur_peers = peers[cur_grp]
    cur_peers = cur_peers[cur_peers!=cur_user]
    U_fav_ranks = U_ranks.ix[cur_peers, cur_top_topics]
    U_new_ranks = U_ranks.ix[cur_peers,:].copy().drop(cur_top_topics, axis=1)
    cur_avg_rank = U_new_ranks.apply(np.mean, axis=0)
  
  rec_topic = cur_avg_rank.index[ cur_avg_rank.argsort().iloc[0] ]
  return rec_topic


def get_rec_topic(U_ranks, grps, peers):
  t1 = print_time('Getting recommended topic')
  print '\nFinding one candidate topic based on peers'
  users = pd.Series( U_ranks.index )
  U_rtopic = users.apply( 
    ftPartial(get_rec_topic_per_user, U_ranks=U_ranks, grps=grps, peers=peers))

  U_rtopic.index = U_ranks.index
  t2 = print_time('Getting recommended topic', t1)
  print 'U_rtopic.shape = {}'.format(U_rtopic.shape)
  return U_rtopic

def get_top_talks_per_topic(f_vals, n_top):
  ''' f_vals (say, average tfidf): list | f_nms (say, vocab): list
  return top f_nms based on f_vals (i.e. show top features)
  '''
  top_idx = np.argsort(f_vals)[::-1][:n_top]
  return top_idx


def get_talks_candidates(V, N_TALK_CANDIDATES=5):
  print '\nFinding talk candidates based on the candidate topic'
  tmpf = ftPartial(get_top_talks_per_topic, n_top=N_TALK_CANDIDATES)
  top_talks_idx = V.apply(tmpf, axis=1)
  top_talk_ids = V.columns[ top_talks_idx.values ]
  return top_talk_ids

def get_rec_talk_per_user(uid, rdf, tdf, rec_topics, talk_candidates):

  rating_cols = ['Beautiful', 'Confusing', 'Courageous', 'Fascinating', \
    'Funny','Informative', 'Ingenious', 'Inspiring', 'Jaw-dropping', \
    'Longwinded', 'OK', 'Obnoxious', 'Persuasive', 'Unconvincing', 'tid']
  fav_tids = rdf.ix[rdf['uid_idiap']==uid, 'tid']
  fav_ratings = tdf.ix[ tdf['tid'].isin(fav_tids), rating_cols]
  fav_ratings = fav_ratings.set_index('tid')

  candidate_tids = talk_candidates[rec_topics[uid]].astype(int)
  candidate_ratings = tdf.ix[ tdf['tid'].isin(candidate_tids), rating_cols]
  candidate_ratings = candidate_ratings.set_index('tid')

  tmpf = ftPartial(get_rating_MSE, fav_ratings=fav_ratings)
  candidate_dists = candidate_ratings.apply( tmpf, axis=1)
  rec_tid = candidate_dists.argsort().index[0]

  #info_cols = ['speaker', 'title', 'ted_event', 'keywords', 'related_themes']
  #fav_talks = tdf.ix[ tdf['tid'].isin(fav_tids), info_cols]
  #rec_talk = tdf.ix[ tdf['tid']==rec_tid, info_cols]
  #print '\nuser {}\n=====fav_talks=====\n{}\n\n======rec_talk=====\n{}'.format(\
  #  uid, fav_talks, rec_talk)

  return rec_tid

def get_rating_MSE(talk_rating, fav_ratings):
  dists = (talk_rating - fav_ratings)**2
  dists = np.apply_along_axis(np.sum, 1, dists)
  mean_dist = np.mean(dists)
  return mean_dist

def get_rec_talk(rec_topics, talk_candidates, tdf, rdf):
  print '\nPicking one recommended talk based on the talk candidates and ratings'
  uids = pd.Series(rec_topics.index)
  tmpf = ftPartial(get_rec_talk_per_user, rdf=rdf, tdf=tdf,
    rec_topics=rec_topics, talk_candidates=talk_candidates)
  rec_tids = uids.apply(tmpf)
  rec_tids.index = uids

  return rec_tids

def print_time(msg, t1=None):
  t2 = time()

  t2_str = datetime.fromtimestamp(t2).strftime('%Y/%m/%d %H:%M:%S')
  if t1 is None:
    print '{}: {}'.format(t2_str, msg)
  else:
    print '{}: {} <== {:.0f} secs'.format(t2_str, msg, t2-t1)
  return t2

  

#if __name__ == '__main__':
if False:
  N_TOTAL_TOPICS = 10
  N_PEER_TOPICS = 2
  N_TALK_CANDIDATES = 5
  
  IS_RUN_ALL = False
  
  talk_info_filename = '/Users/liviachang/Galvanize/capstone/data/talks_info_merged.csv'
  user_rating_filename = '/Users/liviachang/Galvanize/capstone/data/users_info_transformed.csv'
  rating_matrix_filename = ['/Users/liviachang/Galvanize/capstone/data/rating_matrix_small.csv',
    '/Users/liviachang/Galvanize/capstone/data/rating_matrix.csv'][IS_RUN_ALL]

  U, V = build_nmf(rating_matrix_filename, N_TOTAL_TOPICS)
  U_ranks, U_ptopics = get_ptopics(U, N_PEER_TOPICS)
  ptopic_to_peers = get_ptopic_peers(U_ptopics)
  U_rtopic = get_rec_topic(U_ranks, U_ptopics, ptopic_to_peers) ## shape=(n_users,)

  #lf_grps, peers, U_ranks = find_peers(U, N_PEER_TOPICS)

  #t3 = print_time('Time to find peers', t2)

  #rec_topics = get_rec_topic(U_ranks, U_ptopics, ptopic_to_peers) ## shape=(n_users,)
  
  t4 = print_time('Time to find topic from peers', t3)

  talk_candidates = get_talks_candidates(V, N_TALK_CANDIDATES) ## shape=(k, N_TALK_CANDIDATES)
  
  t5 = print_time('Time to find talks from topic', t4)
  
  tdf = pd.read_csv(talk_info_filename)
  rdf = pd.read_csv(user_rating_filename)
  rec_tids = get_rec_talk(rec_topics, talk_candidates, tdf, rdf)

  info_cols = ['tid', 'speaker', 'title', 'ted_event', 'keywords', 'related_themes']
  rec_talks = rec_tids.to_frame(name='tid').reset_index()
  rec_talks = pd.merge( rec_talks, tdf[info_cols], how='left', on='tid')
  rec_talks = rec_talks.set_index('uid_idiap')
  
  t6 = print_time('Time to pick one talk from candidates', t5)

  rec_broader_fn = '/Users/liviachang/Galvanize/capstone/model/rec_talks_broader.pkl'
  with open(rec_broader_fn, 'wb') as f:
    pickle.dump( rec_talks, f)


