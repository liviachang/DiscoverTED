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
#import graphlab as gl
#from mf import matrix_factorization

def load_rating_data(IS_RUN_ALL):
  print '\nLoading the rating data'
  if IS_RUN_ALL:
    R_fn = '/Users/liviachang/Galvanize/capstone/data/rating_matrix.csv'
  else:
    R_fn = '/Users/liviachang/Galvanize/capstone/data/rating_matrix_small.csv'

  ## load the data
  R = pd.read_csv(R_fn)
  R = R.set_index('uid_idiap')

  print 'R.shape={}'.format(R.shape)
  return R

def build_nmf(R, k, IS_RUN_ALL):
  print '\nBuilding NMF models and Finding k latent features'

  nmf = NMF(n_components = k)
  nmf.fit(R.values)

  U = nmf.transform(R.values)
  V = nmf.components_
  E = nmf.reconstruction_err_
  
  U = pd.DataFrame(U, index=R.index)
  V = pd.DataFrame(V, columns=R.columns)

  print 'R.shape={}, U.shape={}, V.shape={}, nmf_err={:.4f}'.format(\
    R.shape, U.shape, V.shape, E)
  
  ## save the model (i.e. U and V matrix)
  if IS_RUN_ALL:
    model_fn = '/Users/liviachang/Galvanize/capstone/model/modelUV{}.pkl'.format(k)
  else:
    model_fn = '/Users/liviachang/Galvanize/capstone/model/modelUV{}_small.pkl'.format(k)
  with open(model_fn, 'wb') as f:
    pickle.dump( (U,V), f)

  return U, V, E


def load_nmf(k, IS_RUN_ALL):
  print '\nLoading NMF models and Finding k latent features'
  if IS_RUN_ALL:
    model_fn = '/Users/liviachang/Galvanize/capstone/model/modelUV{}.pkl'.format(k)
  else:
    model_fn = '/Users/liviachang/Galvanize/capstone/model/modelUV{}_small.pkl'.format(k)

  with open(model_fn) as f:
    U, V = pickle.load(f)
  return U, V


def get_latent_feature_rankings(x):
  n_topcis = len(x)
  n_nonzero = sum(x>0.)
  ranks = n_topcis - rankdata(x) + 1
  ranks = [tmp if tmp<=n_nonzero else np.nan for tmp in ranks]
  return ranks

def get_top_lf(x, N_LFS_PER_PEERGROUP):
  top_lfs = sorted(np.where(x.values<=N_LFS_PER_PEERGROUP)[0])
  return str(top_lfs)

def find_peers(U, N_LFS_PER_PEERGROUP=2):
  print '\nFinding peers based on latent features'
  ## get the rankings of the latent featues for each user based on U matrix
  U_ranks = U.apply(get_latent_feature_rankings, axis=1) # shape=(n_users, k)

  ## get the top n latent features for each user
  ## this is used to define peers
  tmpf = ftPartial(get_top_lf, N_LFS_PER_PEERGROUP=N_LFS_PER_PEERGROUP)
  grps = U_ranks.apply(tmpf, axis=1)

  ## find peers for each group of top n latent features
  ## group_users is a dictionary {top_lfs, users}
  peers = {} 
  for glf in grps.unique():
    user_idx = np.where( grps==glf )
    user_ids = grps.index[user_idx]
    peers[glf] = user_ids

  psize = [len(v) for k,v in peers.iteritems()]
  print '# grps={}, # users={}, min/max grp size={}, {}'.format( \
    len(peers), sum(psize), min(psize), max(psize) )
  #print 'most common size of groups:\n{}'.format(Counter(psize).most_common())

  return grps, peers, U_ranks

def get_rec_topic_per_user(cur_user, U_ranks, grps, peers):
  cur_grp = grps[cur_user]
  cur_top_lfs = cur_grp.replace(r'[', '').replace(']', '').split(', ')

  ## if no latent featues to define peers for a given user
  ## define peers as all users
  if cur_top_lfs==['']: 
    cur_peers = U_ranks.index
    cur_peers = cur_peers[cur_peers!=cur_user]
    U_new_ranks = U_ranks.ix[cur_peers,:]
    cur_avg_rank = U_new_ranks.apply(np.mean, axis=0)

  else:
    cur_top_lfs = map(int, cur_top_lfs)
    cur_peers = peers[cur_grp]
    cur_peers = cur_peers[cur_peers!=cur_user]
    U_fav_ranks = U_ranks.ix[cur_peers, cur_top_lfs]
    U_new_ranks = U_ranks.ix[cur_peers,:].copy().drop(cur_top_lfs, axis=1)
    cur_avg_rank = U_new_ranks.apply(np.mean, axis=0)
  
  rec_topic = cur_avg_rank.index[ cur_avg_rank.argsort().iloc[0] ]
  return rec_topic


def get_rec_topic(U_ranks, grps, peers):
  print '\nFinding one candidate topic based on peers'
  users = pd.Series( U_ranks.index )
  rec_topics = users.apply( 
    ftPartial(get_rec_topic_per_user, U_ranks=U_ranks, grps=grps, peers=peers))

  rec_topics.index = U_ranks.index
  return rec_topics

def get_top_talks_per_topic(f_vals, n_top):
  ''' f_vals (say, average tfidf): list | f_nms (say, vocab): list
  return top f_nms based on f_vals (i.e. show top features)
  '''
  top_idx = np.argsort(f_vals)[::-1][:n_top]
  return top_idx


def get_talks_candidates(V, N_TALKS_PER_LF=5):
  print '\nFinding talk candidates based on the candidate topic'
  tmpf = ftPartial(get_top_talks_per_topic, n_top=N_TALKS_PER_LF)
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
    print '{}: {} {:.0f} secs'.format(t2_str, msg, t2-t1)
  return t2

  

if __name__ == '__main__':

  t1 = print_time('Program start')

  IS_RUN_ALL = True
  R = load_rating_data(IS_RUN_ALL)

  N_TOPICS = 10
  IS_BUILD_NMF = True
  if IS_BUILD_NMF:
    U, V, E = build_nmf(R, N_TOPICS, IS_RUN_ALL)
  else:
    U, V = load_nmf(N_TOPICS, IS_RUN_ALL)

  t2 = print_time('Time to run NMF', t1)

  lf_grps, peers, U_ranks = find_peers(U, N_LFS_PER_PEERGROUP=2)

  t3 = print_time('Time to find peers', t2)

  rec_topics = get_rec_topic(U_ranks, lf_grps, peers) ## shape=(n_users,)
  
  t4 = print_time('Time to find topic from peers', t3)

  talk_candidates = get_talks_candidates(V, N_TALKS_PER_LF=5) ## shape=(k, N_TALKS_PER_LF)
  
  t5 = print_time('Time to find talks from topic', t4)
  
  tfn = '/Users/liviachang/Galvanize/capstone/data/talks_info_merged.csv'
  tdf = pd.read_csv(tfn)
  rfn = '/Users/liviachang/Galvanize/capstone/data/users_info_transformed.csv'
  rdf = pd.read_csv(rfn)
  rec_tids = get_rec_talk(rec_topics, talk_candidates, tdf, rdf)

  info_cols = ['tid', 'speaker', 'title', 'ted_event', 'keywords', 'related_themes']
  rec_talks = rec_tids.to_frame(name='tid').reset_index()
  rec_talks = pd.merge( rec_talks, tdf[info_cols], how='left', on='tid')
  rec_talks = rec_talks.set_index('uid_idiap')
  
  t6 = print_time('Time to pick one talk from candidates', t5)




