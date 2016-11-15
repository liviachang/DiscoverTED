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

def load_rating_data(rating_fn):
  t1 = print_time('Loading the rating data')

  ## load the data
  R = pd.read_csv(rating_fn)
  R = R.set_index('uid_idiap')

  t2 = print_time('Loading the rating data', t1)
  print 'R.shape={}\n'.format(R.shape)

  return R

def build_nmf(rating_fn, k, IS_LOAD_UV=False):
  ## get model filename
  model_fn = rating_fn.replace('/data/rating_matrix', '/model/UV_matrix')

  if IS_LOAD_UV and os.path.isfile(model_fn):
    t1 = print_time('Loading NMF models with k={}'.format(k))
    with open(model_fn) as f:
      U, V = pickle.load(f)
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
  print 'U.shape={}, V.shape={}\n'.format(U.shape, V.shape)
  
  return U, V

def get_topic_rankings_per_user(x):
  n_topcis = len(x)
  n_nonzero = sum(x>0.)
  ranks = n_topcis - rankdata(x) + 1
  ranks = [tmp if tmp<=n_nonzero else np.nan for tmp in ranks]
  return ranks

def get_user_gtopics_per_user(x, N_PEER_TOPICS):
  top_topics = sorted(np.where(x.values<=N_PEER_TOPICS)[0])
  return str(top_topics)

def get_user_gtopics(U, N_PEER_TOPICS):
  t1 = print_time('Getting users\' top {} topics as group topics'.format(N_PEER_TOPICS))
  ## get the rankings of the topcis (i.e. latent featues) for each user based on U matrix
  ## U_ranks.shape = n_users x k_topics
  U_ranks = U.apply(get_topic_rankings_per_user, axis=1)
  
  ## get the top N_PEER_TOPICS topics for each user
  ## this is used to define groups
  tmpf = ftPartial(get_user_gtopics_per_user, N_PEER_TOPICS=N_PEER_TOPICS)
  U_gtopics = U_ranks.apply(tmpf, axis=1)
  U_gtopics = U_gtopics.to_dict()

  t2 = print_time('Getting users\' top {} topics as group topics'.format(N_PEER_TOPICS), t1)
  print 'U_gtopics: #user = {}, # gtopics = {}\n'.format(len(U_gtopics), \
    len(U_gtopics.values()[0].split() ))

  return U_ranks, U_gtopics

def get_group_users(U_gtopics):
  t1 = print_time('Getting groups\' users based on gtopics')

  u_gts = pd.Series( U_gtopics, name='gtopics' )
  ## find group users for each group with top N_PEER_TOPICS group topics
  ## groups is a dictionary {group_topics, user_ids}
  groups = {} 
  for group_topic in u_gts.unique():
    user_ids = u_gts.index[u_gts==group_topic]
    groups[group_topic] = user_ids

  t2 = print_time('Getting groups\' users based on gtopics', t1)
  
  psize = [len(v) for k,v in groups.iteritems()]
  print '# grps={}, # total users={}, (min, max) grp size=({}, {})\n'.format( \
    len(groups), sum(psize), min(psize), max(psize) )

  return groups

def get_group_rtopics_per_group(target_user_gtopics, G_users, U_ranks, N_REC_TOPICS):
  
  ## if no latent featues to define groups for a given user
  ## define groups as all users
  if target_user_gtopics == '[]':
    target_U_ranks = U_ranks

  else:
    target_users = G_users[target_user_gtopics]

    gtopics = target_user_gtopics.replace(r'[', '').replace(']', '').split(', ')
    gtopics = map(int, gtopics)
    target_U_ranks = U_ranks.ix[target_users,:].copy().drop(gtopics, axis=1)

    ## if all rankings are NaN
    if target_U_ranks.isnull().sum().sum() == np.product(target_U_ranks.shape):
      target_U_ranks = U_ranks.copy().drop(gtopics, axis=1)
  
  tcandidates_avg_rank = target_U_ranks.apply(np.mean, axis=0)
  rec_topics_all = tcandidates_avg_rank.index[tcandidates_avg_rank.argsort()]
  rec_topics = sorted(rec_topics_all[:N_REC_TOPICS])
  return rec_topics

def get_group_rtopics(G_users, U_ranks, N_REC_TOPICS):
  t1 = print_time('Getting groups\' {} recommended topics'.format(N_REC_TOPICS))

  G_rtopics = {}
  for gtopics in G_users.iterkeys():
    rtopics = get_group_rtopics_per_group(gtopics, G_users, U_ranks, N_REC_TOPICS)
    G_rtopics[gtopics] = rtopics

  t2 = print_time('Getting groups\' {} recommended topics'.format(N_REC_TOPICS), t1)
  print '# grps={}, # rec topics={}\n'.format(len(G_rtopics), N_REC_TOPICS)

  return G_rtopics

def get_user_rtopics(U_gtopics, G_rtopics):
  t1 = print_time('Getting users\' recommended topics')
 
  U_rtopics = {}
  for (uid, gtopics) in U_gtopics.iteritems():
    rtopics = G_rtopics[gtopics]
    U_rtopics[uid] = rtopics
  
  t2 = print_time('Getting users\' recommended topics', t1)

  return U_rtopics

#def get_group_rtopics_per_group(cur_user, U_ranks, grps, groups): ## STOP HERE, add flag
#  cur_grp = grps[cur_user]
#  cur_top_topics = cur_grp.replace(r'[', '').replace(']', '').split(', ')
#
#  ## if no latent featues to define groups for a given user
#  ## define groups as all users
#  if cur_top_topics==['']: 
#    cur_groups = U_ranks.index
#    cur_groups = cur_groups[cur_groups!=cur_user]
#    U_new_ranks = U_ranks.ix[cur_groups,:]
#    cur_avg_rank = U_new_ranks.apply(np.mean, axis=0)
#
#  else:
#    cur_top_topics = map(int, cur_top_topics)
#    cur_groups = groups[cur_grp]
#    cur_groups = cur_groups[cur_groups!=cur_user]
#    U_fav_ranks = U_ranks.ix[cur_groups, cur_top_topics]
#    U_new_ranks = U_ranks.ix[cur_groups,:].copy().drop(cur_top_topics, axis=1)
#    cur_avg_rank = U_new_ranks.apply(np.mean, axis=0)
#  
#  rec_topic = cur_avg_rank.index[ cur_avg_rank.argsort().iloc[0] ]
#  return rec_topic


def get_topic_talks_per_topic(f_vals, n_top):
  ''' f_vals (say, average tfidf): list | f_nms (say, vocab): list
  return top f_nms based on f_vals (i.e. show top features)
  '''
  top_idx = np.argsort(f_vals)[::-1][:n_top]
  return top_idx


def get_topic_talks(V, N_TALK_CANDIDATES=5):
  t1 = print_time('Getting topics\' top {} talks'.format(N_TALK_CANDIDATES))

  tmpf = ftPartial(get_topic_talks_per_topic, n_top=N_TALK_CANDIDATES)
  top_talks_idx = V.apply(tmpf, axis=1)
  top_talk_ids = V.columns[ top_talks_idx.values ]
  
  t2 = print_time('Getting topics\' top {} talks'.format(N_TALK_CANDIDATES), t1)

  return top_talk_ids

def get_user_fav_ratings(udf, tdf):
  t1 = print_time('Getting ratings for users\' favorite talks')

  U_ftalks = udf[['uid_idiap', 'tid']].groupby('uid_idiap').tid.apply(list)
  U_ftalks = U_ftalks.to_dict()

  U_fratings = {}
  for (uid, tids) in U_ftalks.iteritems():
    U_fratings[uid] = tdf.ix[tids]
  
  t2 = print_time('Getting ratings for users\' favorite talks', t1)
  
  return U_fratings

def get_closest_rtalk(talk_ratings, fav_ratings, OPTION=['MEAN_DIST', 'MIN_DIST'][1]):
  fr = fav_ratings.values
  dists = {}
  for (tid, tr) in zip(talk_ratings.index, talk_ratings.values):
    dists_to_each_fr = np.sum( (tr-fr)**2, axis=1)
    if OPTION=='MEAN_DIST':
      dists[tid] = dists_to_each_fr.mean()
    elif OPTION=='MIN_DIST':
      dists[tid] = dists_to_each_fr.min()

  rtalk = sorted(dists, key=dists.get)[0]
  return rtalk


def get_user_rec_talks_per_user(fratings, rtopics_talks, TK_ratings):
  rtalks = []
  for rtt in rtopics_talks:
    tratings = TK_ratings.ix[rtt,:]
    rtalks.append( get_closest_rtalk(tratings, fratings) )

  return rtalks

def get_user_rec_talks(U_fratings, U_rtopics, TP_talks, TK_ratings):

  t1 = print_time('Getting users\' recommended talks')

  U_rtalks = {}

  for (uid, fratings) in U_fratings.iteritems():
    rtopics = U_rtopics[uid]
    rtopics_talks = TP_talks[rtopics]
    U_rtalks[uid] = get_user_rec_talks_per_user( fratings, rtopics_talks, TK_ratings)
  
  t2 = print_time('Getting users\' recommended talks', t1)

  return U_rtalks
  
def print_time(msg, t1=None):
  t2 = time()

  t2_str = datetime.fromtimestamp(t2).strftime('%Y/%m/%d %H:%M:%S')
  if t1 is None:
    print '{}: {}'.format(t2_str, msg)
  else:
    print '{}: {} <== {:.0f} secs'.format(t2_str, msg, t2-t1)
  return t2

if __name__ == '__main__':
  N_TOTAL_TOPICS = 10
  N_PEER_TOPICS = 2
  N_REC_TOPICS = 2
  N_TALK_CANDIDATES = 5
  
  IS_RUN_ALL = True
  
  talk_info_filename = '/Users/liviachang/Galvanize/capstone/data/talks_info_merged.csv'
  user_rating_filename = '/Users/liviachang/Galvanize/capstone/data/users_info_transformed.csv'
  rating_matrix_filename = ['/Users/liviachang/Galvanize/capstone/data/rating_matrix_small.csv',
    '/Users/liviachang/Galvanize/capstone/data/rating_matrix.csv'][IS_RUN_ALL]
  
  rating_cols = ['Beautiful', 'Confusing', 'Courageous', 'Fascinating', \
    'Funny','Informative', 'Ingenious', 'Inspiring', 'Jaw-dropping', \
    'Longwinded', 'OK', 'Obnoxious', 'Persuasive', 'Unconvincing']
  info_cols = ['speaker', 'title', 'ted_event', 'keywords', 'related_themes']
  
  talk_df_orig = pd.read_csv(talk_info_filename)
  TK_ratings = talk_df_orig.copy()
  TK_ratings.tid = TK_ratings.tid.astype(str)
  TK_ratings = TK_ratings.set_index('tid')
  TK_ratings = TK_ratings.ix[:,rating_cols]
  
  TK_info = talk_df_orig.copy()
  TK_info.tid = TK_info.tid.astype(str)
  TK_info = TK_info.set_index('tid')
  TK_info = TK_info.ix[:, info_cols]

  user_ftalk_df_orig = pd.read_csv(user_rating_filename)
  user_ftalk_df = user_ftalk_df_orig.copy()
  user_ftalk_df.tid = user_ftalk_df.tid.astype(int).astype(str)

  U, V = build_nmf(rating_matrix_filename, N_TOTAL_TOPICS)
  U_ranks, U_gtopics = get_user_gtopics(U, N_PEER_TOPICS)
  G_users = get_group_users(U_gtopics)
  G_rtopics = get_group_rtopics(G_users, U_ranks, N_REC_TOPICS)
  U_rtopics = get_user_rtopics(U_gtopics, G_rtopics)
  TP_talks = get_topic_talks(V, N_TALK_CANDIDATES)
  U_fratings = get_user_fav_ratings(user_ftalk_df, TK_ratings)
  U_rtalks = get_user_rec_talks(U_fratings, U_rtopics, TP_talks, TK_ratings)

