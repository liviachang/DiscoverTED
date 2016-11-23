from src.utils import *
from src.TopicModelLDA import TopicModelLDA
from src.Talk import Talk
from src.User import NewUser

class UserUserRec(object):
  def __init__(self, mdl, talk_ratings):
    self.mdl = mdl

    ## randomly pick one of the closest 5 talks to recommend
    self.N_CANDIDATES = 5
    self.N_PEER_USERS = 20

    user_mat = pd.read_csv(RATING_MATRIX_FN)
    user_mat = user_mat.set_index('uid_idiap')
    user_mat = user_mat.drop_duplicates()

    talk_tscores = mdl.talk_tscores.ix[user_mat.columns, :mdl.n_total_topics]
    self.users = user_mat.dot(talk_tscores)

    talk_tscores = mdl.talk_tscores.ix[:, :mdl.n_total_topics] * 2
    self.talks = pd.concat([talk_ratings, talk_tscores*2], axis=1)

  def recommend(self, user, n_topics=2, n_talks=1):
    N_COMMON_TOPICS = 2
    tscores = self.mdl.transform(user.text)[:self.mdl.n_total_topics]
    tscores_df = pd.DataFrame([tscores*2] * user.ratings.shape[0])
    user_data = pd.concat([user.ratings, tscores_df], axis=1)

    deeper_topics = ['topic{:02d}'.format(tmp) for tmp in tscores.argsort()[::-1][:N_COMMON_TOPICS]]
    deeper_talks = self.mdl.topic_reps[deeper_topics]
    deeper_rtids = deeper_talks.apply(ftPartial(self._get_rtids_knn, user_data=user_data))
    deeper_rtids = reduce(lambda x,y: np.append(x,y), deeper_rtids)

    nbrs = NearestNeighbors(self.N_PEER_USERS).fit(self.users)
    peer_uids = self.users.index[nbrs.kneighbors(tscores.values)[1][0]]
    peer_tscores = self.users.ix[peer_uids,:].apply(np.mean, axis=0)
    wider_topics = peer_tscores.drop(deeper_topics).argsort()[::-1][:n_topics]
    wider_topics = ['topic{:02d}'.format(tmp) for tmp in wider_topics]
    wider_talks = self.mdl.topic_reps[wider_topics]
    wider_rtids = wider_talks.apply(ftPartial(self._get_rtids_knn, user_data=user_data))
    wider_rtids = reduce(lambda x,y: np.append(x,y), wider_rtids)

    rtids = np.unique(np.append(deeper_rtids, wider_rtids))
    return rtids
    
  def _get_rtids_knn(self, tids, user_data, n_nbr=3, n_talks=1):
    cur_talks = self.talks.ix[tids,]

    nbrs = NearestNeighbors(n_nbr).fit(cur_talks.values)
    
    rtalks = []
    for udata in user_data.values:
      tmp = nbrs.kneighbors(udata)
      tmp = pd.Series(tmp[0][0], index=cur_talks.index[tmp[1][0]])
      rtalks.append(tmp)

    if len(rtalks)==1:
      rtalks = rtalks[0]
    else:
      rtalks = reduce(lambda x1,x2: pd.concat([x1,x2], axis=1), rtalks).apply(np.sum, axis=1)
    rtalks = rtalks.sort_values(ascending=False)
    rtid = np.random.choice(rtalks.index[:self.N_CANDIDATES], size=n_talks, replace=False )
    return rtid


if __name__ == '__main__':
  talks = Talk()
  mdlLDA = dill.load(open(TOPIC_MODEL_LDA_FN))
  uurec = UserUserRec(mdlLDA, talks.ratings)
  
  uu = NewUser()
  rtids_uurec = uurec.recommend(uu)
  map(talks.print_talk, rtids_uurec)

