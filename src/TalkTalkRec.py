from src.utils import *
from src.TopicModelLDA import TopicModelLDA
from src.Talk import Talk
from src.User import NewUser

class TalkTalkRec(object):
  def __init__(self, mdl, talk_ratings):
    self.mdl = mdl

    ## randomly pick one of the closest 5 talks to recommend
    self.N_CANDIDATES=5

    talk_tscores = mdl.talk_tscores.ix[:, :mdl.n_total_topics] * 2
    self.talks = pd.concat([talk_ratings, talk_tscores], axis=1)

  def recommend(self, user, n_talks=1):
    tscores = self.mdl.transform(user.text)[:self.mdl.n_total_topics]
    tscores_df = pd.DataFrame([tscores*2] * user.ratings.shape[0])
    user_data = pd.concat([user.ratings, tscores_df], axis=1)
    rtids = self._get_rtids_knn(user_data)
    return rtids
  
  def _get_rtids_knn(self, user_data, n_nbr=5, n_talks=3):
    cur_talks = self.talks
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
    rtids = np.random.choice(rtalks.index[:self.N_CANDIDATES], size=n_talks, replace=False )
    return rtids
  
if __name__ == '__main__':
  talks = Talk()
  mdlLDA = dill.load(open(TOPIC_MODEL_LDA_FN))
  ttrec = TalkTalkRec(mdlLDA, talks.ratings)
  
  uu = NewUser()
  tids_ttrec = ttrec.recommend(uu)
  map(talks.print_talk, tids_ttrec)

