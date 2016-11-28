from src.utils import *
from src.TopicModelLDA import TopicModelLDA

class Recommender(object):
  def __init__(self, mdl, talk_ratings):
    ## randomly pick one of the closest 5 talks to recommend
    self.N_CANDIDATES=5
    
    self.mdl = mdl
    
    talk_tscores = self._adjust_tscores(mdl.talk_tscores.ix[:, :mdl.n_total_topics])
    self.talks = pd.concat([talk_ratings, talk_tscores], axis=1)

  def _adjust_tscores(self, tscores):
    return tscores*2 # weight more for topic scores than rating types

  def _format_user_input(self, user):
    tscores = self.mdl.transform(user.text)[:self.mdl.n_total_topics]
    tscores = self._adjust_tscores(tscores)
    tscores_df = pd.DataFrame([tscores] * user.ratings.shape[0])
    user_data = pd.concat([user.ratings.reset_index(drop=True), tscores_df], axis=1)
    return user_data

  def evaluate(self, test_users):
    import warnings
    warnings.filterwarnings("ignore")

    print ''
    print_time('Evaluating...')
    rec_dists = []
    bmk_dists = []
    for user in test_users.users:
      rtids = self.recommend(user)
      rdists = cdist(self.talks.ix[rtids,:], self.talks.ix[user.true_tids,:])
      rdists = np.apply_along_axis(min, 1, rdists)

      btids = self.talks.index.drop(np.append(user.input_tids, user.true_tids))
      btids = np.random.choice(btids, size=len(rtids))
      bdists = cdist(self.talks.ix[btids,:], self.talks.ix[user.true_tids,:])
      bdists = np.apply_along_axis(min, 1, bdists)

      rec_dists.append( rdists.mean() )
      bmk_dists.append( bdists.mean() )

    rec_dists = np.array(rec_dists)
    bmk_dists = np.array(bmk_dists)

    print_time('Evaluation Result...')
    print 'rec dists = {:.4f}, bmk dists = {:.4f}'.format(np.mean(rec_dists), np.mean(bmk_dists))
    print 'diff (rec-bmk) = {:.4f}, pvalue = {:.4f}'.format(
      np.mean(rec_dists-bmk_dists), ttest_1samp(rec_dists-bmk_dists,0).pvalue )

    return rec_dists, bmk_dists
  
  def _get_rtids_knn(self, user_data, tids, n_nbr, n_talks):
    if tids is None:
      cur_talks = self.talks
    else:
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
    rtids = np.random.choice(rtalks.index[:self.N_CANDIDATES], size=n_talks, replace=False )
    return rtids
