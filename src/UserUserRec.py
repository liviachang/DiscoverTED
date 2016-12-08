from src.utils import *
from src.Recommender import Recommender
from src.TopicModelLDA import TopicModelLDA
from src.Talk import Talk
from src.User import NewUser, TestUsers

class UserUserRec(Recommender):
  def __init__(self, mdl, talk_ratings):
    super(UserUserRec, self).__init__(mdl, talk_ratings)

    ## for each user, find nearest 20 users as the peers
    self.N_PEER_USERS = 20

    user_mat = pd.read_csv(RATING_MATRIX_FN)
    user_mat = user_mat.set_index('uid_idiap')
    user_mat = user_mat.drop_duplicates()

    talk_tscores = mdl.talk_tscores.ix[user_mat.columns, :mdl.n_total_topics]
    users = user_mat.dot(talk_tscores)
    self.users = users.apply(lambda x: x/np.sum(x), axis=1)

  def recommend(self, user, n_topics=2, n_talks=1, include_deeper=False):
    ''' based on the user's inputs, model user to groups and recommen talks
    based on both deeper and wider topics '''

    ## number of deeper topics
    N_COMMON_TOPICS = 2
    user_data = self._format_user_input(user)
    tscores = user_data.ix[0,len(RATING_TYPES):]
    
    deeper_topics = ['topic{:02d}'.format(tmp) for tmp in tscores.argsort()[::-1][:N_COMMON_TOPICS]]
    deeper_talks = self.mdl.topic_reps[deeper_topics]
    func_knn = ftPartial(self._get_rtids_knn, user_data=user_data, n_nbr=5, n_talks=1)

    nbrs = NearestNeighbors(self.N_PEER_USERS).fit(self.users)
    peer_uids = self.users.index[nbrs.kneighbors(tscores.values)[1][0]]
    peer_tscores = self.users.ix[peer_uids,:].apply(np.mean, axis=0)
    wider_topics = peer_tscores.drop(deeper_topics).argsort()[::-1][:n_topics]
    wider_topics = ['topic{:02d}'.format(tmp) for tmp in wider_topics]
    wider_talks = self.mdl.topic_reps[wider_topics]
    func_knn = ftPartial(self._get_rtids_knn, user_data=user_data, n_nbr=5, n_talks=1)
    wider_rtids = wider_talks.apply(lambda x: func_knn(tids=x))
    wider_rtids = reduce(lambda x,y: np.append(x,y), wider_rtids)

    if include_deeper:
      deeper_rtids = deeper_talks.apply(lambda x: func_knn(tids=x))
      deeper_rtids = reduce(lambda x,y: np.append(x,y), deeper_rtids)
      rtids = np.unique(np.append(deeper_rtids, wider_rtids))
    else:
      rtids = np.array(wider_rtids)

    return rtids
    
if __name__ == '__main__':
  talks = Talk()
  
  with open(TOPIC_MODEL_LDA_FN) as f:
    mdlLDA = dill.load(f)

  uurec = UserUserRec(mdlLDA, talks.ratings)
  
  uu = NewUser()
  rtids_uurec = uurec.recommend(uu); map(talks.print_talk, rtids_uurec)
  
  test_users = TestUsers(talks)
  dists_rec, dists_bmk = uurec.evaluate(test_users)

