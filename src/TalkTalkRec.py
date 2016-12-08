from src.utils import *
from src.Recommender import Recommender
from src.TopicModelLDA import TopicModelLDA
from src.Talk import Talk
from src.User import NewUser, TestUsers

class TalkTalkRec(Recommender):
  def __init__(self, mdl, talk_ratings):
    super(TalkTalkRec, self).__init__(mdl, talk_ratings)

  def recommend(self, user, n_talks=2):
    ''' make recommendation based on a user's interested keywords and preferred
    talk types '''
    user_data = self._format_user_input(user)
    rtids = self._get_rtids_knn(user_data, tids=None, n_nbr=5, n_talks=n_talks)
    return rtids
  
if __name__ == '__main__':
  talks = Talk()
  
  with open(TOPIC_MODEL_LDA_FN) as f:
    mdlLDA = dill.load(f)

  ## create talk-talk recommender
  ttrec = TalkTalkRec(mdlLDA, talks.ratings)
  
  ## create new user for one-off testing
  uu = NewUser()
  ## make recommendations and print talk info for quality check
  tids_ttrec = ttrec.recommend(uu); map(talks.print_talk, tids_ttrec);

  ## run evaluation for all testing users
  test_users = TestUsers(talks)
  dists_rec, dists_bmk = ttrec.evaluate(test_users)

