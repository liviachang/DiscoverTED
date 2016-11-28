from src.utils import *
from src.Recommender import Recommender
from src.TopicModelLDA import TopicModelLDA
from src.Talk import Talk
from src.User import NewUser, TestUsers
from src.TalkTalkRec import TalkTalkRec
from src.UserUserRec import UserUserRec

class EnsembleRec(object):
  def __init__(self, mdls):
    self.mdls = mdls
    self.talks = self.mdls[0].talks

  def evaluate(self, test_users):
    import warnings
    warnings.filterwarnings("ignore")

    print ''
    print_time('Evaluating...')
    rec_dists = [] ## distance between rec. talks and fav talks
    bmk_dists = [] ## distance between randomly selected talks and favorite talks
    topic_dists = [] ## distribution between rec. topics
    for user in test_users.users:
      rtids = []
      for rec in self.mdls:
        rtids = np.concatenate([rtids, rec.recommend(user)])
      pdists = cdist(self.talks.ix[rtids,:], self.talks.ix[user.true_tids,:])
      rdists = np.apply_along_axis(min, 1, pdists)

      btids = self.talks.index.drop(np.append(user.input_tids, user.true_tids))
      btids = np.random.choice(btids, size=len(rtids))
      bdists = cdist(self.talks.ix[btids,:], self.talks.ix[user.true_tids,:])
      bdists = np.apply_along_axis(min, 1, bdists)
      
      topic_dist = np.apply_along_axis(np.argmin, 0, pdists)
      topic_dist = 1. * np.histogram( topic_dist, bins=len(rtids) )[0] / pdists.shape[1]
      print topic_dist

      rec_dists.append( rdists.mean() )
      bmk_dists.append( bdists.mean() )
      topic_dists.append( topic_dist )

    rec_dists = np.array(rec_dists)
    bmk_dists = np.array(bmk_dists)
    topic_dists = np.array(topic_dists)

    print_time('Evaluation Result...')
    print 'rec dists = {:.4f}, bmk dists = {:.4f}'.format(np.mean(rec_dists), np.mean(bmk_dists))
    print 'diff (rec-bmk) = {:.4f}, pvalue = {:.4f}'.format(
      np.mean(rec_dists-bmk_dists), ttest_1samp(rec_dists-bmk_dists,0).pvalue )

    return rec_dists, bmk_dists, topic_dists
    

if __name__ == '__main__':
  talks = Talk()
  
  with open(TOPIC_MODEL_LDA_FN) as f:
    mdlLDA = dill.load(f)

  ttrec = TalkTalkRec(mdlLDA, talks.ratings)
  uurec = UserUserRec(mdlLDA, talks.ratings)
  recs = [ttrec, uurec]
  enrec = EnsembleRec(recs)

  test_users = TestUsers(talks)
  dists_rec, dists_bmk, dist_topics = enrec.evaluate(test_users)
