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

  def evaluate(self, test_users, n_talks=None):
    import warnings
    warnings.filterwarnings("ignore")

    print ''
    print_time('Evaluating {}'.format(type(self).__name__))
    
    ## the goal is to find whether distance of the rec. talk is closer to fav talk?
    rec_dists = [] ## distance between rec. talks and favorite talks
    bmk_dists = [] ## distance between randomly selected talks and favorite talks

    ## the goal is to find whether the fav. talk can be better captured if including  wider topics
    fav_dists = []

    ## the goal is to answer how many fav. talks are closer to deeper vs. wider topics
    ## distribution of favorite talks to closest rec. talks
    topic_dists = [] 

    for user in test_users.users:
      rtids = []
      for rec in self.mdls:
        if n_talks is None:
          rtids = np.concatenate([rtids, rec.recommend(user)])
        else:
          rtids = np.concatenate([rtids, rec.recommend(user, n_talks=n_talks)])
      pdists = cdist(self.talks.ix[rtids,:], self.talks.ix[user.true_tids,:])
      rdists = np.apply_along_axis(min, 1, pdists)

      btids = self.talks.index.drop(np.append(user.input_tids, user.true_tids))
      btids = np.random.choice(btids, size=len(rtids))
      bdists = cdist(self.talks.ix[btids,:], self.talks.ix[user.true_tids,:])
      bdists = np.apply_along_axis(min, 1, bdists)
      
      fdists = np.apply_along_axis(min, 0, pdists)
      topic_dist = np.apply_along_axis(np.argmin, 0, pdists)
      topic_dist = 1. * np.histogram( topic_dist, bins=len(rtids) )[0] / pdists.shape[1]

      rec_dists.append( rdists.mean() )
      bmk_dists.append( bdists.mean() )
      fav_dists.append( fdists.mean() )
      topic_dists.append( topic_dist )

    rec_dists = np.array(rec_dists)
    bmk_dists = np.array(bmk_dists)
    fav_dists = np.array(fav_dists)
    topic_dists = np.array(topic_dists)

    print_time('Evaluation Result...')
    print 'rec dists = {:.4f}, bmk dists = {:.4f}'.format(np.mean(rec_dists), np.mean(bmk_dists))
    print 'diff (rec-bmk) = {:.4f}, pvalue = {:.4f}'.format(
      np.mean(rec_dists-bmk_dists), ttest_1samp(rec_dists-bmk_dists,0).pvalue )

    return rec_dists, bmk_dists, fav_dists, topic_dists
    

if __name__ == '__main__':
  talks = Talk()
  
  with open(TOPIC_MODEL_LDA_FN) as f:
    mdlLDA = dill.load(f)

  #uu = NewUser()

  ttrec = TalkTalkRec(mdlLDA, talks.ratings)
  uurec = UserUserRec(mdlLDA, talks.ratings)
  recs = [ttrec, uurec]
  enrec = EnsembleRec(recs)

  test_users = TestUsers(talks)
  ttdist = ttrec.evaluate(test_users, n_talks=4)

  dists_rec, dists_bmk, fav_dists, dist_topics = enrec.evaluate(test_users)
  avg_dtopics = np.apply_along_axis(np.mean, 0, dist_topics)
  print 'From deeper topics: {:.0f}%, From wider topics: {:.0f}%'.format(\
    sum(avg_dtopics[:2])*1e2, sum(avg_dtopics[2:])*1e2)


