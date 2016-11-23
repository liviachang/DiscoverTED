from src.utils import *

class User(object):
  def __init__(self, uid='New'):
    self.uid = uid
    self.text = None
    self.ratings = None

  def __str__(self):
    output = 'User: {}\n'.format(self.uid)
    output = output + textwrap.fill('Text: {}'.format(self.text), 100)
    output = output + '\nRating:\n{}\n'.format(self.ratings)
    return output

class TestUsers(object):
  def __init__(self, talks):
    np.random.seed(0)
    test_udf = pd.read_csv(TEST_USER_TALK_FN)
    test_udf['tid'] = test_udf['tid'].astype(int)
    test_uids = test_udf['uid_idiap'].unique().tolist()

    users = []
    for uid in test_uids:
      cur_user = User(uid)

      tids = map(str, test_udf.ix[test_udf['uid_idiap']==uid, 'tid'] )
      cur_user.input_tids = np.random.choice(tids, 2, replace=False)
      cur_user.true_tids = [x for x in tids if x not in cur_user.input_tids]

      cur_user.text = talks.get_text(cur_user.input_tids)
      cur_user.ratings = talks.ratings.ix[cur_user.input_tids]
      users.append(cur_user)
    self.users = users

class NewUser(User):
  def __init__(self, uid='New'):
    self.text = self._get_user_text()
    self.ratings = self._get_user_ratings()
    self.uid = uid

  def _get_user_text(self):
    default_text = 'machine learning big data artificial intelligence invest get rich finance economics'
    user_text = raw_input('\nTopics you are interested in: (say, \'data science, finance\'):  ')
    user_text = default_text if user_text=='' else user_text
    print user_text
    return user_text

  def _get_user_ratings(self):
    rating_dict = self._print_rating_menu()

    default_rating_idx = '5,7'

    user_rating_idx = raw_input('\nTypes you are interested in: ' + \
      '(say, \'5,7\' for \'Informative+Inspiring\'):  ')
    user_rating_idx = default_rating_idx if user_rating_idx=='' else user_rating_idx
    user_rating_idx = map(int, user_rating_idx.replace(' ', '').split(','))
    print ', '.join([rating_dict[x] for x in user_rating_idx])
  
    rtyp_combs = user_rating_idx
    for comb_len in xrange(2, len(user_rating_idx)+1):
      cur_combs = list(combinations(user_rating_idx, comb_len))
      rtyp_combs = rtyp_combs + cur_combs

    U_fratings = []
    for rcomb in rtyp_combs:
      U_fratings.append(self._get_fratings_per_rtypes(rcomb))

    U_fratings = pd.DataFrame(U_fratings, columns=RATING_TYPES)
    return U_fratings

  def _get_fratings_per_rtypes(self, rcomb):
    U_frating = np.repeat(0., len(RATING_TYPES))

    if isinstance(rcomb, int):
      U_frating[rcomb] = 1.
    else:
      for ridx in rcomb:
        U_frating[ridx] = 1. / len(rcomb)

    return U_frating

  def _print_rating_menu(self):
    ''' Print all rating options for users to choose from '''
    rating_idx = range(len(RATING_TYPES) )
    rating_dict = dict(zip(rating_idx, RATING_TYPES) )

    menu = '\n'
    for (idx, rating) in rating_dict.iteritems():
      menu += '{0: >2d}: '.format(idx)
      menu += '{0: <12} | '.format(rating)
      if idx % 4 == 3:
        menu = menu + '\n'
    print menu
    return rating_dict


if __name__ == '__main__':
  #x = NewUser()

  from src.Talk import Talk
  talks = Talk()
  test_users = TestUsers(talks)
