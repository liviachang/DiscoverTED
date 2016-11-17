from src.utils import *
from src.model_talk_topics import tokenize_talk_doc, get_topics_from_tf, get_topic_score_names

token_mapper, lda_mdl = load_lda_model_data()
TK_topics, TP_info = load_lda_topics_data()
G_rtopics, U_tscores, U_ftalks = load_group_data()
TK_ratings, TK_info = load_talk_data()


def get_new_user_fav_ratings():
  input_rtyp_idx = get_user_rating_types()

  rtyp_combs = input_rtyp_idx
  for comb_len in xrange(2, len(input_rtyp_idx)+1):
    cur_combs = list(combinations(input_rtyp_idx, comb_len))
    rtyp_combs = rtyp_combs + cur_combs

  U_fratings = []
  for rcomb in rtyp_combs:
    U_fratings.append(get_fratings_per_rtypes(rcomb))

  U_fratings = pd.DataFrame(U_fratings, columns=RATING_TYPES)
  return U_fratings

def get_user_rating_types():
  rtyp_idx = range(len(RATING_TYPES) )
  rtyp_dict = dict(zip(rtyp_idx, RATING_TYPES) )

  for (idx, rtyp) in rtyp_dict.iteritems():
      print '{}: {}'.format(idx, rtyp)

  input_rtyp_idx = raw_input('\nTypes you are interested in: ' + \
    '(say, \'4,5\' for \'Funny+Informative\'):  ')
  if input_rtyp_idx == '':
    input_rtyp_idx = '4,5'
  input_rtyp_idx = map(int, input_rtyp_idx.replace(' ', '').split(','))
  print ', '.join([rtyp_dict[x] for x in input_rtyp_idx])

  return input_rtyp_idx

def get_fratings_per_rtypes(rcomb):
  U_frating = np.repeat(0., len(RATING_TYPES))

  if isinstance(rcomb, int):
    U_frating[rcomb] = 1.
  else:
    for ridx in rcomb:
      U_frating[ridx] = 1. / len(rcomb)

  return U_frating

def get_user_topic_keywords():
  user_text = raw_input('\nTopics you are interested in: (say, \'data science, finance\'):  ')
  if user_text == '':
    user_text = 'data science, technology, computer, economics, finance, market, investing'
  print user_text

  return user_text

#def vectorize_topics(V, TK_info, n_talks=N_TALKS_FOR_KWS): ##FIXME tokenize, remove stop words...
#  TP_talks = get_topic_talks(V, n_talks)
#  topic_docs = []
#  for topic_talks in TP_talks:
#    talk_texts = TK_info.ix[list(topic_talks), ['keywords', 'description'] ]
#    talk_texts = talk_texts.apply(lambda x: ' '.join(x.tolist()), axis=1)
#    topic_docs.append(' '.join(list(talk_texts)) )
#  
#  TP_vec = TfidfVectorizer(stop_words='english')
#  TP_tfidf = TP_vec.fit_transform(topic_docs).toarray()
#  TP_vocabs = np.array(TP_vec.get_feature_names())
#
#  TP_kws = get_topic_keywords(TP_tfidf, TP_vocabs)
#
#  return topic_docs, TP_vec, TP_tfidf, TP_vocabs, TP_kws
#
#def get_topic_keywords(TP_tfidf, vocabs, IS_PRINT=False):
#  TP_kws = np.apply_along_axis(lambda x: vocabs[x.argsort()[::-1][:3]], 1, TP_tfidf)
#
#  if IS_PRINT:
#    print TP_kws
#
#  return TP_kws
#
#def get_user_gtopics_from_keywords(kws, TP_vec, TP_tfidf, n_topics=N_GROUP_TOPICS):
#  U_tfidf = TP_vec.transform([kws])
#  cos_sim = linear_kernel(U_tfidf, TP_tfidf)[0]
#  gtopics = cos_sim.argsort()[::-1][:n_topics]
#  return gtopics

def get_topics_from_text(text):
  tknizer = RegexpTokenizer(r'\w+')
  stop_wds = get_stop_words('en')
  pstemmer = PorterStemmer()
  new_tokens = tokenize_talk_doc(text, tknizer, stop_wds, pstemmer)

  new_tf = token_mapper.doc2bow(new_tokens)
  topics = get_topics_from_tf(new_tf, lda_mdl)
  result = pd.Series(topics, index=get_topic_score_names())
  return result



def get_new_user_tscores_fratings():

  ## get user input text and rating preference
  user_text = get_user_topic_keywords()
  user_fratings = get_new_user_fav_ratings()
  
  ## convert the user input text to topic scores
  user_tscores = get_topics_from_text(user_text)

  return user_tscores, user_fratings

def get_user_rec_talks(tscores, user_fratings):

  gtopics_list = map(int, tscores[N_TOTAL_TOPICS:])
  gtopics_key = str(sorted(map(int, tscores[N_TOTAL_TOPICS:])))

  ## for topics to go deeper (topics already liked)
  deeper_rtopics = ['topic{:02d}'.format(x) for x in gtopics_list]
  deeper_candidates = TP_info.ix[deeper_rtopics, 'tids'].tolist()
  deeper_rtalks = get_rtalks_from_ratings(user_fratings, deeper_candidates)

  ## for topics to go broader (topics new to the user)
  broader_rtopics = G_rtopics[gtopics_key]
  broader_candidates = TP_info.ix[broader_rtopics, 'tids'].tolist()
  broader_rtalks = get_rtalks_from_ratings(user_fratings, broader_candidates)

  return deeper_rtalks + broader_rtalks
 
def get_rtalks_from_ratings(user_ratings, candidates):
  rtalks = []
  for rtt in candidates:
    tratings = TK_ratings.ix[rtt,:]
    rtalks.append( get_closest_rtalk(tratings, user_ratings) )
  return rtalks

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


def get_existing_user_tscores_fratings(uid):
  user_tscores = U_tscores.ix[uid,:]

  ##FIXME: should move to model_user_groups.py
  top_topics = user_tscores.argsort()[::-1][:N_GROUP_TOPICS] 
  for idx in xrange(N_GROUP_TOPICS):
    user_tscores['top_topic{}'.format(idx+1)] = top_topics[idx]
  
  tids = U_ftalks.ix[U_ftalks.uid_idiap==uid, 'tid']
  user_fratings = TK_ratings.ix[tids]

  return user_tscores, user_fratings


def rec_talks(uid):
  if uid.lower() =='n':
    tscores, fratings = get_new_user_tscores_fratings()
  else:
    if uid.lower() == 'e':
      uid = random.choice( U_tscores.index )
    tscores, fratings = get_existing_user_tscores_fratings(uid)

  rec_tids = get_user_rec_talks(tscores, fratings)
  print_rtalks(rec_tids)


def print_rtalks(rec_tids):
  LINE_LENGTH = 80

  for rtid in rec_tids:
    tt = TK_info.ix[rtid]

    tthemes = tt.related_themes
    msg = '\n====={}: {} ({})=====\n{}\n[keywords]\n{}'.format(\
        tt.speaker, tt.title, tt.ted_event, #rtid,
        textwrap.fill(tt.description, LINE_LENGTH), \
        textwrap.fill(tt.keywords.replace('[','').replace(']',''), LINE_LENGTH))
    if not isinstance(tthemes, float):
      msg = '{}\n[themes]\n{}'.format(msg, 
        re.sub('\[|\]|u\'|\'|\"|u\"', '', tthemes))

    print msg

  
if __name__ == '__main__':
  msg = '\nPlease enter your UserID, or "n" (for a new user), or "q" (for quit): '

  uid = raw_input(msg)
  while uid.lower() not in ['q', '']:
    rec_talks(uid)
    uid = raw_input(msg)



