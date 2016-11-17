from src.utils import *
from src.model_talk_topics import tokenize_talk_doc, get_topics_from_tf, get_topic_score_names

def get_existing_user_fav_ratings(udf, tdf):
  t1 = print_time('Getting ratings for users\' favorite talks')

  U_ftalks = udf[['uid_idiap', 'tid']].groupby('uid_idiap').tid.apply(list)
  U_ftalks = U_ftalks.to_dict()

  U_fratings = {}
  for (uid, tids) in U_ftalks.iteritems():
    U_fratings[uid] = tdf.ix[tids]
  
  t2 = print_time('Getting ratings for users\' favorite talks', t1)
  
  return U_fratings

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

def get_topics_from_text(text, id2word, mdl):
  tknizer = RegexpTokenizer(r'\w+')
  stop_wds = get_stop_words('en')
  pstemmer = PorterStemmer()
  new_tokens = tokenize_talk_doc(text, tknizer, stop_wds, pstemmer)

  new_tf = id2word.doc2bow(new_tokens)
  topics = get_topics_from_tf(new_tf, mdl)
  result = pd.Series(topics, index=get_topic_score_names())
  return result


def rec_for_new_user():

  ## get user input text and rating preference
  user_text = get_user_topic_keywords()
  new_fratings = get_new_user_fav_ratings()
  
  token_mapper, lda = load_lda_model_data()
  TK_topics, TP_info = load_lda_topics_data()
  G_rtopics = load_group_data()

  new_tscores = get_topics_from_text(user_text, token_mapper, lda)
  gtopics = str(sorted(map(int, new_tscores[N_TOTAL_TOPICS:])))
  rtopics = G_rtopics[gtopics]
  rtopics_talks = TP_info.ix[rtopics, 'tids'].tolist() ## stop here

  ##===
  
#  with open(BROADER_MODEL_NEW_USERS_FN) as f:
#    V, G_rtopics = pickle.load(f)
#  TP_talks = get_topic_talks(V, N_TALK_CANDIDATES)
#
#  TP_docs, TP_vec, TP_tfidf, TP_vocabs, TP_kws = vectorize_topics(V, TK_info)
#  new_gtopics = get_user_gtopics_from_keywords(new_kws, TP_vec, TP_tfidf)
#  new_gtopics = str( sorted(new_gtopics) )
#  new_rtopics = G_rtopics[new_gtopics]
#  new_rtopics_talks = TP_talks[new_rtopics]
#  new_rtalks = get_user_rec_talks_per_user( new_fratings, new_rtopics_talks, TK_ratings)
#  return new_rtalks

def get_user_rec_talks_per_user(fratings, rtopics_talks, TK_ratings):
  rtalks = []
  for rtt in rtopics_talks:
    tratings = TK_ratings.ix[rtt,:]
    rtalks.append( get_closest_rtalk(tratings, fratings) )
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


def rec_for_existing_user(uid, TK_info):

  #U_fratings = get_existing_user_fav_ratings(U_ftalks, TK_ratings)
  #U_rtalks = get_user_rec_talks(U_fratings, U_rtopics, TP_talks, TK_ratings)
  
  with open(BROADER_MODEL_EXISTING_USERS_FN) as f:
    U_rtalks = pickle.load(f)
  new_rtalks = U_rtalks[uid]
  return new_rtalks

def rec_talks(uid, TK_info, TK_ratings):
  if uid.lower() =='n':
    rec_tids = rec_for_new_user(TK_info, TK_ratings)
  else:
    rec_tids = rec_for_existing_user(uid, TK_info)

  LINE_LENGTH = 80
  for rtid in rec_tids:
    tt = TK_info.ix[rtid]
    print '\n====={}: {} (tid={})=====\n{}\n[keywords]\n{}\n[themes]\n{}'.format(\
      tt.speaker, tt.title, rtid,
      textwrap.fill(tt.description, LINE_LENGTH), \
      textwrap.fill(tt.keywords.replace('[','').replace(']',''), LINE_LENGTH),
      re.sub('\[|\]|u\'|\'|\"|u\"', '', tt.related_themes))
      #re.sub('\[|\]|u\'|\'|\"|u\"', '', tt.related_themes).split(', '))

#if __name__ == '__main__':
if False:
  print 'Loading TED data'

  #TK_ratings, TK_info = load_talk_data()
  
  msg = '\nPlease enter your UserID, or "n" (for a new user), or "q" (for quit): '

  uid = raw_input(msg)
  while uid.lower() not in ['q', '']:
    rec_talks(uid, TK_info, TK_ratings)
    uid = raw_input(msg)



