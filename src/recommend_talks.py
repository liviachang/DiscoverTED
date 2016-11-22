from src.utils import *
from src.model_talk_topics import get_talk_doc, tokenize_talk_doc
from src.model_talk_topics import get_topics_from_tf, get_topic_score_names

TK_ratings, TK_info, U_ftalks, R_mat = load_ted_data()

TK_topics_LDA, TP_info_LDA = load_topics_data(mdl_name='LDA')
G_rtopics_LDA, U_tscores_LDA = load_group_data(mdl_name='LDA')
token_mapper, mdl_LDA = load_LDA_model_data()

TK_topics_NMF, TP_info_NMF = load_topics_data(mdl_name='NMF')
G_rtopics_NMF, U_tscores_NMF = load_group_data(mdl_name='NMF')
U_NMF, V_NMF, tfidf_vec_NMF, TP_tfidf_NMF = load_MF_model_data(mdl_name='NMF')

TK_topics_GMF, TP_info_GMF = load_topics_data(mdl_name='GMF')
G_rtopics_GMF, U_tscores_GMF = load_group_data(mdl_name='GMF')
U_GMF, V_GMF, tfidf_vec_GMF, TP_tfidf_GMF = load_MF_model_data(mdl_name='GMF')


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

  output = ''
  for (idx, rtyp) in rtyp_dict.iteritems():
    output += '{0: >2d}: '.format(idx)
    output += '{0: <12} | '.format(rtyp)
    if idx % 4 == 3:
      output = output + '\n'
  print output

  input_rtyp_idx = raw_input('\nTypes you are interested in: ' + \
    '(say, \'5,7\' for \'Informative+Inspiring\'):  ')
  if input_rtyp_idx == '':
    input_rtyp_idx = '5,7'
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
    user_text = 'computer internet technology data stock market finance economics'
  print user_text

  return user_text

def get_topics_from_text(text, mdl_name=MODEL_NAMES[0]):
  tknizer = RegexpTokenizer(r'\w+')
  stop_wds = get_stop_words('en')
  #pstemmer = PorterStemmer()
  #new_tokens = tokenize_talk_doc(text, tknizer, stop_wds, pstemmer)
  
  sstemmer = SnowballStemmer('english')
  new_tokens = tokenize_talk_doc(text, tknizer, stop_wds, sstemmer)

  if mdl_name == 'LDA':
    new_tf = token_mapper.doc2bow(new_tokens)
    topics = get_topics_from_tf(new_tf, mdl_LDA)
    result = pd.Series(topics, index=get_topic_score_names())

  elif mdl_name == 'NMF':
    #tfidf_vec_NMF = TfidfVectorizer(stop_words='english')
    #TP_tfidf_NMF = tfidf_vec_NMF.fit_transform(TP_info['desc'].values)
    new_tfidf = tfidf_vec_NMF.transform([' '.join(new_tokens)])
    cos_sim = linear_kernel(new_tfidf, TP_tfidf_NMF)[0]
    gtopics = cos_sim.argsort()[::-1][:N_GROUP_TOPICS]
    result = pd.Series(np.concatenate([cos_sim, gtopics]), index=get_topic_score_names())
  
  elif mdl_name == 'GMF':
    #tfidf_vec_GMF = TfidfVectorizer(stop_words='english')
    #TP_tfidf_GMF = tfidf_vec_GMF.fit_transform(TP_info['desc'].values)
    new_tfidf = tfidf_vec_GMF.transform([' '.join(new_tokens)])
    cos_sim = linear_kernel(new_tfidf, TP_tfidf_GMF)[0]
    gtopics = cos_sim.argsort()[::-1][:N_GROUP_TOPICS]
    result = pd.Series(np.concatenate([cos_sim, gtopics]), index=get_topic_score_names())

  return result

def get_new_user_tscores_fratings(mdl_name):

  ## get user input text and rating preference
  user_text = get_user_topic_keywords()
  user_fratings = get_new_user_fav_ratings()
  
  ## convert the user input text to topic scores
  user_tscores = get_topics_from_text(user_text, mdl_name=mdl_name)

  return user_tscores, user_fratings

def get_user_rec_talks(tscores, user_fratings, input_tids=None):

  #gtopics_list = map(int, tscores[N_TOTAL_TOPICS:])
  #gtopics_key = str(sorted(map(int, tscores[N_TOTAL_TOPICS:])))
  
  gtopics_list = [x for x in tscores[N_TOTAL_TOPICS:] if ~np.isnan(x)]
  gtopics_list = map(int, gtopics_list)
  gtopics_key = str(sorted(gtopics_list))

  ## for topics to go deeper (topics already liked)
  deeper_rtopics = ['topic{:02d}'.format(x) for x in gtopics_list]
  deeper_candidates = np.array(TP_info.ix[deeper_rtopics, 'tids'].tolist())
  if input_tids is not None:
    deeper_candidates = [ [x for x in tmp_tids if x not in input_tids] \
      for tmp_tids in deeper_candidates]
  deeper_rtalks = get_rtalks_from_ratings(user_fratings, deeper_candidates)

  ## for topics to go wider (topics new to the user)
  wider_rtopics = G_rtopics[gtopics_key]
  wider_candidates = TP_info.ix[wider_rtopics, 'tids'].tolist()
  if input_tids is not None:
    wider_candidates = [ [x for x in tmp_tids if x not in input_tids] \
      for tmp_tids in wider_candidates]
  wider_rtalks = get_rtalks_from_ratings(user_fratings, wider_candidates)

  return deeper_rtalks + wider_rtalks, deeper_rtopics + wider_rtopics
 
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

  top_topics = user_tscores.argsort()[::-1][:N_GROUP_TOPICS] 
  for idx in xrange(N_GROUP_TOPICS):
    user_tscores['top_topic{}'.format(idx+1)] = top_topics[idx]
  
  tids = U_ftalks.ix[U_ftalks.uid_idiap==uid, 'tid']
  user_fratings = TK_ratings.ix[tids]

  return user_tscores, user_fratings

def print_topic_keywords(mdl_name, n_kws=10):
  if mdl_name == 'LDA':
    mdl = mdl_LDA
    topic_kws = mdl.print_topics(num_words=n_kws)
    for (idx, kws) in topic_kws:
      kws = kws.split(' + ')
      kws = [re.findall(r'\w+', kw)[2] for kw in kws]
      kws = ', '.join(kws)
      print 'topic{:02d}: {}'.format(idx, kws)
  elif mdl_name[1:] == 'MF':
    if mdl_name == 'NMF':
      vocabs = tfidf_vec_NMF.get_feature_names()
      tfidf_mat = TP_tfidf_NMF
    elif mdl_name == 'GMF':
      vocabs = tfidf_vec_GMF.get_feature_names()
      tfidf_mat = TP_tfidf_GMF

    for (idx, tfidf_list) in enumerate(tfidf_mat):
      top_idx = sorted( np.argsort( tfidf_list.toarray() )[0][::-1][:n_kws] )
      top_vocabs = np.array(vocabs)[top_idx]
      print 'topic{:02d}: {}'.format(idx, ', '.join(top_vocabs))

def rec_talks(uid, mdl_name):
  if uid.lower() =='n':
    tscores, fratings = get_new_user_tscores_fratings(mdl_name)
  else:
    if uid.lower() == 'e':
      uid = random.choice( U_tscores.index )
    tscores, fratings = get_existing_user_tscores_fratings(uid)

  rec_tids, rec_topics = get_user_rec_talks(tscores, fratings)
  print_rtalks(rec_tids)
  print 'rec_topics = {}'.format(rec_topics)
  print_topic_keywords(mdl_name=mdl_name)


def print_rtalks(rec_tids):
  LINE_LENGTH = 100

  for rtid in rec_tids:
    tt = TK_info.ix[rtid]

    tthemes = tt.related_themes

    t_rating = TK_ratings.ix[rtid]
    t_rating = t_rating[np.argsort(t_rating)[::-1][:3]]
    t_rating = ', '.join( t_rating.reset_index().apply(
      lambda x: '{}:{:.0f}%'.format(x[0], np.round(x[1]*1e2,0)), axis=1) )

    msg = '\n====={}: {} ({})=====\n{}\n[keywords] {}\n[ratings] {}'.format(\
        tt.speaker, tt.title, tt.ted_event, #rtid,
        textwrap.fill(tt.description, LINE_LENGTH), \
        textwrap.fill(tt.keywords.replace('[','').replace(']',''), LINE_LENGTH),
        t_rating)
    if not isinstance(tthemes, float):
      msg = '{}\n[themes] {}'.format(msg, 
        re.sub('\[|\]|u\'|\'|\"|u\"', '', tthemes))

    print msg

def get_success_metrics(test_udf, mdl_name):
  print mdl_name
  global G_rtopics, U_tscores, TK_topics, TP_info

  if mdl_name == 'LDA':
    G_rtopics, U_tscores, TK_topics, TP_info = \
      G_rtopics_LDA, U_tscores_LDA, TK_topics_LDA, TP_info_LDA
  elif mdl_name == 'NMF':
    G_rtopics, U_tscores, TK_topics, TP_info = \
      G_rtopics_NMF, U_tscores_NMF, TK_topics_NMF, TP_info_NMF
  elif mdl_name == 'GMF':
    G_rtopics, U_tscores, TK_topics, TP_info = \
      G_rtopics_GMF, U_tscores_GMF, TK_topics_GMF, TP_info_GMF

  talks_per_topic = TK_topics['top_topic1'].value_counts()
  talks_per_topic = talks_per_topic.sort_index() / sum(talks_per_topic)

  deeper_scores, wider_scores, deeper_bmk, wider_bmk = [], [], [], []

  for uid in test_udf['uid_idiap'].unique().tolist():
    tids = map(str, test_udf.ix[test_udf['uid_idiap']==uid, 'tid'] )
    tids_input = np.random.choice(tids, 2, replace=False)
    tids_truth = [x for x in tids if x not in tids_input]
    #print 'uid={}, tids_input={}'.format(uid, tids_input)

    user_text = TK_info.ix[tids_input].apply(get_talk_doc, axis=1).tolist()
    user_text = reduce(lambda x, y: x+y, user_text)
    user_text = user_text.replace('[', '').replace(']', '')
    user_tscores = get_topics_from_text(user_text, mdl_name=mdl_name)

    user_fratings = TK_ratings.ix[tids_input]

    rec_tids, topics_rec = get_user_rec_talks(user_tscores, user_fratings,
      input_tids = tids_input)
    topics_input = TK_topics.ix[map(str, tids_input), 'top_topic1']
    topics_truth = TK_topics.ix[map(str, tids_truth), 'top_topic1']

    topics_rec_num = [float(x.replace('topic0', '')) for x in topics_rec]
    deeper_scores.append(np.mean(topics_truth.isin(topics_rec_num[:N_GROUP_TOPICS]) ))
    wider_scores.append(np.mean(topics_truth.isin(topics_rec_num[N_GROUP_TOPICS:]) ))
    deeper_bmk.append(sum( talks_per_topic[topics_rec_num[:N_GROUP_TOPICS]] ))
    wider_bmk.append(sum( talks_per_topic[topics_rec_num[N_GROUP_TOPICS:]] ))
  
  return np.array(deeper_scores), np.array(wider_scores), \
    np.array(deeper_bmk), np.array(wider_bmk)

def evaluate_recommender():
  ## get testing uids
  test_udf = pd.read_csv(TEST_USER_TALK_FN)
  test_udf['tid'] = test_udf['tid'].astype(int)
  
  #for mdl in [MODEL_NAMES[0]]:
  for mdl in MODEL_NAMES:
    deeper_scores, wider_scores, deeper_bmk, wider_bmk = get_success_metrics(test_udf, mdl_name=mdl)
    rec_scores = deeper_scores + wider_scores
    bmk_scores = deeper_bmk + wider_bmk
    outperform_scores = (rec_scores - bmk_scores)

    t1 = print_time('\nEvaluation Results for model {}'.format(mdl))
    print 'My recommender: deeper {:4f}, wider {:4f}, total {:4f}'.format(\
      np.mean(deeper_scores), np.mean(wider_scores), np.mean(rec_scores))
    print 'Benchmark: deeper {:4f}, wider {:4f}, total {:4f}'.format(\
      np.mean(deeper_bmk), np.mean(wider_bmk), np.mean(bmk_scores))
    print 'outputform: score {:.4f}, freq {:4f}, pvalue {:4f}'.format(\
      np.mean(outperform_scores), np.mean(outperform_scores>0),
      ttest_1samp(outperform_scores, 0).pvalue)

    print_topic_keywords(mdl_name=mdl)

  
if __name__ == '__main__':
  if len(sys.argv) > 1:
    MODE = sys.argv[1]
  else:
    MODE = ['RECOMMEND', 'EVALUATE'][0]

  if MODE == 'RECOMMEND':
    msg = '\nPlease enter your UserID, or "n" (for a new user), or "q" (for quit): '

    MODEL = BEST_MODEL
    print 'Evaluation is based on model {}'.format(MODEL)

    if MODEL == 'LDA':
      G_rtopics, U_tscores, TK_topics, TP_info = \
        G_rtopics_LDA, U_tscores_LDA, TK_topics_LDA, TP_info_LDA
    elif MODEL == 'NMF':
      G_rtopics, U_tscores, TK_topics, TP_info = \
        G_rtopics_NMF, U_tscores_NMF, TK_topics_NMF, TP_info_NMF
    elif MODEL == 'GMF':
      G_rtopics, U_tscores, TK_topics, TP_info = \
        G_rtopics_GMF, U_tscores_GMF, TK_topics_GMF, TP_info_GMF

    uid = raw_input(msg)
    while uid.lower() not in ['q', '']:
      rec_talks(uid, MODEL)
      uid = raw_input(msg)
  else:
    evaluate_recommender()


if False:
  U_tscores = U_tscores_LDA
  kk = KMeans(n_clusters=55, random_state=0).fit(U_tscores_LDA.values)
  user_grp = kk.predict(user_tscores[:N_TOTAL_TOPICS])
  grps = U_tscores.ix[kk.labels_==user_grp,:].index
  others = U_tscores.ix[kk.labels_!=user_grp,:].index

  np.sum((U_tscores.ix[grps,:]-user_tscores[:N_TOTAL_TOPICS])**2, axis=1).describe()
  np.sum((U_tscores.ix[others,:]-user_tscores[:N_TOTAL_TOPICS])**2, axis=1).describe()



