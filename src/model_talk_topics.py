from src.utils import *

def get_talk_doc(x):
  ''' 
  Input:
    - x (Series): data for one talk
  Output:
    - doc (string): cleaned string with both keywords and description
  '''
  doc = x[['keywords', 'description']].tolist()
  doc = ' '.join(doc).lower()
  doc = doc.decode('utf-8', 'ignore')
  return doc

def tokenize_talk_doc(doc, tokenizer, stop_wds, stemmer):
  ''' 
  Input:
    - doc (string): cleaned string with both keywords and description
    - tokenizer (RegexpTokenizer): used to keep word characters only
    - stop_wds (list): a list of stop words
    - stemmer (PorterStemmer): used to stem tokens
  Output:
    - tokens_stemmed: a list of stemmed tokens. stop words are excluded
  '''
  tokens = tokenizer.tokenize(doc)
  tokens_no_stop_wds = [wd for wd in tokens if not wd in stop_wds]
  tokens_stemmed = [stemmer.stem(wd) for wd in tokens_no_stop_wds]

  return tokens_stemmed

def get_talk_tokens(docs):
  '''
  Input: 
    - docs (Series): 
      one row for one talk. index is talk id. content is a list of tokenized words
  Output: 
  '''
  tknizer = RegexpTokenizer(r'\w+')
  stop_wds = get_stop_words('en')
  pstemmer = PorterStemmer()
  tmpf = ftPartial(tokenize_talk_doc, tokenizer=tknizer, stop_wds=stop_wds, stemmer=pstemmer)
  tokens = docs.apply(tmpf)
  return tokens

def get_topics_from_tf(x, mdl):
  score_tuple = mdl[x]
  result = np.zeros(N_TOTAL_TOPICS+N_GROUP_TOPICS)
  for (idx, score) in score_tuple:
    result[idx] = score

  top_topics = result.argsort()[::-1][:N_GROUP_TOPICS]
  for idx in xrange(N_GROUP_TOPICS):
    result[N_TOTAL_TOPICS+idx] = top_topics[idx]
  return result

def model_talk_topics_NMF(R):
  t1 = print_time('Topic modeling via NMF')
  nmf = NMF(n_components=N_TOTAL_TOPICS, random_state=319).fit(R.values)
  U = pd.DataFrame(nmf.transform(R.values), index=R.index)
  V = pd.DataFrame(nmf.components_, columns=R.columns)

  V_normed = V.apply(lambda x: x/sum(x), axis=0)
  V_normed.index = ['topic{:02d}'.format(x) for x in xrange(V_normed.shape[0])]
  V_top_topics = V_normed.apply(lambda x: x.argsort()[::-1][:N_GROUP_TOPICS])
  V_top_topics.index = ['top_topic{}'.format(x+1) for x in xrange(V_top_topics.shape[0])]

  TK_topics = V_normed.append(V_top_topics).transpose()
  TK_topics.index.name = 'tid'

  return TK_topics, U, V

def model_talk_topics_GMF(R):
  data = R.reset_index()
  df = pd.melt( data, id_vars='uid_idiap', value_vars=list(data.columns[1:]),
    var_name='tid', value_name='rating')
  sdf = gl.SFrame(df)
  gmf = gl.factorization_recommender.create(sdf, 'uid_idiap', 'tid', 'rating', 
    num_factors=N_TOTAL_TOPICS)
  M = gmf.get('coefficients')

  U = M['uid_idiap']
  U = pd.DataFrame( U['factors'].to_numpy(), index=U['uid_idiap'], 
    columns=range(N_TOTAL_TOPICS))
  
  V = M['tid']
  V = pd.DataFrame( V['factors'].to_numpy().transpose(), 
    index=range(N_TOTAL_TOPICS), columns=V['tid'])
  V.index = ['topic{:02d}'.format(x) for x in xrange(V.shape[0])]

  V_top_topics = V.apply(lambda x: x.argsort()[::-1][:N_GROUP_TOPICS])
  V_top_topics.index = ['top_topic{}'.format(x+1) for x in xrange(V_top_topics.shape[0])]
  TK_topics = V.append(V_top_topics).transpose()
  TK_topics.index.name = 'tid'
  
  return TK_topics, U, V

def model_talk_topics_LDA(TK_info):
  TK_docs = TK_info.apply(get_talk_doc, axis=1)
  TK_tokens = get_talk_tokens(TK_docs)

  ## create 1-to-1 mapping from id to word
  id2word = corpora.Dictionary(TK_tokens)
  ## get term frequency
  TK_tf = [id2word.doc2bow(t) for t in TK_tokens]

  mdl = LdaModel(TK_tf, num_topics=N_TOTAL_TOPICS, id2word=id2word, passes=20)

  talk_topics = [get_topics_from_tf(cp, mdl) for cp in TK_tf]

  df = pd.DataFrame(talk_topics, columns=get_topic_score_names())
  df.index = TK_info.index
  df['tokens'] = TK_tokens

  return df, id2word, mdl


def get_topic_score_names():
  df_cols = ['topic{:02d}'.format(x) for x in range(N_TOTAL_TOPICS)]
  df_cols = df_cols + ['top_topic1', 'top_topic2']
  return df_cols

def get_topic_all_docs(tids, TK_info):
  talks = TK_info.loc[tids]
  docs = talks.apply(get_talk_doc, axis=1)
  desc = ' '.join( docs.tolist() )
  return desc

def save_LDA_topics_data():
  with open(LDA_TOPICS_FN, 'wb') as f:
    pickle.dump( (TK_topics_LDA, TP_info_LDA), f)

def save_LDA_model_data():
  with open(LDA_MODEL_FN, 'wb') as f:
    pickle.dump( (token_mapper, mdl_LDA), f)

def save_NMF_topics_data():
  with open(NMF_TOPICS_FN, 'wb') as f:
    pickle.dump( (TK_topics_NMF, TP_info_NMF), f)

def save_NMF_model_data():
  with open(NMF_MODEL_FN, 'wb') as f:
    pickle.dump( (U_NMF, V_NMF, tfidf_vec_NMF, TP_tfidf_NMF), f)

def save_GMF_topics_data():
  with open(GMF_TOPICS_FN, 'wb') as f:
    pickle.dump( (TK_topics_GMF, TP_info_GMF), f)

def save_GMF_model_data():
  with open(GMF_MODEL_FN, 'wb') as f:
    pickle.dump( (U_GMF, V_GMF, tfidf_vec_GMF, TP_tfidf_GMF), f)
  
def get_topic_talks(TK_topics, TK_info):
  talk_df = TK_topics.reset_index()[['tid', 'top_topic1']]
  topic_tids = talk_df.groupby('top_topic1').apply(lambda x: x.tid.tolist())

  tmpf = ftPartial(get_topic_all_docs, TK_info = TK_info)
  topic_desc = topic_tids.apply(tmpf)

  tmpf = ftPartial(tokenize_talk_doc, tokenizer=RegexpTokenizer(r'\w+'), \
    stop_wds=get_stop_words('en'), stemmer=PorterStemmer())
  topic_desc = topic_desc.apply(tmpf)
  topic_desc = topic_desc.apply(lambda x: ' '.join(x))
  
  topic_df = pd.DataFrame({'tids':topic_tids, 'desc':topic_desc})
  topic_df.index = ['topic{:02d}'.format(x) for x in range(N_TOTAL_TOPICS)]
  return topic_df

if __name__ == '__main__':
  TK_ratings, TK_info, U_ftalks, R_mat = load_ted_data()

  TK_topics_LDA, token_mapper, mdl_LDA = model_talk_topics_LDA(TK_info)
  TP_info_LDA = get_topic_talks(TK_topics_LDA, TK_info)
  save_LDA_topics_data()
  save_LDA_model_data()

  TK_topics_NMF, U_NMF, V_NMF = model_talk_topics_NMF(R_mat)
  TP_info_NMF = get_topic_talks(TK_topics_NMF, TK_info)
  tfidf_vec_NMF = TfidfVectorizer(stop_words='english')
  TP_tfidf_NMF = tfidf_vec_NMF.fit_transform(TP_info_NMF['desc'].values)
  save_NMF_topics_data()
  save_NMF_model_data()
  
  TK_topics_GMF, U_GMF, V_GMF = model_talk_topics_GMF(R_mat)
  TP_info_GMF = get_topic_talks(TK_topics_GMF, TK_info)
  tfidf_vec_GMF = TfidfVectorizer(stop_words='english')
  TP_tfidf_GMF = tfidf_vec_GMF.fit_transform(TP_info_GMF['desc'].values)
  save_GMF_topics_data()
  save_GMF_model_data()

  
