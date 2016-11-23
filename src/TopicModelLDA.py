from src.utils import *

class TopicModelLDA(object):
  def __init__(self, n_topics):
    self.tknizer = RegexpTokenizer(r'[a-zA-Z]+')
    self.stop_wds = get_stop_words('en')
    self.stemmer = SnowballStemmer('english') #PorterStemmer()
    self.n_total_topics = n_topics

    ## model information
    self.id2word = None ## model to calculate tfidf
    self.mdl = None
    self.talk_tscores = None
    self.topic_reps = None
  
  def fit(self, TK_info):
    ''' for each talk in TK_info (DataFrame), run LDA and get topic scores '''
    TK_tokens = TK_info.apply(self._tokenize_talk, axis=1)

    ## create 1-to-1 mapping from id to word
    self.id2word = corpora.Dictionary(TK_tokens)

    ## get term frequency
    TK_tf = [self.id2word.doc2bow(t) for t in TK_tokens]

    print_time('Fitting LDA Model')
    self.mdl = LdaModel(TK_tf, num_topics=self.n_total_topics, id2word=self.id2word, \
      random_state=0, passes=20)
    print_time('Fitting LDA Model -- Done')

    talk_tscores = [self._get_tscores_from_tf(tf) for tf in TK_tf]
    talk_tscores = pd.DataFrame(talk_tscores, columns=self._get_topic_score_names())
    talk_tscores.index = map(str, TK_info.index)
    talk_tscores['tokens'] = TK_tokens
    self.talk_tscores = talk_tscores
    self.topic_reps = self._get_rep_talks()

  def transform(self, user_text):
    ''' Transform user_text (string) to topic scores (Series) '''
    user_tokens = self._tokenize_text(user_text)
    user_tf = self.id2word.doc2bow(user_tokens)
    tscores = self._get_tscores_from_tf(user_tf)
    tscores = pd.Series(tscores, index=self._get_topic_score_names())
    return tscores

  def _get_rep_talks(self):
    tscores = self.talk_tscores.ix[:, :self.n_total_topics]
    rep_tids = tscores.apply(self._get_rep_talks_per_topic, axis=0)
    return rep_tids

  def _get_rep_talks_per_topic(self, x, n_rep_talks=100):
    talk_idx = x.values.argsort()[::-1][:n_rep_talks]
    talk_ids = x.index[talk_idx].tolist()
    return talk_ids

  def _get_topic_score_names(self):
    ''' return column names for self.talk_tscores '''
    df_cols_basic = ['topic{:02d}'.format(x) for x in range(self.n_total_topics)]
    df_cols_top = ['top_topic{}'.format(x) for x in xrange(1,N_GROUP_TOPICS+1)]
    df_cols = df_cols_basic + df_cols_top
    return df_cols

  def _get_tscores_from_tf(self, x):
    ''' convert x (a list of term frequency) to tscores (a list of topic scores) '''
    scores_tuple = self.mdl[x]
    tscores = np.zeros(self.n_total_topics + N_GROUP_TOPICS) 
    tscores[self.n_total_topics:] = np.nan
    for (idx, score) in scores_tuple:
      tscores[idx] = score

    top_topics = tscores[:self.n_total_topics].argsort()[::-1]
    top_topics = map(int, top_topics[:min(N_GROUP_TOPICS, sum(tscores>0.))])
    for idx in xrange(len(top_topics)):
      tscores[self.n_total_topics+idx] = top_topics[idx]
    return tscores
  
  def _tokenize_talk(self, x):
    '''  for each talk in TK_info (Series), get tokens for its document (list) '''
    text = x[['keywords', 'description']].tolist()
    text = ' '.join(text).lower().decode('utf-8', 'ignore')
    tokens = self._tokenize_text(text)
    return tokens

  def _tokenize_text(self, text):
    '''  for the input text (string), get tokens for its document (list) '''
    tokens = self.tknizer.tokenize(text)
    tokens_no_stop_wds = [wd for wd in tokens if not wd in self.stop_wds]
    tokens_stemmed = [self.stemmer.stem(wd) for wd in tokens_no_stop_wds]
    tokens_xshort = [x for x in tokens_stemmed if len(x)>=3]
    return tokens_xshort
  
if __name__ == '__main__':
  TK_ratings, TK_info, U_ftalks, R_mat = load_ted_data()

  mdlLDA = TopicModelLDA(N_TOTAL_TOPICS)
  mdlLDA.fit(TK_info)
  
  with open(TOPIC_MODEL_LDA_FN, 'wb') as f:
    print_time('Saving TopicModelLDA')
    dill.dump(mdlLDA, f)


