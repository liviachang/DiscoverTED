from src.utils import *

def clean_idiap_talk_titles(titles_orig): # x=idf.title
  ## flatten the series of talk titles
  titles = []
  for t in titles_orig: # type(t)=list
    titles = titles + t

  ## encode and clean data so it maps better with user data
  for tidx, tdata in enumerate(titles):
    titles[tidx] = tdata.encode('utf-8', 'ignore').rstrip()
  return titles

def clean_idiap_user_fav_titles(titles_orig): # x = udf.favorites
  ## encode and clean data so it maps better with talk data
  ## it is time-consuming to apply the cleaning
  ## so cleaning is applied to unique titles first then all titles
  tunique = titles_orig.unique()
  tcleaned = []
  for t in tunique:
    tcleaned.append(t.encode('utf-8', 'ignore').rstrip())

  tmapper = pd.DataFrame({'title':tcleaned, 'title_orig':tunique})
  tdata = pd.DataFrame({'title_orig':titles_orig})
  titles = pd.merge(tdata, tmapper, how='left', on='title_orig')
  return titles.title

def merge_talk_data():
  print 'Merging talk data from idiap and my scraping results'

  ## load and clean talk data from idiap
  idf = pd.read_json(TALK_DATA_IDIAP_FN)
  idf['title_idiap'] = clean_idiap_talk_titles(idf.title)
  idf['url'] = [x.replace('.html', '') for x in idf['url']]

  ## load data created by src/scrape_ted_talks.py
  mdf = pd.read_csv(TALK_DATA_SCRAPED_FN)

  ## use url to map idiap and my scraped talk data
  idiap_cols = ['title_idiap', 'url', 'related_themes', 'related_videos']
  tdf = pd.merge(mdf, idf[idiap_cols], how='left', on=['url'])
  tdf.ix[tdf.title_idiap.isnull(), 'title_idiap'] = ''

  ## show basic stats to check the merge quality
  print 'idf.shape = {}, mdf.shape = {}, tdf.shape = {}'.format(\
    idf.shape, mdf.shape, tdf.shape)
  
  ## export the data for future use
  tdf.to_csv(TALK_DATA_FN, index=False)

  return tdf
  
def flatten_user_favorites(x): ## x = udf.favorites[0]
  ## from (user, [n titles]) to a list with n tuples
  ## each tuple is (user_id, favorite_title)
  favs = [f for f in x['favorites']]
  #print favs; print len(favs)
  favs = set(favs)
  uids = [x['user_id']] * len(favs)
  user_fav_pairs = zip(uids, favs)
  return user_fav_pairs

def clean_user_data():
  ## load idiap user data
  udf_orig = pd.read_json(USER_DATA_IDIAP_FN)

  ## flatten user data 
  ## each row for each "user/favorite title" combination
  udf = udf_orig.apply(flatten_user_favorites, axis=1)
  udf = list(chain(*udf))
  udf = pd.DataFrame(udf, columns=['uid_idiap', 'fav_title'])

  ## add fid as "favorite_id" 
  ## so it is easier to identify the record for debugging
  udf['fid'] = range(udf.shape[0])

  ## clean fav_title data
  ## so it is easier to map with titles in talk data
  udf['fav_title'] = clean_idiap_user_fav_titles(udf['fav_title'])

  return udf

def transform_user_data(is_split=True):
  print '\nTransforming user data from merged talk data and idiap user data'

  ## load talk data
  tdf = pd.read_csv(TALK_DATA_FN)
  ## load user data
  udf = clean_user_data()

  ## generate rating data from talk + user data
  ## each row for each "user/favorite title" combination
  ## some favorite talks in the user data is no longer available on ted.com
  ## remove them (~0.05% of the original user data)
  tcols = ['title_idiap', 'tid']
  rdf = pd.merge(udf, tdf[tcols], how='left', left_on='fav_title', right_on='title_idiap')
  rdf = rdf.drop_duplicates(rdf).dropna()

  if is_split:
    nftalks_per_user = rdf[['uid_idiap', 'tid']].groupby('uid_idiap').count()
    uids_test = nftalks_per_user.ix[nftalks_per_user['tid']>3]
    uids_test = np.random.choice(uids_test.index, size=N_TESTING_USERS, replace=False)
    rdf_test = rdf.ix[rdf['uid_idiap'].isin(uids_test),:]
    rdf_train = rdf.ix[-rdf['uid_idiap'].isin(uids_test),:]

    rdf_train.to_csv(USER_TALK_FN, index=False)
    rdf_test.to_csv(TEST_USER_TALK_FN, index=False)

    return rdf_train, rdf_test
  else:
    ## export the data for model building
    rdf.to_csv(USER_TALK_FN, index=False)
    return rdf,

def get_rating_matrix(user_talk_fn, rating_fn):
  print 'Save transformed user data into rating matrix'
  rdf = pd.read_csv(user_talk_fn)
  rdf['tid'] = rdf['tid'].astype(int)
  rdf['rating'] = 1

  rmat = rdf.pivot(index='uid_idiap', columns='tid', values='rating').fillna(0)
  rmat.to_csv(rating_fn)
  
if __name__ == '__main__':
  IS_SPLIT_USERS = True
  #tdf = merge_talk_data()
  rdf_train, rdf_test = transform_user_data()
  print 'rdf_train={}, rdf_test={}'.format(rdf_train.shape, rdf_test.shape)
  get_rating_matrix(USER_TALK_FN, RATING_MATRIX_FN)
  get_rating_matrix(TEST_USER_TALK_FN, TEST_RATING_MATRIX_FN)
  pass


