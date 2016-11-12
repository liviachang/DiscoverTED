import pandas as pd
import numpy as np
import itertools
import unicodedata
import sys

def clean_idiap_talk_titles(titles_orig): # x=idf.title
  titles = []
  for t in titles_orig: # type(t)=list
    titles = titles + t

  for tidx, tdata in enumerate(titles):
    titles[tidx] = tdata.encode('utf-8', 'ignore').rstrip()
  return titles

def clean_idiap_user_fav_titles(titles_orig): # x = udf.favorites
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
  idiap_tfn = '/Users/liviachang/TED/idiap/ted_talks-10-Sep-2012.json'
  my_tfn = '/Users/liviachang/Galvanize/capstone/data/talks_info_scraped.csv'

  idf = pd.read_json(idiap_tfn)
  idf['title_idiap'] = clean_idiap_talk_titles(idf.title)
  idf['url'] = [x.replace('.html', '') for x in idf['url']]

  mdf = pd.read_csv(my_tfn)

  idiap_cols = ['title_idiap', 'url', 'related_themes', 'related_videos']
  tdf = pd.merge(mdf, idf[idiap_cols], how='left', on=['url'])
  tdf.ix[tdf.title_idiap.isnull(), 'title_idiap'] = ''

  print 'idf.shape = {}, mdf.shape = {}, tdf.shape = {}'.format(\
    idf.shape, mdf.shape, tdf.shape)
  
  tdf.to_csv('/Users/liviachang/Galvanize/capstone/data/talks_info_merged.csv', index=False)
  return tdf
  
def flatten_user_favorites(x):
  favs = [f for f in x['favorites']]
  uids = [x['user_id']] * len(favs)
  user_fav_pairs = zip(uids, favs)
  return user_fav_pairs

def clean_user_data():
  ufn = '/Users/liviachang/TED/idiap/ted_users-10-Sep-2012.json'
  udf_orig = pd.read_json(ufn)
  udf = udf_orig.apply(flatten_user_favorites, axis=1)
  udf = list(itertools.chain(*udf))
  udf = pd.DataFrame(udf, columns=['uid_idiap', 'fav_title'])
  udf['fid'] = range(udf.shape[0])
  udf['fav_title'] = clean_idiap_user_fav_titles(udf['fav_title'])
  return udf

def transform_user_data():
  print '\nTransforming user data from merged talk data and idiap user data'

  tfn = '/Users/liviachang/Galvanize/capstone/data/talks_info_merged.csv'
  tdf = pd.read_csv(tfn)

  udf = clean_user_data()

  ## generate rating data
  tcols = ['title_idiap', 'tid']
  rdf = pd.merge(udf, tdf[tcols], how='left', left_on='fav_title', right_on='title_idiap')

  print rdf.info()

  rdf = rdf.dropna()
  print 'rdf.shape = {}'.format(rdf.shape)

  rdf.to_csv('/Users/liviachang/Galvanize/capstone/data/users_info_transformed.csv', index=False)
  return rdf

if __name__ == '__main__':
  tdf = merge_talk_data()
  rdf = transform_user_data()
