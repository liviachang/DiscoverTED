import pandas as pd
import numpy as np
import itertools
import unicodedata
import sys


def merge_talk_data():
  print 'Merging talk data from idiap and my scraping results'
  idiap_tfn = '~/TED/TED_dataset/ted_talks-10-Sep-2012.json'
  my_tfn = '~/Galvanize/capstone/data/talks_info_scraped.csv'

  idf = pd.read_json(idiap_tfn)
  idf['title_idiap'] = [xx[0].encode('utf8') for xx in idf.title.values]
  idf['url'] = [x.replace('.html', '') for x in idf['url']]
  mdf = pd.read_csv(my_tfn)

  idiap_cols = ['title_idiap', 'url', 'related_themes', 'related_videos']
  tdf = pd.merge(mdf, idf[idiap_cols], how='left', on=['url'])

  print 'idf.shape = {}, mdf.shape = {}, tdf.shape = {}'.format(\
    idf.shape, mdf.shape, tdf.shape)
  
  tdf.to_csv('~/Galvanize/capstone/data/talks_info_merged.csv', index=False)
  
  #for testing purpose to see which talks in idf is not matched with mdf
  #x = tdf.title_idiap
  #xx = []
  #for tmp in x:
  #  if type(tmp) is list:
  #    xx = xx + tmp
  #
  #y = idf.title_idiap
  #yy = []
  #for tmp in y:
  #  if type(tmp) is list:
  #    yy = yy + tmp


def flatten_user_favorites(x):
  favs = [f for f in x['favorites']]
  uids = [x['user_id']] * len(favs)
  user_fav_pairs = zip(uids, favs)
  return user_fav_pairs

def clean_title(x):
  return x.encode('ascii', 'ignore').rstrip()

def transform_user_data():
  print 'Transforming user data from merged talk data and idiap user data'
  tfn = '~/Galvanize/capstone/data/talks_info_merged.csv'
  ufn = '~/TED/TED_dataset/ted_users-10-Sep-2012.json'

  tdf = pd.read_csv(tfn)
  tdf['title_idiap'] = tdf['title_idiap'].apply(clean_title)

  udf_orig = pd.read_json(ufn)
  udf = udf_orig.apply(flatten_user_favorites, axis=1)
  udf = list(itertools.chain(*udf))
  udf = pd.DataFrame(udf, columns=['uid_idiap', 'fav_title'])
  udf['fid'] = xrange(udf.shape[0])
  udf['fav_title'] = udf['fav_title'].apply(clean_title)

  ## generate rating data
  tcols = ['title_idiap', 'tid']
  rdf = pd.merge(udf, tdf[tcols], how='left', left_on='fav_title', right_on='title_idiap')
  print rdf.info()
  print rdf[rdf.isnull().any(axis=1)].head(2)

  rdf.to_csv('~/Galvanize/capstone/data/users_info_transformed.csv', index=False)

if __name__ == '__main__':
  #merge_talk_data()
  udf = transform_user_data()
  pass






if False:
  def f(x):
    return ''.join(x['title_idiap'].encode('utf-8'))

  def transform_series_of_unicode_list(x):
    output = []
    for tmp in x:
      if type(tmp) is str: 
        tmp = tmp.decode('utf-8', 'ignore')
        output = output + tmp
        print tmp
        print output
    return output

  #tt = pd.read_json(data_dir+tfn); tt['title'] = tt.apply(lambda x: x.title[0], axis=1)
  tdf = pd.read_csv(rfn)
  udf_orig = pd.read_json(data_dir+ufn)
  udf_orig['uid'] = range(udf_orig.shape[0])


  udf = udf_orig.apply(flatten_user_favorites, axis=1)
  udf = list(itertools.chain(*udf))
  udf = pd.DataFrame(udf, columns=['user_id', 'fav_title'])

  fdf = pd.merge(udf, tdf[['title', 'ttid']], how='left',
    left_on = 'fav_title', right_on = 'title')

  missing = fdf[fdf.isnull().any(axis=1)]
  print missing.head(2)
  print missing.shape

  fdf[fdf.isnull().any(axis=1)].head(2)
  "Richard St. John's 8 secrets of succes"
  "richard St. John 8 secrets of succes"
