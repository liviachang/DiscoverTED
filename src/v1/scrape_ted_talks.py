from __future__ import division
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
from matplotlib import pyplot as plt
import random
#import requests

def get_talks_info():
    base_url = 'http://www.ted.com/talks/view/id/'
    ids = sorted(pd.read_csv('~/Galvanize/capstone/data/talks_list.csv').id)
    html_dir = '/Users/liviachang/TED/scraped_talks/'

    talks_info = []

    for vidx, video_id in enumerate(ids):
        print '#{}: for video_id = {}'.format(vidx, video_id)
        tfn = html_dir + str(video_id) + '.html'
        soup = BeautifulSoup(open(tfn), 'html.parser')
        
        ## copy from github: kiwix/TED
        json_data = soup.select('div.talks-main script')
        if len(json_data) > 0: 
            json_data = json_data[-1].text
            json_data = ' '.join(json_data.split(',', 1)[1].split(')')[:-1])
            json_data = json.loads(json_data)
        else:
            continue

        if 'talks' in json_data:
            web_type = 1
            jj = json_data['talks'][0]
            ratings = json_data['ratings']
        elif 'data' in json_data:
            web_type = 2
            jj = json_data['data']['media']
            ratings = json_data['data']['ratings']

        # Extract the speaker of the TED talk
        speaker = jj['speaker']

        # Extract the title of the TED talk
        title = jj['title']
        #title = speaker.lower() + ' ' + short_title.lower()
        #title = title.replace(':','')
        #title = title.replace("'s",'')
        #title = title.replace("'",'')

        # Extract the description of the TED talk
        description = soup.findAll(attrs={"name":"description"})[0]['content']

        # Extract the upload date of the TED talk
        filmed = datetime.fromtimestamp(jj['filmed']).strftime('%Y%m%d')

        # get URL
        url = jj['canonical']

        # Extract the length of the TED talk in minutes
        length = jj['duration']/60

        # Extract the thumbnail of the of the TED talk video
        thumbnail = jj['thumb']

        # number of languages
        if web_type == 1:
            nlang = len(jj['languages'])
        elif web_type == 2:
            nlang = np.nan

        # Extract the keywords for the TED talk
        keywords = jj['tags']

        # Extract the ratings list for the TED talk
        rdf = pd.DataFrame(ratings).set_index('name')['count']
        n_ratings = rdf.sum()
        rdf = rdf / n_ratings
        rdict = rdf.to_dict()
        
        event = jj['event']
        
        if web_type==1:
            n_views = soup.findAll(class_='v-a:m', id="sharing-count")[0].get_text()
            n_views = n_views.split()[0].replace(',', '')
            n_views = int(n_views)
        elif web_type==2:
            n_views = json_data['data']['viewed_count']

        # Append the meta-data to a list
        talk_info = {
            'tid': video_id,
            'title': title.encode('utf-8', 'ignore'),
            #'short_title': short_title.encode('utf-8', 'ignore'),
            'description': description.encode('utf-8', 'ignore'),
            'speaker': speaker.encode('utf-8', 'ignore'),
            'filmed': filmed,
            'url' : url,
            'ted_event': event,
            'thumbnail': thumbnail.encode('utf-8', 'ignore'),
            'length': length,
            'keywords': keywords,
            'nratings': n_ratings,
            'nviews': n_views}

        talk_info.update(rdict)

        talks_info.append(talk_info)

    return talks_info


def get_talk_htmls():
    base_url = 'http://www.ted.com/talks/view/id/'
    ids = pd.read_csv('../data/ted.talks.list.csv').id
    ids = sorted(ids)
    for tid in ids:
        target_url = base_url + str(tid)
        os.system('wget {} -O ../data/talks/{}.html'.format(target_url, tid))
        time.sleep(2)

## =========== code under developement ============
def get_user_info():
    html_dir = 'data/profiles/'

    users_info = []

    for user_id in xrange(1,140):
        tfn = html_dir + str(user_id) + '.html'
        soup = BeautifulSoup(open(tfn), 'html.parser')

        fav_speakers = soup.findAll(class_='h12 talk-link__speaker')
        fav_speakers = [fs.text for fs in fav_speakers]

        if len(fav_speakers)>0:
            print 'user_id = {}, #fav = {}'.format(user_id, len(fav_speakers))
            fav_talks = soup.findAll(class_='h9 m5')
            fav_talks = [ft.text for ft in fav_talks]

            for (fs, ft) in zip(fav_speakers, fav_talks):
                user_info = {
                    'user_id': user_id,
                    'fav_speaker': fs,
                    'fav_talk': ft
                }
                users_info.append(user_info)

    return users_info

def get_user_htmls():
    # https://www.ted.com/profiles/1227693
    # ../data/profiles/1227693
    # ../data/profiles/1621901 ## 3 favorite talks
    # ../data/profiles/1945339 ## 1 favorite talk
    base_url = 'http://www.ted.com/profiles/'
    for tid in xrange(20,1000): ## till 532
        print tid
        target_url = base_url + str(tid)
        os.system('wget {} -O data/profiles/{}.html'.format(target_url, tid))
        x = random.random()*3
        time.sleep( x )
## =========== code under developement ============

if __name__ == '__main__':
    data = get_talks_info()
    df = pd.DataFrame(data)
    df.to_csv('data/talks_info_scraped_tmp.csv', encoding='utf-8', index=False)
    os.system('mv data/talks_info_scraped_tmp.csv data/talks_info_scraped.csv')
