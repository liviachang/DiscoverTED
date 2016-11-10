from __future__ import division
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

data_dir = '~/TED/TED_dataset/'

tfn = ['ted_talks-10-Sep-2012.json', 'ted_talks-25-Apr-2012.json'][0]
ufn = ['ted_users-10-Sep-2012.json', 'ted_users-25-Apr-2012.json'][0]

tdf = pd.read_json(data_dir+tfn)
udf = pd.read_json(data_dir+ufn)

''' python stats.py --sep
--
Talks: 1203
Speakers: 1006
Users: 74760
Active Users: 12605
Tags: 298
Themes: 46
Transcripts: 1203
Related Videos: 3090
Favorites: 134533
Comments: 209566
--
'''



