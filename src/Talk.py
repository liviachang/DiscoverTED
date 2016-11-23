from src.utils import *

class Talk(object):
  def __init__(self):
    talks_df = pd.read_csv(TALK_DATA_FN)
    talks_df.tid = talks_df.tid.astype(str)
    talks_df = talks_df.set_index('tid')

    self.ratings = talks_df.ix[:,RATING_TYPES]
    self.info = talks_df.ix[:,INFO_COLS]

  def print_talk(self, tid):
    LINE_LENGTH = 100

    info = self.info.ix[tid]
    themes = info.related_themes

    ratings = self.ratings.ix[tid]
    ratings = ratings[np.argsort(ratings)[::-1][:3]]
    ratings = ', '.join( ratings.reset_index().apply(
      lambda x: '{}:{:.0f}%'.format(x[0], np.round(x[1]*1e2,0)), axis=1) )

    msg = '\n====={}: {} ({})=====\n{}\n[keywords] {}\n[ratings] {}'.format(\
        info.speaker, info.title, info.ted_event, #rtid,
        textwrap.fill(info.description, LINE_LENGTH), \
        textwrap.fill(info.keywords.replace('[','').replace(']',''), LINE_LENGTH),
        ratings)

    if not isinstance(themes, float):
      msg = '{}\n[themes] {}'.format(msg, 
        re.sub('\[|\]|u\'|\'|\"|u\"', '', themes))

    print msg


if __name__ == '__main__':
  talks = Talk()
