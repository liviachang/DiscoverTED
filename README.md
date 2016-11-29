# DiscoverTED: A TED Talk Recommender to Learn Deeper and Wider

This is a 2-week capstone project for my Galvanize Data Science Immersive program. 
The goal is to build a TED talk recommender for people to learn deeper and
wider topics. The project is still working in progress.

While I am trying to be an expert in data science, I would also like to keep an
eye on the world. Learning some new topics stimulate my thoughts and adds
diversity to my life! [TED.com](https://www.ted.com/) is one of my favorite resources 
to learn. Thus, I would like to build the TED recommender that helps me explore new 
topics that I will potentially like.

Here is a sample result from DiscoverTED.
![Informative DS Talk](img/sample_result_ds.png)

## Data
There are two datasets used in DiscoverTED: **talk data** and **user-talk data**

### Talk Data
Talk data are scraped from [TED.com](https://www.ted.com/). Fields include
titles, tags, description, talk types, related themes,...etc. Data was scraped
as of Oct 2016.
- Total 2,318 talks
- On average, there are 84.3 users per favorited talk

### User-Talk Data
User-talk data are sourced from [Idiap TED dataset](https://www.idiap.ch/dataset/ted). 
Fields include user IDs, favorite talks. Data was extracted as of Sep 2012.
- Total users: 12,401
- Active users w/ 4+ favorite talks: 6,449 (52% of total users)
- Favorited talks: 1,201
- On average, there are 9.3 favorite talks per user

## Methodology
DiscoverTED is an ensamble recommender based on both **talk-talk recommender** for
talks to learn deeper and **user-user recommender** for talks to learn wider.
Users have to enter some keywords to describe their interested topics (say,
"machine learning big data") and target talk types (say, "Informative" talks).
DiscoverTED will recommend two talks to learn deeper and two talks to learn
wider.

### Talk-Talk Recommender for Deeper Topics
For the talk-talk recommender, the talks are modeled into `k` topics based on
their description and tags. The recommended talks are the most similar talks
based on users' inputs.

To model the talk topics, I choose to use 
**Natural language processing (NLP) + Latent Dirichlet Allocation (LDA)** 
on talk descriptions. These texts are tokenized and stemized. Punctuation and 
stop words are removed. This model gives better recommendations than matrix
factorization (MF) as MF does not work well when the input data is very sparse.
The MF models I have tried include non-negative MF and graphlab's MF.

Pipeline for the talk-talk recommender.
![Pipeline for Talk-Talk Reommender](img/talk_talk_rec.png)


### User-User Recommender for Wider Topics
For the user-user recommender, users are modeled into groups based on their
interested keywords and preferred talk types, and representative talks are 
picked for each topic based on the topic modeling results. 
The target "wider" topics are the next favorite `w` topics of all users in the same group. 
The recommended talks are the most similar talks to users' input 
from the representative talks of the wider topics.

To model the user groups, I choose to use **Nearest Neighbor** based on users'
keywords and preferred talk types. For each user, `n` nearest existing users
are identified as the peers. Similar to the talk-talk recommender, this model 
gives better recommendations than matrix factorization (MF) 
as MF does not work well when the input data is very sparse. 
Also, it works better than k-mean clustering as users around the clustering boundary
may be more closer to the users on the other side of the boundary than the
users with the same clustered labels.

Pipeline for the user-user recommender.
![Pipeline for User-User Reommender](img/user_user_rec.png)

## Evaluation
Evaluation can be challenging for the recommendation system. 
One way to evaluate DiscoverTED is to ask: 
1. Compared to the *random recommendations*, are recommended talks from DiscoverTED
  closer to users' favorite talks? **Yes!!**

  I measured Euclidean distance from each of the recommended
  talk to its closest favorite talk, based on numeric features of talks (i.e. scores
  for each topic and talk types). 
    - The smaller the distance, the closer the recommended talk to the favorite
      talks
    - The average distance of the random recommender: 1.01
    - The average distance of DiscoverTED talk-talk recommender: 0.84
    - The average distance of DiscoverTED ensamble recommender: 0.89

  Both DiscoverTED talk-talk only recommender and DiscoverTED ensamble
  recommender have smaller distances than the random recommender. As expected,
  the distance of DiscoverTED ensamble recommender is slightly larger than the
  talk-talk recommender because the ensamble recommender include
  recommendations for the new and wider topics which may not be liked by users.

2. Compared to the *"deeper" topics only*, do "wider" topics improve coverage in
  users' favorite talks? **Yes!!**

  I measured Euclidean distance from each of the favorite talk to its
  recommended talk, based on numeric features of talks (i.e. scores
  for each topic and talk types). 
    - The smaller the distance, the closer the favorite talk to its closest recommended talk
    - The average distance of DiscoverTED talk-talk only recommender: 1.17
    - The average distance of DiscoverTED ensamble recommender: 1.11

  DiscoverTED ensamble recommender has smaller distance than DiscoverTED
  talk-talk only recommender, implying adding "wider-topic" recommendations
  help capture users' favorite talks.


## Acknowledge
Nikolaos Pappas, Andrei Popescu-Belis, "Combining Content with User
Preferences for TED Lecture Recommendation", 11th International Workshop on
Content Based Multimedia Indexing, Veszpré Hungary, IEEE, 2013 
[PDF](http://publications.idiap.ch/downloads/papers/2013/Pappas_CBMI_2013.pdf)
[Bibtex](http://publications.idiap.ch/index.php/export/publication/2564/bibtex)
