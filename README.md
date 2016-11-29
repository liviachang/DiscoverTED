# DiscoverTED: A TED Talk Recommender to Learn Deeper and Wider

This is a 2-week capstone project for my Galvanize Data Science Immersive program. 
The goal is to build a TED talk recommender for people to learn deeper and
wider topics. The project is still working in progress.

While I am trying to be an expert in data science, I would also like to keep an
eye on the world. Learning some new topics stimulate my thoughts and adds
diversity to my life! TED.com is one of my favorite resources to learn. Thus, I
would like to build the TED recommender that helps me explore new topics that I
will potentially like.

[TODO] Add Demo Screenshot

## Methodology
- Model Topics of Talks: model topics from descriptions of talks and users' favorite talks via
    - Non-Negative Matrix Factorization (NMF)
    - Natural language processing (NLP) + Latent Dirichlet Allocation (LDA)
- Model Groups of Users: model groups from users' topic favorite topics
    - ***Deeper topics*** are defined as a user's top "d" topics based on his/her
      favorite talks
    - ***Wider topics*** are defined as the next top "w" topics based on average topic
      rankings of the target user's peers. 
    - Peers are defined as other users with same deeper topics as the target user.
- The final recommended talks are the talks with most similar ratings (say, x% informative + 
  y% Funny + z% Inspiring) as ratings of the users' favorite talks.

## Evaluation
Evaluation can be challenging for the recommendation system. One way to
evaluate my recommender is to ask: is my recommender bringing the topics a user
is interested?



## Data
The data is mainly sourced from [Idiap TED dataset](https://www.idiap.ch/dataset/ted) 
and [ted.com](https://ted.com). 
- Idiap: Scraped as of Sep 2012. Includes data of both users and talks (except talk ratings).
- Rating: Scraped as of Nov 2016 from ted.com
- Exploratory Data Analysis (EDA):
    - Total 12,401 users
    - Total 6,449 active users (users with 4+ favorite talks). 52% of total users
    - Total 2,318 talks (1,201 talks are favorited)
    - Average # favorate talks per user = 9.3
    - Average # users per talk favorited = 84.3

## Acknowledge
- Nikolaos Pappas, Andrei Popescu-Belis, "Combining Content with User
  Preferences for TED Lecture Recommendation", 11th International Workshop on
  Content Based Multimedia Indexing, Veszpré Hungary, IEEE, 2013 
  [PDF](http://publications.idiap.ch/downloads/papers/2013/Pappas_CBMI_2013.pdf)
  [Bibtex](http://publications.idiap.ch/index.php/export/publication/2564/bibtex)
