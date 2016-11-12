# HappyLearning: A TED Talk Recommender You Haven't Liked

This capstone project is to build a TED talk recommender for people to learn
new topics.

While I am trying to be an expert in data science, I would also like to keep an
eye on the world. Learning some new topics stimulate my thoughts and adds
diversity to my life! TED.com is one of my favorite resources to learn. Thus, I
would like to build the TED recommender that helps me explore new topics that I
will potentially like.

## Proposed Process Flow
- Exploratory data analysis (EDA)
- Matrix Factorization to find peers for a given user
- Analyze peers' latent features to find one ***potential new topic*** a user may like mostly
- Analyze the potential new topic to find ***potential talks*** to recommend
- The final recommended talk is the talk with the ratings (say, x% informative + 
  y% Funny + z% Inspiring) most similar to those of a user's favorites talk. 


## Data
The data is mainly sourced from [Idiap TED dataset](https://www.idiap.ch/dataset/ted) 
and [ted.com](https://ted.com). 
- Idiap's dataset includes both user and talk data. It was scraped as of Sep 2012.
It does not include rating data of talks.
- Rating data was scraped from ted.com as of Nov 2016.

## Acknowledge
- Nikolaos Pappas, Andrei Popescu-Belis, "Combining Content with User
  Preferences for TED Lecture Recommendation", 11th International Workshop on
  Content Based Multimedia Indexing, Veszpré Hungary, IEEE, 2013 
  [PDF](http://publications.idiap.ch/downloads/papers/2013/Pappas_CBMI_2013.pdf)
  [Bibtex](http://publications.idiap.ch/index.php/export/publication/2564/bibtex)
