# HappyLearning: A TED Talk Recommender to Learn Deeper and Wider

This is a 2-week capstone project for my Galvanize Data Science Immersive program. 
The goal is to build a TED talk recommender for people to learn deeper and
wider topics. 

While I am trying to be an expert in data science, I would also like to keep an
eye on the world. Learning some new topics stimulate my thoughts and adds
diversity to my life! TED.com is one of my favorite resources to learn. Thus, I
would like to build the TED recommender that helps me explore new topics that I
will potentially like.

[TODO] Add Demo Screenshot

## Methodology
- Exploratory data analysis (EDA)
- Topic Modeling: find users' peers based on same top "d" topics (latent features), 
  and find talks' topics
    - Non-Negative Matrix Factorization (NMF)
    - Natural language processing (NLP) + Latent Dirichlet Allocation (LDA)
- Define ***deeper topics*** as the top "d" topics, and define ***wider topics*** as
  the next top "w" topics based on peers' latent features
- The final recommended talk is the talk with the ratings (say, x% informative + 
  y% Funny + z% Inspiring) most similar to those of a user's favorites talk. 

## Evaluation

## Data
The data is mainly sourced from [Idiap TED dataset](https://www.idiap.ch/dataset/ted) 
and [ted.com](https://ted.com). 
- Idiap: Scraped as of Sep 2012. Includes data of both users and talks (except ratings).
- Rating: Scraped as of Nov 2016 from ted.com

## Acknowledge
- Nikolaos Pappas, Andrei Popescu-Belis, "Combining Content with User
  Preferences for TED Lecture Recommendation", 11th International Workshop on
  Content Based Multimedia Indexing, Veszpré Hungary, IEEE, 2013 
  [PDF](http://publications.idiap.ch/downloads/papers/2013/Pappas_CBMI_2013.pdf)
  [Bibtex](http://publications.idiap.ch/index.php/export/publication/2564/bibtex)
