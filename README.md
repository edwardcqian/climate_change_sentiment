# Climate Change Sentiment Analysis 

This repository was created as a companion to [this article](https://medium.com/@edward.c.qian/using-machine-learning-to-measure-user-sentiment-towards-climate-change-d817c21c5887).

## Obtaining the Data

The `data` folder contains labelled tweets based on their sentiment towards climate change: 

* 2 - the tweet links to factual news about climate change 
* 1 - the tweet supports the belief of man-made climate change 
* 0 - the tweet neither supports nor refutes the belief of man-made climate change 
* -1 - the tweet does believe in man-made climate change 

The Tweepy folder contains a script which can be used to obtain these tweets. Human labour will be needed to label them. To use the script, specify keywords in  `my_queries.txt` (each keyword is separated by newline). Only tweets containing those keywords will be pulled. 

Then run in the directory the repo is cloned to:
`cd data && python twitter_query.py`

## Building the Model

Download the pre-trained embedding vectors from the Stanford GloVe page and extract it to the repo directory.  

`wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip glove.twitter.27B.zip`

Run the model.py script in a terminal which will train the model, output its weights and also test it against the testing set. 

The weights of a trained model is stored in the `weights` folder. 
