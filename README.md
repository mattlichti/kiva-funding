#[How to get loans funded on kiva](http://mattlichti.github.io/kiva-fundraising-success/)

This project is an analysis of what features make microfinance loans on [kiva.org](kiva.org) more likely to get funded. The full analysis and explanation of my process is [here](http://mattlichti.github.io/kiva-fundraising-success/)


##Motivation: 

 There are over 2 million small time lenders on kiva who have funded over 880,000 microfinance loans since 2005. The loans are originally made by 296 different microfinance organizations who use kiva to raise money to backfill some of their loans. Lenders can view the loans on the website and lend as little as $25 to any of the borrowers they chose. Loans usually have a 30 day expiration policy, which means that loans that are not fully funded in that time are refunded and the microfinance organization does not receive any money. Because of this, it is important for the microfinance organizations to understand the characteristics that make loans more likely to get funded.

##Data:

Kiva makes their loan data available through their [api](http://build.kiva.org/). They also periodically make [downloadable snapshots](http://build.kiva.org/docs/data/snapshots) of the data. I most recently used the May 18 2015 json snapshot for this analysis. The loan data is in a 1 GB zip file that is 5 GB when unzipped. 

###Pipeline:

[data_pipeline.py](https://github.com/mattlichti/kiva-fundraising-success/blob/master/data_pipeline.py) is used for processing the raw kiva loan data. It extracts the relevant data, performs the feature engineering, and then stores it all in a postgres SQL database. To run the pipeline, unzip the loans folder containing around 1800 json files that each have the data from 500 loans. Then setup a postgres database and run pipeline.py in the terminal the loans folder location and sql information as command line arguments. The most important part of the process is feature engineering. The features include anything that a potential borrower sees when viewing a loan on kiva org that they can use to decide whether or not to loan to a particular individual. 


##Modeling:
[build_model.py](https://github.com/mattlichti/kiva-fundraising-success/blob/master/build_model.py) is used to train the model, predict which loans have a higher risk of expiring, and determine which features are most important in predicting loan success. The model converts the categorical features into boolean dummy variables, and tokenizes, lemmatizes, and performs TF-IDF on the text. I used a random forest model. The classes were unbalanced with a much higher number of funded loans than expired loans, so I heavily weighted the expired loans in order to increase recall of expired loans at the expense of precision. The model can output a confusion matrix and a list of feature importance which could be used as recommendations on how to improve their odds of getting their loans funded.  

### Running the model
[run_model.py](https://github.com/mattlichti/kiva-fundraising-success/blob/master/run_model.py) is used to load the relevant data from the postgres sql database and run the model on that data. It can be run from the command line like data_pipeline.py.

## Plotting
[plots.py](https://github.com/mattlichti/kiva-fundraising-success/blob/master/plots.py) is used to make plots of the feature importance of the most important features in the random forest model, as well as plot the expiration rate based on a variety of features and the expiration rate over time.