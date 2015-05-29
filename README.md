#How to get Loans Funded on Kiva

##Motivation: 

This project is an analysis of what features make loans on Kiva.org more likely to get funded. There are over 2 million small time lenders on kiva who have funded over 880,000 microfinance loans totaling $712 million since 2005. The loans are originally made by 296 different microfinance organizations in 85 countries who use kiva to raise money to backfill some of their loans. Lenders can view the loans on the website and lend as little as $25 to any of the borrowers they chose.

Since January 2012, kiva has usually had a 30 day expiration policy. If a loan is not fully funded in that time, the lenders are refunded and the microfinance organization does not receive any money. This is similar to the funding model used on kickstarter where projects do not receive any money unless they are fully funded. Because of the risk of loan expiration, it is important for the microfinance organizations to understand the characteristics that make loans more likely to get funded. The analysis could also help kiva.org increase the total number of loans getting funded, which would help more struggling entrepreneurs around the globe get the capital they need to build their businesses.

##Data Pipeline:

Kiva makes their loan data available through their [api](http://build.kiva.org/). They also periodically make downloadable snapshots of the data. I most recently used the May 18 2015 json snapshot for this analysis. The loan data is in a 1 GB zip file that is 5 GB when unzipped. 

The pipeline.py file is used for processing the raw kiva loan data. It extracts the relevant data, performs the feature engineering, and then stores it all in a postgres SQL database. To run the pipeline, unzip the loans folder containing around 1800 json files that each have the data from 500 loans. Then setup a postgres database and run pipeline.py in the terminal the loans folder location and sql information as command line arguments. 

The most important part of the process is feature engineering. The features include anything that a potential borrower sees when viewing a loan on kiva org that they can use to decide whether or not to loan to a particular individual. A typical view of a loan from the kiva website is pictured below with some of the important features highlighted.

![Kiva Loan](https://github.com/mattlichti/Fundraising-Success/blob/master/img/feature_engineering.jpg)

 The features include the loan amount, repayment term (anywhere from 4 months to several years), gender, group size (loans can be for 1 person or a group of people), whether the borrower is liable for losses due to currency fluctuations, whether the borrower has their name and photo on the website rather than remain anonymous. Categorical variables include country (currently 84 countries), sector (15 categories like transportation or agriculture), and a narrower activity category (150 categories like "rickshaw" or "cattle"). The loans also have various searchable themes like "green", "fair trade", "conflict zones", etc. I used the one sentence description of how the loan will be used to engineer features out of the most commonly used terms. The length of this text as well as the length of the larger description text were also useful features. 

In addition to the features that impact demand for particular loans, the supply of loans on the site can impact the chances of each loan getting funded. I used SQL to calculate the number of other loans on kiva at the time each loan was posted by comparing the timestamp of when each loan was posted to the timestamps for when other loans were posted and funded or expired.

![Kiva Loan](https://github.com/mattlichti/Fundraising-Success/blob/master/plots/competing loans.png)

When there are few loans on the site, almost all of the loans get funded. When there is more competition, lenders have more options each loan has a higher chance of expiring. In the future, it might be useful to look at the total value of the loans currently fundraising and how far along they are in their fundraising, not just the number of loans. It might also be useful to look at attributes of those loans like if the loans are similar to the loan being analyzed, like if there are a lot of other loans from the same country or same economic sector.

##Modeling:

build_model.py is used to train the model, predict which loans have a higher risk of expiring, and determine which features are most important in predicting loan success. run_model.py is used to load the relevant data from the postgres sql database and run the model on that data. It can be run from the command line like data_pipeline.py. 

The model converts the categorical features into boolean dummy variables, and tokenizes, lemmatizes, and performs TF-IDF on the text. I used a random forest model which I had to tune quite a bit to avoid overfitting. The classes were unbalanced with a much higher number of funded loans than expired loans, so I heavily weighted the expired loans in order to increase recall of expired loans at the expense of precision. I also tried logistic regression and SVM but they have not performed quite as well. The model can output a confusion matrix and a list of feature importances which could be used as recommendations on how to improve their odds of getting their loans funded. 

The plots.py file is used to make plots of some of the important features.

##Results:

![Feature Importance](https://github.com/mattlichti/Fundraising-Success/blob/master/plots/feature_importance.png)

Lenders on kiva show a strong preference for loans with a schedule repayment term as well as loans to women. Loans for larger amounts of money are not surprisingly more difficult to fund since they require more lenders to fully fund. Lenders also prefer lending to certain countries, especially African countries and countries with fewer loans available on kiva. Loans to countries with lots of loans available like El Salvador (SV) and Colombia (CO) are at greater risk for loan expiration. Lenders also have strong preferences for different loan uses with loans for personal housing much more likely to expire than loans for education or starting a business.