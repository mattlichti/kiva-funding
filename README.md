#How to get Loans Funded on Kiva

##Motivation: 

This project is an analysis of what features make loans on Kiva.org more likely to get funded. There are over 2 million small time lenders on kiva who have funded over 880,000 microfinance loans totaling $712 million since 2005. The loans are originally made by 296 different microfinance organizations in 85 countries who then raise money on kiva to backfill the loans. Since January 2012, kiva has usually had a 30 day expiration policy. If a loan is not fully funded in that time, the lenders are refunded and the microfinance organization does not receive any money. This is similar to the funding model used on kickstarter where the project does not receive any money unless it is fully funded. Because of the risk of loan expiration, it is important for the microfinance organizations to understand the characteristics that make loans more likely to get funded. The analysis could also useful for kiva.org to increase the total number of loans getting funded would help more struggling entreprenours around the globe get the loans they need to build their businesses.

##Data Pipeline:

Kiva makes their loan data available through their [api](http://build.kiva.org/). They also periodically make downloadable snapshots of the data. I most recently used the May 18 2015 json snapshot for this analysis. The loan data is in a 1 GB zip file that is 5 GB when unzipped. 

The pipeline.py file is used for processing the raw kiva loan data. It extracts the relevent data, performs the feature engineering, and then stores it all in a postgres SQL database. To run the pipeline, unzip the loans folder containing around 1800 json files that each have the data from 500 loans. Then setup a postgres database and run pipeline.py in the terminal the loans folder location and sql information as command line arguments. 

The most important part of the process is feature engineering. The features include anything that a potential borrower sees when viewing a loan on kiva org that they can use to decide whether or not to loan to a particular individual. A typical view of a loan from the kiva website is pictured below with some of the important features highlighted.

![Kiva Loan](https://github.com/mattlichti/Fundraising-Success/blob/master/img/feature_engineering.jpg)

 The features include the loan amount, repayment term (anywhere from 4 months to several years), gender, group size (loans can be for 1 person or a group of people), whether the borrower is liable for losses due to currency fluctuations, whether the borrower has their name and photo on the website rather than remain anonymous. Categorical variables include country (currently 84 countries), sector (15 categories like transportation or agriculture), and a narrower activity category (150 categories like "rickshaw" or "cattle"). The loans also have various searchable themes like "green", "fair trade", "conflict zones", etc. I used TF-IDF on the one sentence description of how the loan will be used to engineer features out of the most commonly used terms. The length of this text as well as the length of the larger description text were also useful features.


##Modeling:

build_model.py is used to train the model, predict which loans have a higher risk of expiring, and determine which features are most important in predicting loan success. The model converts the categorical features into boolean dummy variables, tokenizes and lemmatizes the text describing the loan use and creates a vector of the 200 most common terms after stop words are removed. I used a random forest which I had to tune quite a bit to avoid overfitting. The classes were unbalanced with a much higher number of funded loans than expired loans, so I heavily weighted the expired loans in order to increase recall at the expense of precision. I also tried logistic regression and SVM but they did not perform quite as well. The model can output a confusion matrix and a list of feature importances which can be make reccomendations on how microfinance organizations can imrove their odds of getting their loans funded. run_model.py is used to run the model. It can be run in the command line like data_pipeline.py.

The plots.py file is used to make plots of some of the important features. Plots of the most important features in the random forest model and the random expiration rate by gender, repayment schedule, and month are in the plot folder.

![Feature Importance](https://github.com/mattlichti/Fundraising-Success/blob/master/plots/feature_importance.jpg)

presentation_slides.pdf is the slides I used for a 3 minute presentation on my project and results.