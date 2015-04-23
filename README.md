#How to get Loans Funded on Kiva

####Matt Lichti

##Summary: 

My project is an analysis of what makes a campaign successful on Kiva.org. Over 840,000 microfinance loans have been funded on Kiva since it was founded in 2005. The loans on kiva.org are available for funding on the site for 30 days. If they are not funded in that time, the loan expires and the lenders are refunded their money. Because of this, it is important for the microfinance organizations to understand the preferences of kiva lenders to increase the chances that the loans they post get funded. The most important results are a determination of which features positively and negatively impact the chance of getting funded. 

##Data:

Kiva has data on 844,000 loans at http://build.kiva.org/. They archive the public data periodically in a a set of over 1600 json files with 500 loans each at http://s3.kiva.org/snapshots/kiva_ds_json.zip which is 5 GB unzipped.

##Feature Engineering 

![Kiva Loan](https://github.com/mattlichti/Fundraising-Success/blob/master/img/features.jpg)

The features include country (categorical, 84 countries), sector (15 categories like transportation,  agriculture), loan Amount, whether it is an individual or group loan, gender, and the specific activity (150 categories like "rickshaw" or "cattle") . I also used TFIDF and a lemmatizer to vectorize the one sentence description of how the loan will be used. The loans also have various searchable attributes like "green", "fair trade", "conflict zones", etc.

##Process:

The file pipe.py is used to read in the 1600 json files and convert them into pandas dataframes. A lot of the useful data is several dictionaries deep in the json file and needs a lot of preprocessing to extract the useful information. The cleaned up data can then be saved as json files or a pickle with pipe.py The filter.py file is used to choose a date ranges to build and test the models. The timeframe for my model was chosen based on loan expiration times. I determined that kiva changed their expiration policies during the Christmas 2013 and Christmas 2014 seasons, making it hard to compare expiration rates during that timefram to the rest of the year. I used the Jan 1 through Nov 17 2014 timefram to build my model because expiration policies were consistent during the timeframe.

The most important file is model.py which is used to train and test the model. The model converts the categorical features into around 250 dummy variables. It tokenizes and lemmatizes the text describing the loan use and creates a vector of the 250 most common terms after the stop words are removed. I used a weighted random forest which I had to tune quite a bit to avoid overfitting. I also tried logistic regression and SVM but they did not perform quite as well. My model can output a confusion matrix and a list of feature importances, which I use to make reccomendations on how microfinance organizations can imrove their odds of getting their loans funded.

The plots.py file is used to make univariate plots of some of the important features. Plots of the expiration rate by gender, repayment schedule, and month are in the plot folder.

presentation_slides.pdf is the slides for a 3 minute presentation on my project and results.