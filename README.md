#Fundraising Success

####Matt Lichti

####Feb 25, 2015

##Summary: 

My project is an analysis of what makes a campaign successful on peer to peer charitable fundraising platforms. I will probably get all my data from Kiva.org because it is has a very rich data set. Other similar platforms include zidisha, global giving, vittana, watsi, and givology. Over 800,000 microfinance loans have been funded and around 21,000 expired on Kiva since it was founded in 2005. The loans on kiva.org are available for funding the site for 30 days before expiration, so I could also use the number days it took to get funded mas the y-variable in addition to whether the loan was funded or expired. I will do natural language processing and feature engineering on the kiva loan data and run some machine learning algorithms to determine feature importance, and make a recommendations based on the results. 

##Motivation: 

Peer to peer platforms provide a useful new tool for charities to raise money. It is important to understand what motivates people to give to a specific person or cause in order for charities to effectively raise money.

##Deliverables: 

I will make a website with my results and some visualizations, perhaps using plotly. The main results will be a determination of which features positively or negatively impact the chance of getting funded. The features can include words or phrases in the descriptions. I will present a series of lessons from my analysis on what kiva, the field partners, and lenders could do to improve their experience based on my findings. It might be cool to make a recommender that recommends loans to people based on pulling data on their previous loans from the kiva api.

##Data:

Kiva has data on 844,000 loans at http://build.kiva.org/. They archive the public data nightly in a few thousand json files http://s3.kiva.org/snapshots/kiva_ds_json.zip which is 5 GB unzipped. The data on the 290 field partners comes from the api http://api.kivaws.org/v1/partners.json The pictures for all 840k borrowers are on the kiva website and available through the API http://www.kiva.org/lend/844181 

##Feature Engineering 

I think the most important features are the ones that potential lenders can easily see when looking at a page of loans. Here is an example of what someone sees when searching for a loan to fund on kiva:

![Kiva Loan](https://github.com/mattlichti/Fundraising-Success/blob/master/img/kiva.png)
testing 456

The obvious features would be country (categorical, 84 countries), sector (15 categories like transportation,  agriculture), loan Amount, whether it is an individual or group loan, and gender. The specific activity (like "rickshaw" or "pigs") is a bit more complicated because there are hundreds of different activities in the data so I might want to group them together using clustering algorithms. I also want to do some NLP on the one sentence description of what the loan is for.

The photo would be very difficult to categorize but is probably one of the most important feature lenders use when deciding what loan to fund. I will see if there is some pre-existing image recognition software that can detect features like if the person is smiling in the photo, number of people, if there are children or animals in the photo, etc. I could also use services like croudflower or mechanical turk for that, but I would only be able to do a small subset of the data. There are some Kiva lending teams that base their lending specifically on the photos like “Women wearing hats” and “Guys holding fish” so I could create features based on whether a loan received funding from these groups, which I think I can find using the API or json files. 

The loans also have various searchable attributes like "green", "fair trade", "conflict zones", etc, which could be categorical variables. When you click on a loan, you get a lot more information including several paragraphs about the borrower, which I could do some NLP on this as well. In addition, this page contains tons of information and statistics about the field partner (the organization that administers the loan) including a 1 to 5 rating of default risk, various social performance criteria, and average interest rates, and more information about the loan like the repayment schedule. There is a lot of potential feature engineering to do with some of this information.

One problem is that the number of other loans on the website strongly impact whether a loan gets funded and how long it takes to get funded. When there aren't enough loans on the website, all of the loans get funded quickly. The number of lenders also increased greatly over time which reduces funding time if the number of loans doesn't increase at the same rate. I'm not sure yet how to build that into the model.

##Process:

I already have the data in json form and a small portion of it loaded into pandas. A lot of the useful data is several dictionaries deep in the json file and needs some cleaning. I'll do exploratory analysis and model building in python. I may want to use AWS when I train the model because the 5 GB dataset would take forever to train on my laptop. I'll do a grid search with lots of different machine learning algorithms. I'll write up my results and recommendations, make a bunch of visuals, and build a really basic webpage with flask and embed the visuals in it using plotly or something similar that doesn't require learning much javascript. 

##References:

#####Other research projects on Kiva Loan Funding:

http://gauravparuthi.com/kiva-networks/

http://www.rug.nl/research/globalisation-studies-groningen/research/conferencesandseminars/conferences/eumicrofinconf2011/papers/1new.9.theseira.pdf

####Misc

http://www.kiva.org/about/stats

http://www.kiva.org/team/xprd_lns

http://www.kiva.org/updates/kiva/2011/12/11/kiva-launches-social-performance-badges-and-increases-the-information-available-for-your-lending-decisions.html

http://www.ted.com/talks/dan_pallotta_the_way_we_think_about_charity_is_dead_wrong/transcript?language=en
