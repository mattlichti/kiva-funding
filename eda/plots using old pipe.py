import pandas as pd
import matplotlib.pyplot as plt
import pipe
import filter
import build_model
import seaborn

def expired_plot(df):
    '''plot expiration rate by month for 2014'''
    months = filter.get_months(df)
    fundedm = []
    expirem = []
    expper = []
    for month in months:
        expired = len(month[month.expired])
        funded = len(month[~month.expired])
        fundedm.append(funded)
        expirem.append(expired)
        exp = expired/float(expired+funded)
        expper.append(exp)
    plt.bar(range(1,12),expper)
    plt.ylabel('expiration rate')
    plt.xticks(range(1,12),['Jan','Feb','Mar','April','May','June','July','Aug','Sep','Oct','Nov'])
    plt.show()

def payment_int(df):
    monthly = df[df['repayment_interval: Monthly']].expired.mean()
    irreg = df[df['repayment_interval: Irregularly']].expired.mean()
    end = df[df['repayment_interval: At end of term']].expired.mean()
    plt.bar([0,1,2],[monthly,irreg, end])
    plt.xticks([0.5,1.5,2.5],['Monthly','Irregularly', 'End of Term'])
    plt.ylabel('Expiration Rate')
    plt.xlabel('Repayment Schedule')
    # fig = plt.gcf()
    # fig.set_size_inches(1.5,1.5)
    # fig.savefig('test.png',dpi=200)
    plt.show()
    # fig.show()

def gender(df):
    male = df[~df.gender].expired.mean()
    female = df[df.gender].expired.mean()
    plt.bar([0,1],[male,female])
    plt.xticks([0.5,1.5],['Male','Female'])
    plt.ylabel('Expiration Rate')
    plt.show()

def feat_imp(df):
    mod = build_model.FundingModel()
    mod.fit(df)
    mod.feat_imp()

def expired_plot(df):
    funded = (df[df.status!='expired'].posted_date.values - np.datetime64(44, 'Y'))/np.timedelta64(1, 'D')/365 + 2014
    expired = (df[df.status=='expired'].posted_date.values - np.datetime64(44, 'Y'))/np.timedelta64(1, 'D')/365 + 2014
    z = list(np.arange(2012.1,2015,1./12))
    plt.hist(funded,bins=z)
    plt.hist(expired,bins=z)
    plt.show()

def expr_time(df):
    plt.scatter((df.posted_date.values - np.datetime64(44, 'Y'))/np.timedelta64(1, 'D')/365 + 2014,df.days_available)
    plt.show()







