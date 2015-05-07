import pandas as pd
import matplotlib.pyplot as plt
import pipe
import filter
import build_model
import seaborn

def load_transformed():
    df = pd.io.pickle.read_pickle('data/transformed2.pkl')
    return df

def load_untrans():
    df = pd.io.pickle.read_pickle('data/2014_df.pkl')
    return df

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

def repayment(df):
    '''try running predict on irregular loans with all the payment interval
    info dropped to see if model can identify it as high risk in other ways'''
    df = load_untrans()
    months = filter.get_months(df)
    print "months"
    mod = build_model.FundingModel()
    months = filter.get_months(df)
    print "starting"
    for month in months:
        df = mod.transform_training(month)
        print df[df['repayment_interval: Irregularly']].expired.mean()
        print df[df['repayment_interval: Monthly']].expired.mean()
        print df[df['repayment_interval: At end of term']].expired.mean()
        print ""
     # print df[~df['sector: Agriculture'] & df['repayment_interval: Irregularly']].expired.mean()

def themes(df):
    print df[df['theme: none']].expired.mean()
    print df[~df['theme: none']].expired.mean()
    print df[~df['theme: Water and Sanitation']].expired.mean()

def bla(df):
    print df[df['use: pay']>0].expired.mean()
    print df[df['use: pay']==0].expired.mean()
    print df[df['use: purchase']>0].expired.mean()
    print df[df['use: purchase']==0].expired.mean()
    print df[df.use_text_len > 190].expired.mean()
    print df[df.use_text_len < 190].expired.mean()
    print df[df['use: additional']>0].expired.mean()






