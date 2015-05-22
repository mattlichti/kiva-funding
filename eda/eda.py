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