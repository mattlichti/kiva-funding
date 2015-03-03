import pandas as pd
import json
import glob
import os

def import_loans():
    lst = []
    os.chdir("/home/matt/gdrive/final_project/Fundraising-Success/data/loans")
    for file in glob.glob("*.json"):
        f = open(file)
        dic = json.loads(f.readline())
        lst +=(dic['loans'])
    return lst

def get_country(df):
    ldf = pd.DataFrame(list(df.location))
    df['country'] = ldf['country_code']
    df.drop('location',axis=1,inplace=True)
    return df  

def get_desc(df):
    ddf = pd.DataFrame(list(df.description))
    tdf = pd.DataFrame(list(ddf.texts))
    df['description'] = tdf['en']
    return df

def payment_terms(df):
    terms = pd.DataFrame(list(df.terms))
    df['repayment_interval'] = terms['repayment_interval']
    df['repayment_term'] = terms['repayment_term']
    curr_loss = pd.DataFrame(list(terms.loss_liability))
    df['currency_loss'] = curr_loss.currency_exchange
    df.drop('terms',axis=1,inplace=True)
    return df

def borrower_info(df):
    df['group_size'] = df.borrowers.map(lambda x: len(x))
    df['gender'] = df.borrowers.map(lambda x: x[0]['gender'])
    df.drop('borrowers',axis=1,inplace=True)
    return df

def build_df(lst):
    df = pd.DataFrame(lst)
    droplist = ['basket_amount','currency_exchange_loss_amount','delinquent', 'paid_date', 'paid_amount', 'journal_totals', 'payments']
    droplist += ['lender_count', 'funded_date','funded_amount']
    droplist += ['translator', 'video']
    df.drop(droplist,axis=1,inplace=True)

    df = get_country(df)
    df = get_desc(df)
    df = payment_terms(df)
    df = borrower_info(df)
    return df

def dump(df):
    loans = df.to_json()
    with open('data.json', 'w') as outfile:
        json.dump(loans, outfile)