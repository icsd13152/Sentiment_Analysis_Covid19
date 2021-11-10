
import os
import tweepy as tw
import pandas as pd
from datetime import date, timedelta, datetime

def getTweets():
    sysdateminus7 = ( datetime.now()- timedelta(days=7)).date()
    today = date.today()


    consumer_key= 
    consumer_secret= 
    access_token= 
    access_token_secret= 


    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)

    # Define the search term and the date_since date as variables
    search_words = '#Covid19 OR #covid19 OR #Covid-19 OR COVID-19'
    date_since = sysdateminus7
    #twitter has changed his API and we can only get tweets from 1 week ago.
    tweets = tw.Cursor(api.search_tweets,
                       q=search_words,
                       lang="en",
                       since=date_since,
                       until = today).items(200)

    listTweets = list()

    for tweet in tweets:
        listTweets.append(tweet.text)


    tweet_text = pd.DataFrame(data=listTweets,
                              columns=['tweet'])

    return tweet_text

