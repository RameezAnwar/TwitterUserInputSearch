from keyword import iskeyword
import tweepy
import configparser
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import seaborn as sns

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import pandas as pd
import numpy as np
import re
import datetime as dt
import spacy
from spacy import displacy
from collections import Counter
#only the following spacy.cli.download lines the first time 
#running the code if an import error 
#occurs for the spacy.load line
#spacy.cli.download('en_core_web_lg')
#spacy.cli.download('en_core_web_sm')
nlp = spacy.load('en_core_web_lg')



#get credentials

config = configparser.ConfigParser()
config.read('config.ini')


api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

#check if config file working
#print(api_key)

#get aunthentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#####################################################################################



#########################################################################################################3

#a function that gets user input of what they want to search
#then runs word analysis on the tweets obtained

def userSearch():
    nlp = spacy.load('en_core_web_lg')
    limit = 100
    text = []
    isUserSearch = False
    isKeywordSearch = False

    #getting user input about what they want to search
    userinput = input('Do you want to search a user? Type YES or NO\n')

    if userinput == 'YES':
        searchInput = input('What user do you want to search? \n')
        isUserSearch = True

    if userinput == 'NO':
        searchInput = input('What keyword do you want to search? \n')
        isKeywordSearch = True

    #obtaining tweets based off username or keyword
    if isUserSearch == True:
        input_tweets = tweepy.Cursor(api.user_timeline,  screen_name = searchInput, 
        count = 200, tweet_mode = 'extended').items(limit)
    if isKeywordSearch == True:
        input_tweets = tweepy.Cursor(api.search_tweets,  q = searchInput, 
        count = 200, tweet_mode = 'extended').items(limit)

    #remove retweets
    for tweet in input_tweets:
        if( tweet.retweeted == False and 'RT @' not in tweet.full_text): 
            text.append(tweet.full_text)

    input_df = pd.DataFrame( {'text': text })
    #getting just the words
    input_df = input_df['text']


    sentences = []

    for word in input_df:
        sentences.append(word)

    #print('SENTENCES')
    #print(sentences)


    #splitting list into single words
    lines = list()
    for line in sentences:
        words = line.split()
        for w in words:
            lines.append(w)

    #print('LINES')
    #print(lines)


    #removing punctuation
    lines = [re.sub( r'[^A-Za-z0-9]+', '', x) for x in lines]

    lines2 = []
    for words in lines:
        if words != '':
            lines2.append(words)

    #print('LINES2')
    #print(lines2)


    #using stem to make words into root version
    s_stemmer = SnowballStemmer( language = 'english')

    stem = []
    for word in lines2:
        stem.append( s_stemmer.stem(word))

    #print('STEM')
    #print(stem)

    #remove stop words 
    stem2 = []
    for word in stem:
        if word not in nlp.Defaults.stop_words:
            stem2.append(word)

    #print('STEM2')
    #print(stem2)


    #now that we have the data for analysis time to store it in 
    #a new dataframe to use for visualizations
    input_df2 = pd.DataFrame(stem2)
    #group words on count
    input_df2 = input_df2[0].value_counts()

    #print( analysis_df2)

    #getting frequencies of words
    freqdetect = FreqDist()

    for words in input_df:
        freqdetect[words] += 1

    #word count analysis 
    #doing visualizations
    input_df2 = input_df2[:30,]
    plt.figure( figsize = (10,5))
    sns.barplot( x = input_df2.values, y = input_df2.index, alpha = 0.8)

    plt.title('Top Words Used')
    plt.ylabel('Word From Tweet', fontsize = 12)
    plt.xlabel('Count Of Words', fontsize = 12)
    #plt.show()


    #hange spacy load
    nlp = spacy.load('en_core_web_sm')


    def show_ents(doc):
        if doc.ents:
            for ent in doc.ents:
                print(ent.text + ' - ' + ent.label_ + ' - ' + str(spacy.explain(ent.label_)))


    #getting organizations from tweets
    str1 = " "
    stem2 = str1.join(lines2)

    stem2 = nlp(stem2)

    label = [ (X.text, X.label_) for X in stem2.ents]

    input_df3 = pd.DataFrame( label, columns = ['Word', 'Entity'])

    org_df = input_df3.where( input_df3['Entity'] == 'ORG')
    org_df = org_df['Word'].value_counts()
    #org visualization
    org_df = org_df[:20,]
    plt.figure( figsize=(10,5))
    sns.barplot( x = org_df.values, y = org_df.index, alpha = 0.8)
    plt.title('Top Organizations Mentioned')
    plt.ylabel('Word From Tweet', fontsize = 12)
    plt.xlabel('Count Of Words', fontsize = 12)
    plt.tight_layout()
    #plt.show()



    #getting people mentioned 
    str2 = " "
    stem2 = str2.join(lines2)

    stem2 = nlp(stem2)

    label2 = [ (X.text, X.label_) for X in stem2.ents]

    input_df4 = pd.DataFrame( label2, columns = ['Word', 'Entity'])

    person_df = input_df4.where( input_df4['Entity'] == 'PERSON')
    person_df = person_df['Word'].value_counts()


    #people visualization
    person_df = person_df[:20,]
    plt.figure( figsize=(10,5))
    sns.barplot( x = person_df.values, y = person_df.index, alpha = 0.8)
    plt.title('Top People Mentioned')
    plt.ylabel('Word From Tweet', fontsize = 12)
    plt.xlabel('Count Of Words', fontsize = 12)
    plt.tight_layout()
    plt.show()



###############################################################################################


#running the function
userSearch()