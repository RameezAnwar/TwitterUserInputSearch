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




#test if api connection is working
public_tweets = api.home_timeline()
#for tweet in public_tweets:
#    print(tweet.text)



#checking panda functionality
columns = ['Time', 'User', 'Tweet']
data = []
for tweet in public_tweets:
    data.append([tweet.created_at, tweet.user.screen_name, tweet.text])

#print(data)

#adding tweets to dataframe to be saveable
df = pd.DataFrame(data, columns = columns)
#print(df)

#saving tweets to csv file
#df.to_csv( 'tweets.csv')


###########################################################################################

#getting tweets from specific user
user = 'BarackObama'
limit = 10

#tweepy method of bypassing twitter 200 limit, set limit above 200 if using this version
#user_tweets = tweepy.Cursor(api.user_timeline,  screen_name = user, 
#count = 200, tweet_mode = 'extended').items(limit)

#without using limit bypass
user_tweets = api.user_timeline(screen_name = user,count = limit, tweet_mode = 'extended' )

#for tweet in user_tweets:
#    print( tweet.full_text)


#saving user tweets to dataframe and saving it
user_columns = ['User', 'Tweet']
user_data = []

for tweet in user_tweets:
    user_data.append([tweet.user.screen_name, tweet.full_text])

user_df = pd.DataFrame(user_data, columns = user_columns)

#print(user_df)
#user_df.to_csv( 'user_tweets.csv')


#####################################################################################


#getting tweets off keywords
#can be @username, a word or phrase, or a #hashtag
searching = 'God of War'
limit = 10

#to bypass set limit above 100 and count to 100
keyword_tweets = tweepy.Cursor(api.search_tweets, q = searching, 
count = 10, tweet_mode = 'extended').items(limit)

#without bypass
#keyword_tweets = api.search_tweets(q = searching, count = limit, tweet_mode = 'extended' )

#for tweet in keyword_tweets:
#    print( tweet.full_text)

keyword_columns = ['User', 'Tweet']
keyword_data = []

for tweet in keyword_tweets:
    keyword_data.append([tweet.user.screen_name, tweet.full_text])

keyword_df = pd.DataFrame(keyword_data, columns = keyword_columns)

#print(keyword_df)
#keyword_df.to_csv( 'keyword_tweets.csv')


#########################################################################################################3


#obtaining more data and simple analysis

user = 'BarackObama'
limit = 100

tweets = []
likes = []
time = []

user_tweets2 = tweepy.Cursor(api.user_timeline,  screen_name = user, 
count = 200, tweet_mode = 'extended').items(limit)


#obtaining tweets without retweets
for tweet in user_tweets2:
    if( tweet.retweeted == False and 'RT @' not in tweet.full_text): 
        tweets.append( tweet.full_text)
        likes.append( tweet.favorite_count)
        time.append( tweet.created_at)

user_df2 = pd.DataFrame( {'tweets':tweets, 'likes':likes, 'time':time} )

#print(user_df2)

#getting most liked tweets
mostLiked = user_df2.loc[user_df2.likes.nlargest(10).index]
#print(mostLiked)



#make sure matplotlib working right
#plt.plot([1,2,3,4], [1,4,9,16])
#plt.ylabel('y numbers')
#plt.xlabel('x numbers')
#plt.show()


###############################################################################################

#more in depth semantic/word analysis with spacy using user tweets

#tweets between these dates
begin_data = (2022, 8, 1)
end_date = (2022, 8, 15)

limit = 300
lang = 'english'
text = []

analysis_user = 'BarackObama'
analysis_tweets = tweepy.Cursor(api.user_timeline,  screen_name = analysis_user, 
count = 100, tweet_mode = 'extended').items(limit)

#getting rid of retweets
for tweet in analysis_tweets:
    if( tweet.retweeted == False and 'RT @' not in tweet.full_text): 
        text.append( tweet.full_text)

analysis_df = pd.DataFrame( {'text': text })
#getting only the text
analysis_df = analysis_df['text']
#print(analysis_df)


sentences = []

for word in analysis_df:
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
analysis_df2 = pd.DataFrame(stem2)
#group words on count
analysis_df2 = analysis_df2[0].value_counts()

#print( analysis_df2)

#getting frequencies of words
freqdetect = FreqDist()

for words in analysis_df:
    freqdetect[words] += 1

#word count analysis 
#doing visualizations
analysis_df2 = analysis_df2[:30,]
plt.figure( figsize = (10,5))
sns.barplot( x = analysis_df2.values, y = analysis_df2.index, alpha = 0.8)

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

analysis_df3 = pd.DataFrame( label, columns = ['Word', 'Entity'])

org_df = analysis_df3.where( analysis_df3['Entity'] == 'ORG')
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

analysis_df4 = pd.DataFrame( label2, columns = ['Word', 'Entity'])

person_df = analysis_df4.where( analysis_df4['Entity'] == 'PERSON')
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