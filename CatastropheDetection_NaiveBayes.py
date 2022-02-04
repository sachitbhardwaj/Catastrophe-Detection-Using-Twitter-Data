import numpy as np 
import pandas as pd 
import os
import string 
import nltk 
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re 
import demoji
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB


data = pd.read_csv("Data_tweets.csv")
X = data[["id", "keyword", "location", "text"]] 
y = data[["id","target"]] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("Training Data", len(y_train))
print("Testing Data", len(y_test))

def Url_Removal(string):
    return re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|\-)*\b', '', string)

X_train['text'] = X_train['text'].apply(Url_Removal)

def Tag_Removal(string):
    pattern = re.compile(r'[@|#][^\s]+')
    matches = pattern.findall(string)
    tags = [match[1:] for match in matches]
    string = re.sub(pattern, '', string)
    return string + ' ' + ' '.join(tags) + ' '+ ' '.join(tags) + ' ' + ' '.join(tags)

X_train['text'] = X_train['text'].apply(Tag_Removal)

demoji.download_codes()
def Emoji_Removal(string):
    return demoji.replace_with_desc(string)

X_train['text'] = X_train['text'].apply(Emoji_Removal)

def HTML_Removal(string):
    return re.sub(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', str(string))

X_train['text'] = X_train['text'].apply(HTML_Removal)

stemmer  = SnowballStemmer('english')

stopword = stopwords.words('english')
def Stopping_Stemming_Words_Removal(string):
    string_list = string.split()
    return ' '.join([stemmer.stem(i) for i in string_list if i not in stopword])

X_train['text'] = X_train['text'].apply(Stopping_Stemming_Words_Removal)

def UC_Removal(string):
    thestring = re.sub(r'[^a-zA-Z\s]','', string)
    thestring = re.sub(r'\b\w{1,2}\b', '', thestring)
    return re.sub(' +', ' ', thestring)

X_train['text'] = X_train['text'].apply(UC_Removal)

def Iterate_Over_Rows(data):
    df_list = []
    for row in data.itertuples():
        df_dict = {}
        keyword = re.sub(r'[^a-zA-Z\s]','', str(row[2]))
        location = re.sub(r'[^a-zA-Z\s]','', str(row[3]))
        keyword = re.sub(r'\b\w{1,2}\b', '', keyword)
        location = re.sub(r'\b\w{1,2}\b', '', location)
        text = str(row[4])

        if keyword == 'nan':
            if location == 'nan':    
                prs_data = text
            else:
                prs_data = location + ' ' + text
        else:
            if location == 'nan':    
                prs_data = keyword + ' ' + text
            else:
                prs_data = keyword + ' ' + location + ' ' + text                
            
        prs_data = re.sub(' +', ' ', prs_data) 
            
        df_dict['Cleaned_data'] = prs_data
            
        df_list.append(df_dict)
                 
    return pd.DataFrame(df_list)

X_train = Iterate_Over_Rows(X_train)

X_test['text'] = X_test['text'].apply(Url_Removal)
X_test['text'] = X_test['text'].apply(Tag_Removal)
X_test['text'] = X_test['text'].apply(Emoji_Removal)
X_test['text'] = X_test['text'].apply(HTML_Removal)
X_test['text'] = X_test['text'].apply(Stopping_Stemming_Words_Removal)
X_test['text'] = X_test['text'].apply(UC_Removal)
X_test = Iterate_Over_Rows(X_test)

y_train = y_train['target']
y_test = y_test['target']

vectorizer = TfidfVectorizer(min_df = 0.0005, 
                             max_features = 100000, 
                             tokenizer = lambda x: x.split(),
                             ngram_range = (1,4))


X_train = vectorizer.fit_transform(X_train['Cleaned_data'])
X_test = vectorizer.transform(X_test['Cleaned_data'])

#Gaussian
Model1 = GaussianNB()
Model1.fit(X_train.toarray(), y_train)
y_pred1 = Model1.predict(X_test.toarray())

print("Gaussian Accuracy", accuracy_score(y_test, y_pred1))
print("Gaussian Weighted Precision, Recall, F-Score & Support1")
print(precision_recall_fscore_support(y_test, y_pred1, average='weighted'))

#Bernoulli 
Model2 = BernoulliNB()
Model2.fit(X_train.toarray(), y_train)
y_pred2 = Model2.predict(X_test.toarray())

print("Bernoulli Accuracy", accuracy_score(y_test, y_pred2))
print("Bernoulli Weighted Precision, Recall, F-Score & Support1")
print(precision_recall_fscore_support(y_test, y_pred2, average='weighted'))

#Complement
Model3 = ComplementNB()
Model3.fit(X_train.toarray(), y_train)
y_pred3 = Model3.predict(X_test.toarray())

print("Complement Accuracy", accuracy_score(y_test, y_pred3))
print("Complement Weighted Precision, Recall, F-Score & Support1")
print(precision_recall_fscore_support(y_test, y_pred2, average='weighted'))

#Multinomial
Model4 = MultinomialNB()
Model4.fit(X_train.toarray(), y_train)
y_pred4 = Model4.predict(X_test.toarray())

print("Multinomial Accuracy", accuracy_score(y_test, y_pred4))
print("Multinomial Weighted Precision, Recall, F-Score & Support1")
print(precision_recall_fscore_support(y_test, y_pred4, average='weighted'))

