
"""
Created on Sun Nov 18 12:12:02 2018

@author: aninyams
"""


"""
Spyder Editor

This is a temporary script file.
"""
# importing the libraries
import pandas as pd 
import nltk
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score


#reading in the csv file. To do this I had to include the "ISO ISO-8859-1 because the file was not reading in correctly.
#Since the file ended up creating addtional columns, I had to drop them as they were not necessary 

data= pd.read_csv('spam.csv', encoding = "ISO-8859-1")
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
data = data.rename(columns={'v1': 'label','v2': 'text'})

#viewing the top 6 rows of the dataset

data.head( n=6)

#setting spam to 1 and ham to zero 
for i in range(0, len(data)):
    if data.label.loc[i]=='spam':
        data.label.loc[i]=1
    else:
        data.label.loc[i]=0

        
         
#Cleaning up the data and removing the stopwords, punctuations, converting uppercase letters to lowercase and keepting the "clean words"
def cleanup(text):
    

    rempunct = [char for char in text if char not in string.punctuation]
   
    rempunct = ''.join(rempunct)  
   
    cleaned = [word for word in rempunct.split() if word.lower() not in stopwords.words('english')]    
    return cleaned


 
data['text'].apply(cleanup).head()
    
#splitting the dataset into 2:traning and test and setting the test size to 0.33
data_train, data_test, X_train, X_test = train_test_split(data['text'],data['label'],test_size= 0.3)

   # Converting/changing strings to integers, counts to weighted if-idf and using the multinomialNB classifier 
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=cleanup)), 
    ('tfidf',TfidfTransformer()), 
    ('classifier',MultinomialNB()) 
])
    
   ##testing to see if the actually works (up to this point) and sending the input to fit the data 
pipeline.fit(data_train,X_train)
    

predict = pipeline.predict(data_test)
 
#getting the F1 score using the classification_report library from sklearn.metrics    
print(classification_report(X_test,predict))




































 



    
 
