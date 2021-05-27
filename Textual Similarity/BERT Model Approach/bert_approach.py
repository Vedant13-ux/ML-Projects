import pandas as pd
import nltk 
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
  
# Importing the dataset
dataset = pd.read_csv('Text_Similarity_Dataset.csv')

#Data Preprocessing
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
 
def preprocessing(para):
    para = re.sub(r"can\'t", "can not", para)
    para = re.sub(r"won't", "will not", para)
    para = re.sub(r"n\'t", " not", para)
    para = re.sub(r"\'re", " are", para)
    para = re.sub(r"\'d", " would", para)
    para = re.sub(r"\'ll", " will", para)
    para = re.sub(r"\'t", " not", para)
    para = re.sub(r"\'s", " is", para)
    para = re.sub(r"\'m", " am", para)
    para = re.sub(r"\'ve", " have", para)
    return para

preprocessed_text1 = []
preprocessed_text2 = []
wl=WordNetLemmatizer()

for para in dataset['text1'].values:
    sent = preprocessing(para)
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = sent.replace('\\r', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    sent = ' '.join(e for e in sent.split() if e not in set(stopwords.words('english')))
    preprocessed_text1.append(sent.lower().strip())
    


for para in dataset['text2'].values:
    sent = preprocessing(para)
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = sent.replace('\\r', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    sent = ' '.join(e for e in sent.split() if e not in set(stopwords.words('english')))
    preprocessed_text2.append(sent.lower().strip())



#Loading the BERT Model
model_name='distilbert-base-nli-mean-tokens'
model=SentenceTransformer(model_name)

def tokenizer(sent):
    tokens=word_tokenize(sent)
    tokens=' '.join([wl.lemmatize(word) for word in tokens]) #Lemmatizing the words in the texts
    sent_vec=model.encode(tokens) #Converting the Text into a Set of Vectors (Sentence Embeddings)
    return [sent_vec]


similarity_score=[]


for i in dataset.index : 
    text1 = preprocessed_text1[i]
    text2 = preprocessed_text2[i]
    
    if text1==text2:
             similarity_score.append(1.0) #1 means highly similar
            
    else:   
        text1vecs = tokenizer(text1)
        text2vecs = tokenizer(text2)
        
        
        if len(text1vecs and text2vecs)==0:
                similarity_score.append(0.0)

        else:
            score=cosine_similarity([text1vecs[0]], [text2vecs[0]])
            similarity_score.append(score[0][0]) # as it is given 0 means highly dissimilar & 1 means highly similar    


#Loading the Dataframe into a CSV File
score = pd.DataFrame({'Unique_ID':dataset['Unique_ID'],
                     'Similarity_score':similarity_score})

score.to_csv('similarity_score.csv',index=False)
    
    
    
    
    
    
    
    
    
    