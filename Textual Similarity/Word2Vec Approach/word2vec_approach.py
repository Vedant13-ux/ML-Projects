import pandas as pd
import nltk 
import gensim
from nltk.tokenize import word_tokenize
from gensim.models.word2vec import Word2Vec
  
# Importing the dataset
dataset = pd.read_csv('Text_Similarity_Dataset.csv')

#Data Preprocessing
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
 
def preprocessing(para):
    para = re.sub(r"won't", "will not", para)
    para = re.sub(r"can\'t", "can not", para)
    para = re.sub(r"\'s", " is", para)
    para = re.sub(r"\'d", " would", para)
    para = re.sub(r"\'ll", " will", para)
    para = re.sub(r"\'t", " not", para)
    para = re.sub(r"\'m", " am", para)
    para = re.sub(r"\'re", " are", para)
    para = re.sub(r"\'ve", " have", para)
    para = re.sub(r"n\'t", " not", para)


    return para

preprocessed_text1 = []
preprocessed_text2 = []
wl=WordNetLemmatizer()

for para in dataset['text1'].values:
    sent = preprocessing(para)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    sent = ' '.join(word for word in sent.split() if word not in set(stopwords.words('english')))
    preprocessed_text1.append(sent.lower().strip())
    


for para in dataset['text2'].values:
    sent = preprocessing(para)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    sent = ' '.join(word for word in sent.split() if word not in set(stopwords.words('english')))
    preprocessed_text2.append(sent.lower().strip())

model_file="GoogleNews-vectors-negative300.bin.gz"
model= gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)


def tokenizer(sent):
    tokens=word_tokenize(sent)
    tokens=[wl.lemmatize(word) for word in tokens]
    return tokens


similarity_score=[]

for i in dataset.index:
    
    text1 = preprocessed_text1[i]
    text2 = preprocessed_text2[i]
    
    if text1==text2:
             similarity_score.append(1.0) # 0 means highly dissimilar
            
    else:   
        text1vecs = tokenizer(text1)
        text2vecs = tokenizer(text2)
        
        vocab = model.key_to_index #the vocabulary considered in the word embeddings
        
        if len(text1vecs and text2vecs)==0:
                similarity_score.append(0.0)

        else:
            
            for word in text1vecs.copy(): #remove sentence words not found in the vocab
                if (word not in vocab):
                        text1.remove(word)

            for word in text2vecs.copy(): #idem
                if (word not in vocab):
                        text2.remove(word)
                        
                        
            similarity_score.append(model.n_similarity(text1, text2)) # as it is given 1 means highly similar & 0 means highly dissimiliar    

final_score = pd.DataFrame({'Unique_ID':dataset['Unique_ID'],
                     'Similarity_score':similarity_score})

final_score.to_csv('similarity_score.csv',index=False)
    
    
    
    
    
    
    
    
    
    