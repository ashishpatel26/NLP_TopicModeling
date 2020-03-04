## Topic Modeling [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ashishpatel26/NLP_TopicModeling/master?filepath=Topic%20Modeling.ipynb)

This is practice Notebook.

**Article** : https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

**Dataset** :  [Download](https://storage.googleapis.com/kaggle-data-sets/1692/893258/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1583492607&Signature=FeyNnNde5KnsNzkqMRDtl6Re9%2BuwsTISjV5PkpaIGq3c2BAEgoKdLYGzj6k2V6yEUSlXwC0IybkmjNPIhYRlEgN0JQ6swOUOE1tStarjBtxGRlg8KH48mVmCCI1m7NtQjn6pHsbLPNVGX6CQFZ5gFu85HF2lJse6Ow9RQ4R9m71n65TioBp2Yj1nsXAGam5cX%2F85TXNPHwDInWF1JgjzNgVdsro7t0IID4FLatGUnmpn8Zxjb349WBaXLdIDMdKbcBMkhIKDKm6PpVX5dBzCc1%2BkVy3GWBFlLgiziOXe1FTiIPhACINIc0w5thkdFtw5ABncb1GKHT8j%2BDnufDQLoA%3D%3D&response-content-disposition=attachment%3B+filename%3Dmillion-headlines.zip)


```python
from pyforest import *
```


```python
data = pd.read_csv("abcnews-date-text.csv", error_bad_lines=False)
data.head()
```



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>publish_date</th>
      <th>headline_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20030219</td>
      <td>aba decides against community broadcasting lic...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20030219</td>
      <td>act fire witnesses must be aware of defamation</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20030219</td>
      <td>a g calls for infrastructure protection summit</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20030219</td>
      <td>air nz staff in aust strike for pay rise</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20030219</td>
      <td>air nz strike to affect australian travellers</td>
    </tr>
  </tbody>
</table>
</div>


```python
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text
```


```python
print(f"Length of Documents: {len(documents)}")
documents[:5]
```

    Length of Documents: 1186018



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>headline_text</th>
      <th>index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aba decides against community broadcasting lic...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>act fire witnesses must be aware of defamation</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a g calls for infrastructure protection summit</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>air nz staff in aust strike for pay rise</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>air nz strike to affect australian travellers</td>
      <td>4</td>
    </tr>
  </tbody>
</table>




# Data Pre-processing

We will perform the following steps:

- **Tokenization**: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.
- Words that have fewer than 3 characters are removed.
- All **stopwords** are removed.
- Words are **lemmatized** — words in third person are changed to first person and verbs in past and future tenses are changed into present.
- Words are **stemmed** — words are reduced to their root form.


```python
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')
```

    [nltk_data] Downloading package wordnet to C:\Users\Ashish
    [nltk_data]     Patel\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!

    True



### Lemmitize Example


```python
print(WordNetLemmatizer().lemmatize('went', pos='v'))
```

    go


### Stemmer Example


```python
stemmer = SnowballStemmer('english')
original_words = [
    'caresses', 'flies', 'dies', 'mules', 'denied', 'died', 'agreed', 'owned',
    'humbled', 'sized', 'meeting', 'stating', 'siezing', 'itemization',
    'sensational', 'traditional', 'reference', 'colonizer', 'plotted'
]
singles = [stemmer.stem(plural) for plural in original_words]
pd.DataFrame(data={'original word': original_words, 'stemmed': singles})
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>original word</th>
      <th>stemmed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>caresses</td>
      <td>caress</td>
    </tr>
    <tr>
      <th>1</th>
      <td>flies</td>
      <td>fli</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dies</td>
      <td>die</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mules</td>
      <td>mule</td>
    </tr>
    <tr>
      <th>4</th>
      <td>denied</td>
      <td>deni</td>
    </tr>
    <tr>
      <th>5</th>
      <td>died</td>
      <td>die</td>
    </tr>
    <tr>
      <th>6</th>
      <td>agreed</td>
      <td>agre</td>
    </tr>
    <tr>
      <th>7</th>
      <td>owned</td>
      <td>own</td>
    </tr>
    <tr>
      <th>8</th>
      <td>humbled</td>
      <td>humbl</td>
    </tr>
    <tr>
      <th>9</th>
      <td>sized</td>
      <td>size</td>
    </tr>
    <tr>
      <th>10</th>
      <td>meeting</td>
      <td>meet</td>
    </tr>
    <tr>
      <th>11</th>
      <td>stating</td>
      <td>state</td>
    </tr>
    <tr>
      <th>12</th>
      <td>siezing</td>
      <td>siez</td>
    </tr>
    <tr>
      <th>13</th>
      <td>itemization</td>
      <td>item</td>
    </tr>
    <tr>
      <th>14</th>
      <td>sensational</td>
      <td>sensat</td>
    </tr>
    <tr>
      <th>15</th>
      <td>traditional</td>
      <td>tradit</td>
    </tr>
    <tr>
      <th>16</th>
      <td>reference</td>
      <td>refer</td>
    </tr>
    <tr>
      <th>17</th>
      <td>colonizer</td>
      <td>colon</td>
    </tr>
    <tr>
      <th>18</th>
      <td>plotted</td>
      <td>plot</td>
    </tr>
  </tbody>
</table>
</div>




```python
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(
                token) > 3:
            result.append(lemmatize_stemming(token))
    return result
```


```python
doc_sample = documents[documents['index'] == 4310].values[0][0]
```


```python
doc_sample = documents[documents['index'] == 4309].values[0][0]

print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))
```

    original document: 
    ['rain', 'helps', 'dampen', 'bushfires']


​    
​     tokenized and lemmatized document: 
​    ['rain', 'help', 'dampen', 'bushfir']



```python
processed_docs = documents['headline_text'].map(preprocess)
```


```python
processed_docs[:10]
```




    0            [decid, communiti, broadcast, licenc]
    1                               [wit, awar, defam]
    2           [call, infrastructur, protect, summit]
    3                      [staff, aust, strike, rise]
    4             [strike, affect, australian, travel]
    5               [ambiti, olsson, win, tripl, jump]
    6           [antic, delight, record, break, barca]
    7    [aussi, qualifi, stosur, wast, memphi, match]
    8            [aust, address, secur, council, iraq]
    9                         [australia, lock, timet]
    Name: headline_text, dtype: object



### Bag of words on the dataset


```python
dictionary = gensim.corpora.Dictionary(processed_docs)
```


```python
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
```

    0 broadcast
    1 communiti
    2 decid
    3 licenc
    4 awar
    5 defam
    6 wit
    7 call
    8 infrastructur
    9 protect
    10 summit



```python
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
```


```python
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[4309]
```




    [(76, 1), (112, 1), (484, 1), (4038, 1)]




```python
bow_doc_4310 = bow_corpus[4309]

for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(
        bow_doc_4310[i][0], dictionary[bow_doc_4310[i][0]],
        bow_doc_4310[i][1]))
```

    Word 76 ("bushfir") appears 1 time.
    Word 112 ("help") appears 1 time.
    Word 484 ("rain") appears 1 time.
    Word 4038 ("dampen") appears 1 time.


### TF-IDF


```python
from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
```


```python
corpus_tfidf = tfidf[bow_corpus]
```


```python
from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break
```

    [(0, 0.5850076620505259),
     (1, 0.38947256567331934),
     (2, 0.4997099083387053),
     (3, 0.5063271308533074)]


### Running LDA using Bag of Words


```python
lda_model = gensim.models.LdaMulticore(bow_corpus,
                                       num_topics=10,
                                       id2word=dictionary,
                                       passes=2,
                                       workers=2)
```


```python
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
```

    Topic: 0 
    Words: 0.023*"hous" + 0.022*"south" + 0.020*"north" + 0.017*"bushfir" + 0.016*"miss" + 0.013*"interview" + 0.012*"west" + 0.012*"hospit" + 0.011*"coast" + 0.010*"investig"
    Topic: 1 
    Words: 0.032*"kill" + 0.023*"shoot" + 0.021*"protest" + 0.021*"dead" + 0.020*"attack" + 0.020*"polic" + 0.014*"offic" + 0.014*"assault" + 0.011*"michael" + 0.011*"bank"
    Topic: 2 
    Words: 0.057*"australia" + 0.046*"australian" + 0.026*"world" + 0.018*"canberra" + 0.017*"test" + 0.013*"win" + 0.011*"final" + 0.011*"farm" + 0.010*"return" + 0.009*"beat"
    Topic: 3 
    Words: 0.030*"polic" + 0.029*"charg" + 0.026*"court" + 0.024*"death" + 0.024*"murder" + 0.020*"woman" + 0.020*"crash" + 0.017*"face" + 0.016*"alleg" + 0.013*"trial"
    Topic: 4 
    Words: 0.019*"chang" + 0.019*"say" + 0.015*"speak" + 0.015*"power" + 0.013*"worker" + 0.012*"climat" + 0.012*"concern" + 0.011*"flood" + 0.011*"save" + 0.011*"fear"
    Topic: 5 
    Words: 0.021*"market" + 0.020*"news" + 0.018*"women" + 0.018*"live" + 0.016*"tasmania" + 0.013*"high" + 0.013*"rise" + 0.012*"price" + 0.012*"lose" + 0.012*"break"
    Topic: 6 
    Words: 0.035*"elect" + 0.018*"water" + 0.017*"state" + 0.015*"tasmanian" + 0.012*"labor" + 0.011*"liber" + 0.011*"morrison" + 0.010*"parti" + 0.010*"leader" + 0.010*"campaign"
    Topic: 7 
    Words: 0.020*"donald" + 0.014*"farmer" + 0.014*"nation" + 0.013*"time" + 0.013*"rural" + 0.013*"council" + 0.012*"indigen" + 0.011*"school" + 0.011*"commiss" + 0.011*"plan"
    Topic: 8 
    Words: 0.044*"trump" + 0.037*"year" + 0.035*"sydney" + 0.028*"queensland" + 0.022*"home" + 0.021*"adelaid" + 0.018*"perth" + 0.016*"brisban" + 0.015*"leav" + 0.015*"peopl"
    Topic: 9 
    Words: 0.031*"govern" + 0.020*"warn" + 0.018*"feder" + 0.015*"countri" + 0.015*"fund" + 0.014*"claim" + 0.014*"life" + 0.012*"say" + 0.012*"stori" + 0.012*"health"



```python
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf,
                                             num_topics=10,
                                             id2word=dictionary,
                                             passes=2,
                                             workers=4)
```


```python
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
```

    Topic: 0 Word: 0.014*"market" + 0.011*"stori" + 0.009*"tuesday" + 0.009*"share" + 0.009*"friday" + 0.008*"australian" + 0.007*"peter" + 0.007*"dollar" + 0.006*"financ" + 0.006*"novemb"
    Topic: 1 Word: 0.009*"michael" + 0.009*"wall" + 0.008*"street" + 0.007*"inquest" + 0.006*"june" + 0.006*"tree" + 0.006*"know" + 0.006*"hong" + 0.006*"john" + 0.006*"kong"
    Topic: 2 Word: 0.009*"monday" + 0.009*"kill" + 0.009*"thursday" + 0.008*"grandstand" + 0.007*"insid" + 0.006*"syria" + 0.006*"quiz" + 0.006*"babi" + 0.006*"mother" + 0.005*"victorian"
    Topic: 3 Word: 0.026*"news" + 0.013*"interview" + 0.011*"search" + 0.010*"miss" + 0.009*"hobart" + 0.009*"david" + 0.008*"beach" + 0.007*"speak" + 0.007*"busi" + 0.007*"mark"
    Topic: 4 Word: 0.010*"world" + 0.010*"australia" + 0.009*"final" + 0.009*"queensland" + 0.008*"leagu" + 0.007*"scott" + 0.007*"test" + 0.007*"cricket" + 0.006*"beat" + 0.006*"rugbi"
    Topic: 5 Word: 0.018*"polic" + 0.018*"charg" + 0.015*"murder" + 0.012*"woman" + 0.011*"alleg" + 0.010*"court" + 0.010*"crash" + 0.009*"shoot" + 0.009*"jail" + 0.009*"arrest"
    Topic: 6 Word: 0.015*"rural" + 0.014*"countri" + 0.010*"hour" + 0.010*"drum" + 0.009*"govern" + 0.007*"fund" + 0.006*"nation" + 0.006*"water" + 0.005*"drought" + 0.005*"christma"
    Topic: 7 Word: 0.027*"trump" + 0.015*"donald" + 0.009*"commiss" + 0.009*"royal" + 0.008*"turnbul" + 0.007*"farm" + 0.006*"coal" + 0.005*"energi" + 0.005*"malcolm" + 0.005*"say"
    Topic: 8 Word: 0.015*"elect" + 0.008*"labor" + 0.008*"polit" + 0.008*"climat" + 0.008*"liber" + 0.007*"parti" + 0.007*"say" + 0.007*"senat" + 0.006*"chang" + 0.006*"juli"
    Topic: 9 Word: 0.010*"weather" + 0.010*"sentenc" + 0.009*"wednesday" + 0.008*"morrison" + 0.008*"sport" + 0.008*"mental" + 0.007*"zealand" + 0.007*"tasmania" + 0.007*"histori" + 0.007*"outback"



```python
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf,
                                             num_topics=10,
                                             id2word=dictionary,
                                             passes=2,
                                             workers=4)
```


```python
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
```

    Topic: 0 Word: 0.016*"news" + 0.016*"rural" + 0.010*"market" + 0.010*"price" + 0.008*"busi" + 0.007*"nation" + 0.007*"scott" + 0.006*"share" + 0.006*"sport" + 0.006*"mental"
    Topic: 1 Word: 0.012*"bushfir" + 0.010*"weather" + 0.008*"david" + 0.007*"northern" + 0.006*"queensland" + 0.006*"victoria" + 0.006*"april" + 0.005*"territori" + 0.005*"farm" + 0.005*"wild"
    Topic: 2 Word: 0.010*"friday" + 0.010*"john" + 0.009*"violenc" + 0.008*"juli" + 0.007*"andrew" + 0.007*"august" + 0.006*"domest" + 0.006*"cancer" + 0.005*"flight" + 0.005*"patient"
    Topic: 3 Word: 0.011*"final" + 0.010*"world" + 0.009*"australia" + 0.007*"leagu" + 0.007*"cricket" + 0.007*"win" + 0.007*"grandstand" + 0.006*"open" + 0.006*"financ" + 0.006*"australian"
    Topic: 4 Word: 0.018*"countri" + 0.013*"hour" + 0.011*"royal" + 0.010*"commiss" + 0.009*"climat" + 0.007*"care" + 0.006*"zealand" + 0.006*"rugbi" + 0.006*"australia" + 0.006*"age"
    Topic: 5 Word: 0.017*"interview" + 0.016*"crash" + 0.013*"drum" + 0.010*"polic" + 0.009*"street" + 0.008*"tuesday" + 0.008*"coast" + 0.008*"dead" + 0.008*"shoot" + 0.008*"christma"
    Topic: 6 Word: 0.012*"stori" + 0.008*"michael" + 0.006*"council" + 0.006*"jam" + 0.006*"centr" + 0.006*"know" + 0.005*"govern" + 0.005*"money" + 0.005*"quiz" + 0.005*"ash"
    Topic: 7 Word: 0.019*"charg" + 0.018*"murder" + 0.013*"court" + 0.013*"alleg" + 0.013*"polic" + 0.010*"woman" + 0.010*"jail" + 0.010*"sentenc" + 0.009*"guilti" + 0.009*"accus"
    Topic: 8 Word: 0.024*"trump" + 0.013*"elect" + 0.013*"donald" + 0.010*"govern" + 0.008*"labor" + 0.007*"wednesday" + 0.007*"say" + 0.007*"thursday" + 0.007*"liber" + 0.006*"feder"
    Topic: 9 Word: 0.008*"monday" + 0.008*"live" + 0.007*"protest" + 0.006*"kill" + 0.006*"australia" + 0.006*"presid" + 0.006*"turnbul" + 0.006*"novemb" + 0.006*"indonesia" + 0.005*"syria"


## Classification of the topics
Performance evaluation by classifying sample document using LDA Bag of Words model


```python
processed_docs[4309]
```




    ['rain', 'help', 'dampen', 'bushfir']




```python
for index, score in sorted(lda_model[bow_corpus[4310]],
                           key=lambda tup: -1 * tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score,
                                             lda_model.print_topic(index, 10)))
```


    Score: 0.45221951603889465	 
    Topic: 0.020*"donald" + 0.014*"farmer" + 0.014*"nation" + 0.013*"time" + 0.013*"rural" + 0.013*"council" + 0.012*"indigen" + 0.011*"school" + 0.011*"commiss" + 0.011*"plan"
    
    Score: 0.44774875044822693	 
    Topic: 0.031*"govern" + 0.020*"warn" + 0.018*"feder" + 0.015*"countri" + 0.015*"fund" + 0.014*"claim" + 0.014*"life" + 0.012*"say" + 0.012*"stori" + 0.012*"health"
    
    Score: 0.012506862170994282	 
    Topic: 0.035*"elect" + 0.018*"water" + 0.017*"state" + 0.015*"tasmanian" + 0.012*"labor" + 0.011*"liber" + 0.011*"morrison" + 0.010*"parti" + 0.010*"leader" + 0.010*"campaign"
    
    Score: 0.012503857724368572	 
    Topic: 0.019*"chang" + 0.019*"say" + 0.015*"speak" + 0.015*"power" + 0.013*"worker" + 0.012*"climat" + 0.012*"concern" + 0.011*"flood" + 0.011*"save" + 0.011*"fear"
    
    Score: 0.01250350009649992	 
    Topic: 0.023*"hous" + 0.022*"south" + 0.020*"north" + 0.017*"bushfir" + 0.016*"miss" + 0.013*"interview" + 0.012*"west" + 0.012*"hospit" + 0.011*"coast" + 0.010*"investig"
    
    Score: 0.01250350009649992	 
    Topic: 0.032*"kill" + 0.023*"shoot" + 0.021*"protest" + 0.021*"dead" + 0.020*"attack" + 0.020*"polic" + 0.014*"offic" + 0.014*"assault" + 0.011*"michael" + 0.011*"bank"
    
    Score: 0.01250350009649992	 
    Topic: 0.057*"australia" + 0.046*"australian" + 0.026*"world" + 0.018*"canberra" + 0.017*"test" + 0.013*"win" + 0.011*"final" + 0.011*"farm" + 0.010*"return" + 0.009*"beat"
    
    Score: 0.01250350009649992	 
    Topic: 0.030*"polic" + 0.029*"charg" + 0.026*"court" + 0.024*"death" + 0.024*"murder" + 0.020*"woman" + 0.020*"crash" + 0.017*"face" + 0.016*"alleg" + 0.013*"trial"
    
    Score: 0.01250350009649992	 
    Topic: 0.021*"market" + 0.020*"news" + 0.018*"women" + 0.018*"live" + 0.016*"tasmania" + 0.013*"high" + 0.013*"rise" + 0.012*"price" + 0.012*"lose" + 0.012*"break"
    
    Score: 0.01250350009649992	 
    Topic: 0.044*"trump" + 0.037*"year" + 0.035*"sydney" + 0.028*"queensland" + 0.022*"home" + 0.021*"adelaid" + 0.018*"perth" + 0.016*"brisban" + 0.015*"leav" + 0.015*"peopl"


### Performance evaluation by classifying sample document using LDA TF-IDF model


```python
for index, score in sorted(lda_model_tfidf[bow_corpus[4310]],
                           key=lambda tup: -1 * tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(
        score, lda_model_tfidf.print_topic(index, 10)))
```


    Score: 0.5840891003608704	 
    Topic: 0.012*"stori" + 0.008*"michael" + 0.006*"council" + 0.006*"jam" + 0.006*"centr" + 0.006*"know" + 0.005*"govern" + 0.005*"money" + 0.005*"quiz" + 0.005*"ash"
    
    Score: 0.1756652444601059	 
    Topic: 0.012*"bushfir" + 0.010*"weather" + 0.008*"david" + 0.007*"northern" + 0.006*"queensland" + 0.006*"victoria" + 0.006*"april" + 0.005*"territori" + 0.005*"farm" + 0.005*"wild"
    
    Score: 0.15267711877822876	 
    Topic: 0.019*"charg" + 0.018*"murder" + 0.013*"court" + 0.013*"alleg" + 0.013*"polic" + 0.010*"woman" + 0.010*"jail" + 0.010*"sentenc" + 0.009*"guilti" + 0.009*"accus"
    
    Score: 0.012512458488345146	 
    Topic: 0.024*"trump" + 0.013*"elect" + 0.013*"donald" + 0.010*"govern" + 0.008*"labor" + 0.007*"wednesday" + 0.007*"say" + 0.007*"thursday" + 0.007*"liber" + 0.006*"feder"
    
    Score: 0.012511294335126877	 
    Topic: 0.016*"news" + 0.016*"rural" + 0.010*"market" + 0.010*"price" + 0.008*"busi" + 0.007*"nation" + 0.007*"scott" + 0.006*"share" + 0.006*"sport" + 0.006*"mental"
    
    Score: 0.012509411200881004	 
    Topic: 0.008*"monday" + 0.008*"live" + 0.007*"protest" + 0.006*"kill" + 0.006*"australia" + 0.006*"presid" + 0.006*"turnbul" + 0.006*"novemb" + 0.006*"indonesia" + 0.005*"syria"
    
    Score: 0.012509251944720745	 
    Topic: 0.018*"countri" + 0.013*"hour" + 0.011*"royal" + 0.010*"commiss" + 0.009*"climat" + 0.007*"care" + 0.006*"zealand" + 0.006*"rugbi" + 0.006*"australia" + 0.006*"age"
    
    Score: 0.012509125284850597	 
    Topic: 0.010*"friday" + 0.010*"john" + 0.009*"violenc" + 0.008*"juli" + 0.007*"andrew" + 0.007*"august" + 0.006*"domest" + 0.006*"cancer" + 0.005*"flight" + 0.005*"patient"
    
    Score: 0.012508826330304146	 
    Topic: 0.011*"final" + 0.010*"world" + 0.009*"australia" + 0.007*"leagu" + 0.007*"cricket" + 0.007*"win" + 0.007*"grandstand" + 0.006*"open" + 0.006*"financ" + 0.006*"australian"
    
    Score: 0.01250819955021143	 
    Topic: 0.017*"interview" + 0.016*"crash" + 0.013*"drum" + 0.010*"polic" + 0.009*"street" + 0.008*"tuesday" + 0.008*"coast" + 0.008*"dead" + 0.008*"shoot" + 0.008*"christma"


### Testing model on unseen document


```python
unseen_document = 'How a Pentagon deal became an identity crisis for Google'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
    print("Score: {}\t Topic: {}".format(score,
                                         lda_model.print_topic(index, 5)))
```

    Score: 0.48324039578437805	 Topic: 0.021*"market" + 0.020*"news" + 0.018*"women" + 0.018*"live" + 0.016*"tasmania"
    Score: 0.21708554029464722	 Topic: 0.035*"elect" + 0.018*"water" + 0.017*"state" + 0.015*"tasmanian" + 0.012*"labor"
    Score: 0.18293601274490356	 Topic: 0.030*"polic" + 0.029*"charg" + 0.026*"court" + 0.024*"death" + 0.024*"murder"
    Score: 0.0166776180267334	 Topic: 0.019*"chang" + 0.019*"say" + 0.015*"speak" + 0.015*"power" + 0.013*"worker"
    Score: 0.01667742058634758	 Topic: 0.031*"govern" + 0.020*"warn" + 0.018*"feder" + 0.015*"countri" + 0.015*"fund"
    Score: 0.016676612198352814	 Topic: 0.023*"hous" + 0.022*"south" + 0.020*"north" + 0.017*"bushfir" + 0.016*"miss"
    Score: 0.016676612198352814	 Topic: 0.032*"kill" + 0.023*"shoot" + 0.021*"protest" + 0.021*"dead" + 0.020*"attack"
    Score: 0.016676612198352814	 Topic: 0.057*"australia" + 0.046*"australian" + 0.026*"world" + 0.018*"canberra" + 0.017*"test"
    Score: 0.016676612198352814	 Topic: 0.020*"donald" + 0.014*"farmer" + 0.014*"nation" + 0.013*"time" + 0.013*"rural"
    Score: 0.016676612198352814	 Topic: 0.044*"trump" + 0.037*"year" + 0.035*"sydney" + 0.028*"queensland" + 0.022*"home"

