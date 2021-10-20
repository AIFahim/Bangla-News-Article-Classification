## Bangla News Article Classification

### Problem Defination:
**Classify Bengali news articles into 7 diffferent classes**:     
- economy
- education
- entertainment
- international
- sports
- state
- technology

### Resources Used
- **Developement Envioronment :** Google Colab
- **Python Version :** 3.7
- **Framework and Packages :** Tensorflow 2.1.0 ,keras, Scikit-Learn, Pandas, Numpy, Matplotlib, Seaborn, Ploty
- **Runtim Type:** Colab TPU, Colab GPU

### Dataset: 
- A publicly available dataset [BanglaMCT7: Bangla Multi-Class Text Dataset 7 Tags](https://www.kaggle.com/gakowsher/banglamct7-bangla-multiclass-text-dataset-7-tags) which contains news articles of 7 different classes.      

### Class Distribution of Dataset :
<img src="https://drive.google.com/uc?export=view&id=1YGqUJQOd_4t1V0DYiYSkqL5-5yGgrnWb" width="530" height="350"/>

### EDA of Dataset
#### Number of words in each classes : 

<img src="https://drive.google.com/uc?export=view&id=1sTrp48nsphSGo3ujjFJexeQJHSwA_sJX" width="700" height="350"/>  |  <img src="https://drive.google.com/uc?export=view&id=1YVP5v0XyqU85LkSgqR_rZNKBRql9JIdb" width="700" height="350"/> |
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1eSZv2MhBVHPQkGiseEgOPiqHy2JDHwlW" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=1x2k5vMLv63q0Wbos7eFGsw-xyiZF4N5X" width="700" height="350"/> | 


<img src="https://drive.google.com/uc?export=view&id=11RP1uhd_2EHIDAoGg_WiREQIGRfl44YA" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=1i-i5-1H_yiP2wZuQV3okcESsQT5ij_aY" width="700" height="350"/> | 
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1VvKm_jwovJdzhBJnsHDntSHJ6E6N0JMp" width="700" height="350"/> | | 


#### Number of character in each classes : 

<img src="https://drive.google.com/uc?export=view&id=1UWaHTXbb9ZZ8cqSY-ZJ9HnFOySjDgZg2" width="700" height="350"/>  |  <img src="https://drive.google.com/uc?export=view&id=1jW7qYq3Z1qsV-wtmOg6luE406zvJZ0gF" width="700" height="350"/> |
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1d6W7iTYwOF5oTjfWEVkqKGVk8KoQ5NS4" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=15moPm50C-EyQhiaOsHRwr-lReMW8m61s" width="700" height="350"/> | 


<img src="https://drive.google.com/uc?export=view&id=1XTzv5oPpVgHGOkrFUgTvYfww8g_sIhhZ" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=13nbh_gI7xjaPby37JibIkZLSZ0sNDtew" width="700" height="350"/> | 
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1ymFUMttwyo1L_UUDDDf8lo3LUsw6wk22" width="700" height="350"/> | | 



#### Average word length in text in each classes : 

<img src="https://drive.google.com/uc?export=view&id=1U802FZoTb4Ak1UgiNIXflFEfkqbFlgqE" width="700" height="350"/>  |  <img src="https://drive.google.com/uc?export=view&id=1e2j_pEfUPbioyKeE56Ml9t3bRuHD0l36" width="700" height="350"/> |
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1u5z0S1LzeC4KOEEX2SSK8BLvdkrfdIwY" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=1T7GHoURdz9tt5ODBjw0KOrbRHYNB19Nv" width="700" height="350"/> | 


<img src="https://drive.google.com/uc?export=view&id=1OXXVKTHddxmhTTkQS3YDZ5uRXav5YPs8" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=1n2vhcZeYVqwNIW6CgNEpVB8u-T3zYW0M" width="700" height="350"/> | 
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=11ufrSKxgL5NKhWQfUS-m-7eviBUGSp_5" width="700" height="350"/> | | 




#### Analyze punction : 
<img src="https://drive.google.com/uc?export=view&id=120GJnUV6BkELzYBeU62Jgg9ecL0XqM2z" width="700" height="350"/>

**Found punction in text column but didn't found any punction in cleanText column**


#### Common Words in each classes: 
<img src="https://drive.google.com/uc?export=view&id=1FlwiUzcIIqUbypJ2ROCu_bWIiCLXlTZG" width="700" height="350"/>  |  <img src="https://drive.google.com/uc?export=view&id=1lImrxqfmLLVOmbzfzBjpPnAxTyUMWZWR" width="700" height="350"/> |
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1cu8ft1Z-FhQqb7NhRKeGkXjAIrggCfXW" width="700" height="350"/>  |  <img src="https://drive.google.com/uc?export=view&id=1c_QcXoW2KvME1TEXYubaeQ_KNGXlUIIh" width="700" height="350"/> |


<img src="https://drive.google.com/uc?export=view&id=1kDCnN9kNgAAMVUk6wSHQwUU-nHt4VQNx" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=1eYUQhHLpa-qEQPidtwgBFteIOA7j96gI" width="700" height="350"/> | 
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1i7HLEGqTYFI53eAOn6ykjgdpkHH1qcKW" width="700" height="350"/> | | 





#### Common stop word in sport category (Based on text & clenText) : 

<img src="https://drive.google.com/uc?export=view&id=1EFFj020iRKhh8maLPMV76DURnBnVfKFE" width="700" height="350"/>  |  <img src="https://drive.google.com/uc?export=view&id=1_Guj651hAhXkCpU9i1mhdnjf10mcgAKz" width="700" height="350"/> |
:-------------------------:|:-------------------------:

** By observing Found that cleanText column of Sports category much less stop words contains compare with text column. So check others category for punctions on cleanText feature. **

<img src="https://drive.google.com/uc?export=view&id=1WUPnufPCLKTMx3aGQnoG-3lJxo8_4czH" width="700" height="350"/>  |  <img src="https://drive.google.com/uc?export=view&id=14PcREQ-489PDbz7CROFGXc82dxaPb8Gw" width="700" height="350"/> |
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1bUdoaRLz-loB5vuKrWq_CtPYeMFb1mHW" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=1wL4CQ8etW7BIkHxIYQT2Bfd5EJHAVvT6" width="700" height="350"/> | 


<img src="https://drive.google.com/uc?export=view&id=186pA_mhY7z3N0JjHCD4I0MCI6TvpKFh_" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=10qA28xX_Ea9XvPZdJypVYmiFzLHTwjyg" width="700" height="350"/> | 
:-------------------------:|:-------------------------:

#### Bigram Analysis: 
<img src="https://drive.google.com/uc?export=view&id=1StcYQYUbJ60MJSGrVqQ239vmQ1Ho9q9S" width="700" height="350"/> 

**EDA 


### Cleaning of Dataset
As analysis found some stop words in cleanText columns. Removed those stop words using `from bnlp.corpus.util import remove_stopwords`.

### Model Development
- Bert Model (Pretained : bert_en_uncased_L-24_H-1024_A-16)
- Tokennizer Bangla Bert Tokenizer `bnbert_tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")`

##### Model Summary: 
<img src="https://drive.google.com/uc?export=view&id=1aKtMooWqJh80aWaAGYIeLGJVjexwLEoq" width="700" height="350"/> 
