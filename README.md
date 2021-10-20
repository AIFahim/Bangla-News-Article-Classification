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

### Solution Notebooks: 
- [EDA & Data Cleaning](https://colab.research.google.com/drive/1jIrsJulLduJgHWqBf2XjneG9o3NhEjrS?usp=sharing)
- [Bert Training Code](https://colab.research.google.com/drive/17tQ9DQvFB8sTXfHQfmD_ivf1TTovdIXl?usp=sharing)
- [Glove Single LSTM Training Code](https://colab.research.google.com/drive/1mv24jJccO4RKwel6EsOOsFusM-AV8r_-?usp=sharing)
- [Glove Model Multiple LSTM CNN Training Code](https://colab.research.google.com/drive/146Ddldkd0-iqM1lobLHhwyBJ4xPNTztC?usp=sharing)
- [Evaluation of Bert](https://colab.research.google.com/drive/1nmjIQcgrg09XI8_QxXCG6SGVsF8IoMTv?usp=sharing)
- [Evaluation of Glove Single LSTM](https://colab.research.google.com/drive/1darPESwq1pQ8SL_HwlpDMfqHZjlMc82Q?usp=sharing)
- [Evaluation of Glove Model Multiple LSTM CNN](https://colab.research.google.com/drive/13c3noRsExOFHzv1DrtBxMhoqk6AR4yx9?usp=sharing)

### Trained Model Weight Files
- [Bert Model](https://drive.google.com/file/d/1-5Y29UkrhBKVT1ICM0GFqBk_bTwxjDGD/view?usp=sharing)
- [Glove Single LSTM](https://drive.google.com/file/d/1-5Y29UkrhBKVT1ICM0GFqBk_bTwxjDGD/view?usp=sharing)
- [Glove Model Multiple LSTM CNN](https://drive.google.com/file/d/1Eglkzg5na5i6C7Derq1tz-lEcuebNLPe/view?usp=sharing)

### Resources Used
- **Developement Envioronment :** Google Colab
- **Python Version :** 3.7
- **Framework and Packages :** Tensorflow 2.1.0 ,keras, Scikit-Learn, Pandas, Numpy, Matplotlib, Seaborn, Ploty
- **Runtime Type:** Colab TPU, Colab GPU

### Dataset: 
- A publicly available dataset [BanglaMCT7: Bangla Multi-Class Text Dataset 7 Tags](https://www.kaggle.com/gakowsher/banglamct7-bangla-multiclass-text-dataset-7-tags) which contains news articles of 7 different classes.      

### Class Distribution of Dataset :
<img src="https://drive.google.com/uc?export=view&id=1YGqUJQOd_4t1V0DYiYSkqL5-5yGgrnWb" alt="load again" width="530" height="350"/>

### EDA of Dataset
#### Number of words in each classes : 

<img src="https://drive.google.com/uc?export=view&id=1sTrp48nsphSGo3ujjFJexeQJHSwA_sJX" alt="load again" width="700" height="350"/>  |  <img src="https://drive.google.com/uc?export=view&id=1YVP5v0XyqU85LkSgqR_rZNKBRql9JIdb" alt="load again" width="700" height="350"/> |
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1eSZv2MhBVHPQkGiseEgOPiqHy2JDHwlW" alt="load again" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=1x2k5vMLv63q0Wbos7eFGsw-xyiZF4N5X" alt="load again" width="700" height="350"/> | 


<img src="https://drive.google.com/uc?export=view&id=11RP1uhd_2EHIDAoGg_WiREQIGRfl44YA" alt="load again" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=1i-i5-1H_yiP2wZuQV3okcESsQT5ij_aY" alt="load again" width="700" height="350"/> | 
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1VvKm_jwovJdzhBJnsHDntSHJ6E6N0JMp" alt="load again" width="700" height="350"/> | | 


#### Number of character in each classes : 

<img src="https://drive.google.com/uc?export=view&id=1UWaHTXbb9ZZ8cqSY-ZJ9HnFOySjDgZg2" alt="load again" width="700" height="350"/>  |  <img src="https://drive.google.com/uc?export=view&id=1jW7qYq3Z1qsV-wtmOg6luE406zvJZ0gF" alt="load again" width="700" height="350"/> |
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1d6W7iTYwOF5oTjfWEVkqKGVk8KoQ5NS4" alt="load again" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=15moPm50C-EyQhiaOsHRwr-lReMW8m61s" alt="load again" width="700" height="350"/> | 


<img src="https://drive.google.com/uc?export=view&id=1XTzv5oPpVgHGOkrFUgTvYfww8g_sIhhZ" alt="load again" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=13nbh_gI7xjaPby37JibIkZLSZ0sNDtew" alt="load again" width="700" height="350"/> | 
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1ymFUMttwyo1L_UUDDDf8lo3LUsw6wk22" alt="load again" width="700" height="350"/> | | 



#### Average word length in text in each classes : 

<img src="https://drive.google.com/uc?export=view&id=1U802FZoTb4Ak1UgiNIXflFEfkqbFlgqE" alt="load again" width="700" height="350"/>  |  <img src="https://drive.google.com/uc?export=view&id=1e2j_pEfUPbioyKeE56Ml9t3bRuHD0l36" alt="load again" width="700" height="350"/> |
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1u5z0S1LzeC4KOEEX2SSK8BLvdkrfdIwY" alt="load again" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=1T7GHoURdz9tt5ODBjw0KOrbRHYNB19Nv" alt="load again" width="700" height="350"/> | 


<img src="https://drive.google.com/uc?export=view&id=1OXXVKTHddxmhTTkQS3YDZ5uRXav5YPs8" alt="load again" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=1n2vhcZeYVqwNIW6CgNEpVB8u-T3zYW0M" alt="load again" width="700" height="350"/> | 
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=11ufrSKxgL5NKhWQfUS-m-7eviBUGSp_5" alt="load again" width="700" height="350"/> | | 




#### Analyze punction : 
<img src="https://drive.google.com/uc?export=view&id=120GJnUV6BkELzYBeU62Jgg9ecL0XqM2z" alt="load again" width="700" height="350"/>

**Found punction in text column but didn't found any punction in cleanText column**


#### Common Words in each classes: 
<img src="https://drive.google.com/uc?export=view&id=1FlwiUzcIIqUbypJ2ROCu_bWIiCLXlTZG" alt="load again" width="700" height="350"/>  |  <img src="https://drive.google.com/uc?export=view&id=1lImrxqfmLLVOmbzfzBjpPnAxTyUMWZWR" alt="load again" width="700" height="350"/> |
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1cu8ft1Z-FhQqb7NhRKeGkXjAIrggCfXW" alt="load again" width="700" height="350"/>  |  <img src="https://drive.google.com/uc?export=view&id=1c_QcXoW2KvME1TEXYubaeQ_KNGXlUIIh" alt="load again" width="700" height="350"/> |


<img src="https://drive.google.com/uc?export=view&id=1kDCnN9kNgAAMVUk6wSHQwUU-nHt4VQNx" alt="load again" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=1eYUQhHLpa-qEQPidtwgBFteIOA7j96gI" alt="load again" width="700" height="350"/> | 
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1i7HLEGqTYFI53eAOn6ykjgdpkHH1qcKW" alt="load again" width="700" height="350"/> | | 





#### Common stop word in sport category (Based on text & clenText) : 

<img src="https://drive.google.com/uc?export=view&id=1EFFj020iRKhh8maLPMV76DURnBnVfKFE" alt="load again" width="700" height="350"/>  |  <img src="https://drive.google.com/uc?export=view&id=1_Guj651hAhXkCpU9i1mhdnjf10mcgAKz" alt="load again" width="700" height="350"/> |
:-------------------------:|:-------------------------:

** By observing Found that cleanText column of Sports category much less stop words contains compare with text column. So check others category for punctions on cleanText feature. **

<img src="https://drive.google.com/uc?export=view&id=1WUPnufPCLKTMx3aGQnoG-3lJxo8_4czH" alt="load again" width="700" height="350"/>  |  <img src="https://drive.google.com/uc?export=view&id=14PcREQ-489PDbz7CROFGXc82dxaPb8Gw" alt="load again" width="700" height="350"/> |
:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1bUdoaRLz-loB5vuKrWq_CtPYeMFb1mHW" alt="load again" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=1wL4CQ8etW7BIkHxIYQT2Bfd5EJHAVvT6" alt="load again" width="700" height="350"/> | 


<img src="https://drive.google.com/uc?export=view&id=186pA_mhY7z3N0JjHCD4I0MCI6TvpKFh_" alt="load again" width="700" height="350"/> |   <img src="https://drive.google.com/uc?export=view&id=10qA28xX_Ea9XvPZdJypVYmiFzLHTwjyg" alt="load again" width="700" height="350"/> | 
:-------------------------:|:-------------------------:

#### Bigram Analysis: 
<img src="https://drive.google.com/uc?export=view&id=1StcYQYUbJ60MJSGrVqQ239vmQ1Ho9q9S" alt="load again" width="700" height="350"/> 


### Cleaning of Dataset
As analysis found some stop words in cleanText columns. Removed those stop words using `from bnlp.corpus.util import remove_stopwords`.

### Model Development
##### Model 1:
- Bert Model (Pretained : bert_en_uncased_L-24_H-1024_A-16)
- Tokennizer Bangla Bert Tokenizer `bnbert_tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")`

##### Model 1 Summary: 
<img src="https://drive.google.com/uc?export=view&id=1aKtMooWqJh80aWaAGYIeLGJVjexwLEoq" alt="load again" width="700" height="350"/> 
<hr>

##### Model 2:
- One LSTM Layer with Embedding Layer 
- Use Glove pretrained corpus model to represent words

##### Model 2 Summary: 
<img src="https://drive.google.com/uc?export=view&id=1tDswvoc9S-Ud4A5l3Rzmr4NMmelhxXf3" alt="load again" width="700" height="350"/> 

##### Model 3:
- Multi LSTM Layer & 1D CNN with Embedding Layer 
- Use Glove pretrained corpus model to represent words

##### Model 3 Summary: 
<img src="https://drive.google.com/uc?export=view&id=1H0vFeBzaDehO5jmQYjzUXvuiyuG_gpF-" alt="load again" width="700" height="350"/> 

### Model Evaluation
#### Bert
<img src="https://drive.google.com/uc?export=view&id=1OEuhSCYBq5QbbGKNgQcQdFd_3eh-NEPN" alt="load again" width="700" height="350"/> | <img src="https://drive.google.com/uc?export=view&id=YqjXHwdSDBGjXcQqx28SOH" alt="load again" width="700" height="350"/> 
:-------------------------:|:-------------------------:

#### Glove Single LSTM
<img src="https://drive.google.com/uc?export=view&id=1laykz7ZLvPwB5vy3wt-g16jmcHNPBeWQ" alt="load again" width="700" height="350"/> | <img src="https://drive.google.com/uc?export=view&id=17oSgrPnOTttdxlVZY1sVD5Rr9jUMQSbg" alt="load again" width="700" height="350"/> 
:-------------------------:|:-------------------------:

#### Glove Multiple LSTM CNN
<img src="https://drive.google.com/uc?export=view&id=1LcRIBPTTDTnk91im4VakCRnuLm_zIuJm" alt="load again" width="700" height="350"/> | <img src="https://drive.google.com/uc?export=view&id=1NBriTudS_aTi6upUp9k9xEOCvAJoKFar" alt="load again" width="700" height="350"/> 
:-------------------------:|:-------------------------:
