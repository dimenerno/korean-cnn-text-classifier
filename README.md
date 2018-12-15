# korean-cnn-text-classifier

Structure
=============
![CNN Text Classification](http://d3kbpzbmcynnmx.cloudfront.net/wp-content/uploads/2015/11/Screen-Shot-2015-11-06-at-12.05.40-PM.png)

data_helpers.py
=============

### load_data_and_labels(positive_data_file, negative_data_file)   

긍정/부정 파일을 임포트해서 x_text와 y 행렬을 형성.   
Returns x_text and y   

### tokenize(sentence)      

형태소 단위로 분리: “나는 너가 좋아＂ -> “나 는 너 가 좋아”   
Returns tokenized sentence


train.py
=============

### preprocess()

load_data_and_labels에서 데이터 불러옴.   
vocab_processor로 단어사전을 구축하고 각 문장의 단어 번호 매김   
*fit_transform으로 문장들을 벡터화함*   
np.array로 벡터화된 문장을 행렬로 표현->shuffle   

dev_sample_percentage * len(y) 기준으로 셔플된 배열을 train 과 dev로 분리

알고 있으면 쓸모있는 것들
===============
- Linux에서 한글 에러날 때: 콘솔에 `export PYTHONIOENCODING=utf8` 입력
