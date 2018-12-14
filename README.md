# korean-cnn-text-classifier

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

dev_sample_percentage * len(y) 기준으로 셔플된 배열을 train 과 dev로 
