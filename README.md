# korean-cnn-text-classifier

함수 설명
=============

### load_data_and_labels(positive_data_file, negative_data_file)   

긍정/부정 파일을 임포트해서 x_text와 y 행렬을 형성.   
Returns x_text and y   

### tokenize(sentence)      

형태소 단위로 분리: “나는 너가 좋아＂ -> “나 는 너 가 좋아”   
Returns tokenized sentence
