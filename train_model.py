from underthesea import word_tokenize
import numpy as np

sentiments_dir = 'sentiments.txt'
sents_dir = 'sents.txt'
topics_dir = 'topics.txt'

##read data
with open(sentiments_dir,'r',encoding='utf-8') as f1:
    sentis = [i.strip() for i in f1.readlines()]
with open(topics_dir,'r',encoding='utf-8') as f2:
    topics = [i.strip() for i in f2.readlines()]
with open(sents_dir,'r',encoding='utf-8') as f3:
    sents = [i.strip() for i in f3.readlines()]

##prob per class
##nếu dữ liệu train không cùng tồn tại một topic-sentiment thì khi tính log kết quả trả về -inf
prob_c = {}
for i in sorted(set(topics)):
    prob_c[i] = {}
    for j in sorted(set(sentis)):
        prob_c[i][j] = 0

total = len(sents)

for topic,senti in zip(topics,sentis):
        prob_c[topic][senti] += 1

for i in prob_c:
    for j in prob_c[i]:
        prob_c[i][j] /= total

##making Vocabulary
punctuations=['.','!',',','(',')']

vocab = {}
for sent in sents:
    temp1 = set(word_tokenize(word_tokenize(sent,format='text')))
    for w in temp1:
        if w not in punctuations:
            w = w.lower()
            vocab[w] = 0
#print(len(vocab))

for sent in sents:
    temp2 = word_tokenize(word_tokenize(sent,format='text'))
    for w in temp2:
        if w not in punctuations:
            w = w.lower()
            vocab[w] += 1

##making information per class
info_per_class = {}
for i in sorted(set(topics)):
    info_per_class[i] = {}
    for j in sorted(set(sentis)):
        info_per_class[i][j] = {}
        info_per_class[i][j]['prob'] = {}
        info_per_class[i][j]['words'] = []


for topic,senti,sent in zip(topics,sentis,sents):
    temp = set(word_tokenize(word_tokenize(sent,format='text')))
    for w in temp:
        if w not in punctuations:
            w = w.lower()
            info_per_class[topic][senti]['words'].append(w)

##tính tỉ lệ với laplace smoothing
##smoothing alpha --> làm cho dữ liệu khác 0

for topic in info_per_class.keys():
    for senti in info_per_class[topic].keys():
        for w in vocab.keys():
            info_per_class[topic][senti]['prob'][w]  = (info_per_class[topic][senti]['words'].count(w)
                                                        + 1) / (len(info_per_class[topic][senti]['words'])
                                                        + len(vocab) + 1)


##tính điểm số cho mỗi classy cho tập thử --> sử dụng Multinomial NB dạng log (để có trọng số lớn hơn)
def predict(test_data):
    words = []
    temp = word_tokenize(word_tokenize(test_data,format='text'))
    for w in temp:
        if w not in punctuations:
            w = w.lower()
            words.append(w)
    word_set = set(words)
    score = {}
    for topic in sorted(set(topics)):
        score[topic] = {}
        for senti in sorted(set(sentis)):
            score[topic][senti] = 1
            for w in word_set:
                freq = words.count(w)
                if w not in vocab:
                    score[topic][senti] += 0
                else:
                    score[topic][senti] += np.log(info_per_class[topic][senti]['prob'][w])*freq
            score[topic][senti] += np.log(prob_c[topic][senti])
    max_topic = max_sentiment = '-1'
    max_score = 0
    for i in score:
        for j in score[i]:
            if max_topic == max_sentiment == '-1':
                max_score = score[i][j]
                max_topic = i
                max_sentiment = j
            else:
                if score[i][j] > max_score:
                    max_topic = i
                    max_sentiment = j
    return max_topic,max_sentiment,max_score
