import time
import argparse

start = time.time()

import train_model as model

parser = argparse.ArgumentParser()
parser.add_argument('--input', help = "put the string input here", type = str)
parser.add_argument('--result', help = "put the output txt file here", type = str)
args = parser.parse_args()

topic,sentiment,score = model.predict(args.input)
if args.result:
    with open(args.result,'w') as fout:
        fout.write(sentiment)
        print('Xuất file thành công!')

print('Topic: ' + str(topic) + '; Sentiment: ' + str(sentiment) + '\nScore (by log): ' + str(score))

print('Execute time: ' + str(round(time.time()-start,2)) + ' sec')