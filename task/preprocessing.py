import os
import pandas as pd

    all_data = []
    Articles = []
    Summaries = []

    for d, path,filenames in (os.walk('/content/bbc news summary/BBC News Summary')):
        for file in filenames:
            if os.path.isfile(d+'/'+file):
                if('Summaries' in d+'/'+file):
                    with open(d+'/'+file,'r',errors='ignore') as f:
                        summary=''.join([i.rstrip() for i in f.readlines()])
                        Summaries.append(summary)
                        f.close()
                else:
                    with open(d+'/'+file,'r',errors='ignore') as f:
                        Article=''.join([i.rstrip() for i in f.readlines()])
                        Articles.append(' '.join([w for w in Article.split()]))
                        f.close()


    # train_data = pd.read_csv('./cnn_dailymail/train.csv').drop('id', axis=1)
    # train_data['label'] = 1
    # train_input = train_data['article']
    # train_summary = train_data['highlights']

    # validation_data = pd.read_csv('./cnn_dailymail/validation.csv').drop('id', axis=1)
    # validation_data['label'] = 1
    # validation_input = validation_data['article']
    # validation_summary = validation_data['highlights']
    # validation_label = validation_data['label']

    data_ = pd.DataFrame({'article':Articles, 'highlights':Summaries})
    data_['label'] = 0

    _data = pd.read_csv('./cnn_dailymail/test.csv').drop('id', axis=1)
    _data['label'] = 1

    # csv 파일로 저장
    train_data = pd.concat([data_[:int(data_.shape[0]*0.6)], _data[:int(_data.shape[0]*0.6)]], ignore_index=True)
    valid_data = pd.concat([data_[int(data_.shape[0]*0.6):], _data[int(_data.shape[0]*0.6):]], ignore_index=True)
    test_data = valid_data[int(valid_data.shape[0]*0.5):]
    valid_data = valid_data[:int(valid_data.shape[0]*0.5)]

    train_data.to_csv('data/train_data.csv', sep='\t', index=False)
    valid_data.to_csv('data/valid_data.csv', sep='\t', index=False)
    test_data.to_csv('data/test_data.csv', sep='\t', index=False)