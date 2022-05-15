import os
import pandas as pd

def shift_challenge_processing(args):

    # WikiMatrix (Processed version)
    with open(os.path.join(args.data_path, 'WikiMatrix.en-ru.txt.en'), 'r') as f:
        wiki_matrix_en = [x.replace('\n', '') for x in f.readlines()]

    with open(os.path.join(args.data_path, 'WikiMatrix.en-ru.txt.ru'), 'r') as f:
        wiki_matrix_ru = [x.replace('\n', '') for x in f.readlines()]

    assert len(wiki_matrix_en) == len(wiki_matrix_ru)

    # Back-translated News (Need to pre-processing)
    with open('/HDD/dataset/shift_challenge/news.en', 'r') as f:
        news_en = [x.replace('\n', '') for x in f.readlines()]
        
    with open('/HDD/dataset/shift_challenge/news.ru', 'r') as f:
        news_ru = [x.replace('\n', '') for x in f.readlines()]
        
    with open('/HDD/dataset/shift_challenge/news.en.translatedto.ru', 'r') as f:
        news_en_to_ru = [x.replace('\n', '') for x in f.readlines()]
        
    with open('/HDD/dataset/shift_challenge/news.ru.translatedto.en', 'r') as f:
        news_ru_to_en = [x.replace('\n', '') for x in f.readlines()]

    # News Commentary (v15)
    news_commentary = pd.read_csv(os.path.join(args.data_path, 'news-commentary-v15.en-ru.tsv'), 
                                  sep='\t', names=['en', 'ru'])
    news_commentary = news_commentary.dropna()

    news_commentary_en = news_commentary['en']
    news_commentary_ru = news_commentary['ru']

    assert len(news_commentary_en) == len(news_commentary_ru)

    # 