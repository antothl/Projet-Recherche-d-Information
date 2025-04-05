import pandas as pd

def count_unique_questions(file_path):
    df = pd.read_csv(file_path, sep='\t')
    unique_questions = df['Question'].nunique()
    print(f"{file_path} : {unique_questions} questions uniques")

count_unique_questions('datasets/TRECQA/train.tsv')
count_unique_questions('datasets/TRECQA/dev.tsv')
count_unique_questions('datasets/TRECQA/test.tsv')


