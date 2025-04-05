import pandas as pd

def filter_questions_with_positive_labels(path_in, path_out):
    df = pd.read_csv(path_in, sep='\t')
    q_has_positive = df[df['Label'] == 1]['Question'].unique()
    df_filtered = df[df['Question'].isin(q_has_positive)]
    df_filtered.to_csv(path_out, sep='\t', index=False)

filter_questions_with_positive_labels('WikiQACorpus/WikiQA-train.tsv', 'WikiQA-train-filtered.tsv')
filter_questions_with_positive_labels('WikiQACorpus/WikiQA-dev.tsv', 'WikiQA-dev-filtered.tsv')
filter_questions_with_positive_labels('WikiQACorpus/WikiQA-test.tsv', 'WikiQA-test-filtered.tsv')