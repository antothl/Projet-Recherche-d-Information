import my_scripts
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
import numpy as np
import scipy
import matplotlib.pyplot as plt
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import architecture
import random

SEED = 12
TRAIN = pd.read_csv("data/post_train.tsv", sep="\t")
TEST = pd.read_csv("data/post_test.tsv", sep="\t")
DEV = pd.read_csv("data/post_dev.tsv", sep="\t")

D = {data.DocumentID for data in TRAIN.itertuples()}.union({data.DocumentID for data in TEST.itertuples()}).union({data.DocumentID for data in DEV.itertuples()})
DP = my_scripts.index_document_phrase(D, TRAIN, TEST, DEV)
indice_doc, doc_indice, indice_phrase, phrase_indice, phrase_par_doc_indice, doc_indice_phrase_par_doc = my_scripts.indice(D, DP)

indice_question = {i: q for i,q in enumerate(TRAIN["QuestionID"].unique().tolist() + TEST["QuestionID"].unique().tolist() + DEV["QuestionID"].unique().tolist())}
question_indice = {q: i for i,q in indice_question.items()}

Xphrase_train, Xphrase_test, Xphrase_dev = [], [], []
Xquestion_train, Xquestion_test, Xquestion_dev = [], [], []

indice_question = {}

i = 0

for exemple in TRAIN.itertuples():
    if exemple.Sentence not in Xphrase_train:
        Xphrase_train.append(exemple.Sentence)
    if exemple.QuestionID not in indice_question.values():
        Xquestion_train.append(exemple.Question)
        indice_question[i] = exemple.QuestionID
        i += 1

for exemple in TEST.itertuples():
    if exemple.Sentence not in Xphrase_test:
        Xphrase_test.append(exemple.Sentence)
    if exemple.QuestionID not in indice_question.values():
        Xquestion_test.append(exemple.Question)
        indice_question[i] = exemple.QuestionID
        i += 1

for exemple in DEV.itertuples():
    if exemple.Sentence not in Xphrase_dev:
        Xphrase_dev.append(exemple.Sentence)
    if exemple.QuestionID not in indice_question.values():
        Xquestion_dev.append(exemple.Question)
        indice_question[i] = exemple.QuestionID
        i += 1
        
question_indice = {q: i for i,q in indice_question.items()}

Xphrase_train = ["" if pd.isna(x) else str(x) for x in Xphrase_train]
Xquestion_train = ["" if pd.isna(x) else str(x) for x in Xquestion_train]
Xphrase_test = ["" if pd.isna(x) else str(x) for x in Xphrase_test]
Xquestion_test = ["" if pd.isna(x) else str(x) for x in Xquestion_test]
Xphrase_dev = ["" if pd.isna(x) else str(x) for x in Xphrase_dev]
Xquestion_dev = ["" if pd.isna(x) else str(x) for x in Xquestion_dev]

Xphrase = Xphrase_train + Xphrase_test + Xphrase_dev
Xquestion = Xquestion_train + Xquestion_test + Xquestion_dev

vectorizer_tfidf = TfidfVectorizer(analyzer="char", ngram_range=(5, 7))

vectorizer_tfidf.fit(Xphrase)

Xphrase = vectorizer_tfidf.transform(Xphrase)
Xquestion = vectorizer_tfidf.transform(Xquestion)

DIM = 10000

lsa = TruncatedSVD(n_components=DIM)
lsa.fit(Xphrase)

Xphrase_lsa = lsa.transform(Xphrase)
Xquestion_lsa = lsa.transform(Xquestion)

Xdoc = np.array([np.mean([Xphrase_lsa[phrase_indice[p]] for p in doc_indice_phrase_par_doc[doc].values()], axis=0) for doc in D])

unique_pairs_train = TRAIN.drop_duplicates(subset=["QuestionID", "DocumentID"])
unique_pairs_test = TEST.drop_duplicates(subset=["QuestionID", "DocumentID"])

Xss_train = []
Yss_train = []

nb_exemples = 100

for exemple in tqdm.tqdm(unique_pairs_train.itertuples()):
    Qi, Di = question_indice[exemple.QuestionID], doc_indice[exemple.DocumentID]
    q, d = Xquestion_lsa[Qi], Xdoc[Di]

    indices = list(range(len(Xdoc)))
    indices.remove(Di)
    exemples_aleatoires = random.sample(indices, min(nb_exemples, len(indices)))

    for j in exemples_aleatoires:
        Xss_train.append(np.concatenate([q, d, q, Xdoc[j]]))
        Xss_train.append(np.concatenate([q, Xdoc[j], q, d]))
        Yss_train.append(1)
        Yss_train.append(0)

Xss_train = torch.tensor(Xss_train, dtype=torch.float32)
Yss_train = torch.tensor(Yss_train, dtype=torch.float32)

Xss_test = []
Yss_test = []

for exemple in tqdm.tqdm(unique_pairs_test.itertuples()):
    Qi, Di = question_indice[exemple.QuestionID], doc_indice[exemple.DocumentID]
    q, d = Xquestion_lsa[Qi], Xdoc[Di]

    indices = list(range(len(Xdoc)))
    indices.remove(Di)
    exemples_aleatoires = random.sample(indices, min(nb_exemples, len(indices)))

    for j in exemples_aleatoires:
        Xss_test.append(np.concatenate([q, d, q, Xdoc[j]]))
        Xss_test.append(np.concatenate([q, Xdoc[j], q, d]))
        Yss_test.append(1)
        Yss_test.append(0)

Xss_test = torch.tensor(Xss_test, dtype=torch.float32)
Yss_test = torch.tensor(Yss_test, dtype=torch.float32)

batch_size = 1024

dataset_train = TensorDataset(Xss_train, Yss_train)
dataset_test = TensorDataset(Xss_test, Yss_test)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False)

NN = nn.Sequential(
    architecture.shared_Linear(d_in=DIM*2, d_out=DIM*4, branches=2),
    nn.Tanh(),
    nn.Dropout(p=0.3),
    architecture.shared_Linear(d_in=DIM*4, d_out=DIM*8, branches=2),
    nn.Tanh(),
    nn.Dropout(p=0.3),
    architecture.shared_Linear(d_in=DIM*8, d_out=DIM, branches=2),
    nn.Tanh(),
    nn.Dropout(p=0.3),
    nn.Linear(DIM*2, DIM),
    nn.Tanh(),
    nn.Dropout(p=0.3),
    nn.Linear(DIM, 1),
    nn.Sigmoid()
)


criterion = nn.BCELoss()
optimizer = optim.Adam(NN.parameters(), lr=0.01)

n_epochs = 20

for epoch in tqdm.tqdm(range(n_epochs)):
    NN.train()
    total_loss = 0
    for xb, yb in dataloader_train:

        optimizer.zero_grad()
        y_pred = NN(xb)
        loss = criterion(y_pred.squeeze(-1), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(dataloader_train.dataset)

    NN.eval()
    with torch.no_grad():
        test_loss = 0
        for xb, yb in dataloader_test:
            y_pred = NN(xb)
            loss = criterion(y_pred.squeeze(-1), yb)
            test_loss += loss.item() * xb.size(0)

        avg_test_loss = test_loss / len(dataloader_test.dataset)

torch.save(NN.state_dict(), "document_infer.pth")

Y_true = []
Y_pred = []

NN.eval()
with torch.no_grad():
    for xb, yb in dataloader_test:
        y_pred = NN(xb)
        pred_labels = (y_pred.squeeze(-1) > 0.5).float()

        Y_true.append(yb)
        Y_pred.append(pred_labels)

Y_true = torch.cat(Y_true).numpy()
Y_pred = torch.cat(Y_pred).numpy()

with open("report_pairwise.txt", "w") as f:
    f.write(classification_report(Y_true, Y_pred))
