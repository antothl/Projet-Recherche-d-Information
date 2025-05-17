import re
import tqdm

def index_document_phrase(D, TRAIN, TEST, DEV):
    """
    """
    DP = {d: set() for d in D}

    for data in TRAIN.itertuples():
        DP[data.DocumentID].add(data.SentenceID)

    for data in TEST.itertuples():
        DP[data.DocumentID].add(data.SentenceID)

    for data in DEV.itertuples():
        DP[data.DocumentID].add(data.SentenceID)   

    return DP


def indice(D, DP):
    """
    """
    indice_doc = {i: d for i, d in enumerate(D)}
    doc_indice = {d: i for i, d in indice_doc.items()}

    indice_phrase = {}
    phrase_indice = {}

    c = 0

    for i in range(len(indice_doc)):
        for phrase in DP[indice_doc[i]]:
            indice_phrase[c] = phrase
            c += 1

    phrase_indice = {p: i for i, p in indice_phrase.items()}

    phrase_par_doc_indice = {}

    for doc in D:
        c = 0
        for phrase in DP[doc]:
            phrase_par_doc_indice[phrase] = c
            c += 1

    doc_indice_phrase_par_doc = {}

    for doc in D:
        doc_indice_phrase_par_doc[doc] = {}
        for phrase in DP[doc]:
            doc_indice_phrase_par_doc[doc][phrase_par_doc_indice[phrase]] = phrase

    return indice_doc, doc_indice, indice_phrase, phrase_indice, phrase_par_doc_indice, doc_indice_phrase_par_doc
