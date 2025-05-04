import pandas as pd
from collections import defaultdict
import xml.etree.ElementTree as ET

# Chargement du fichier WikiQA
df = pd.read_csv('WikiQA-dev.tsv', sep='\t')

# Groupement par question
grouped = df.groupby("Question")

# Racine XML
root = ET.Element("data")

for i, (question_text, group) in enumerate(grouped, start=1):
    # Extraire l'ID de la question sans le "Q"
    question_id = group.iloc[0]["QuestionID"].replace("Q", "")  # Enlever le "Q"

    qapair = ET.SubElement(root, "QApairs", id=question_id)

    # Ajouter la balise <question>
    question = ET.SubElement(qapair, "question")
    question.text = question_text  # Ajouter la question

    # Vérifier s'il y a des réponses positives
    positives = group[group["Label"] == 1]
    if not positives.empty:
        # Ajouter toutes les <positive>
        for _, row in positives.iterrows():
            positive = ET.SubElement(qapair, "positive")
            positive.text = row["Sentence"]

        # Ajouter les balises <negative> pour les réponses négatives
        negatives = group[group["Label"] == 0]
        for _, row in negatives.iterrows():
            neg = ET.SubElement(qapair, "negative")
            neg.text = row["Sentence"]
    else:
        # Si aucune réponse positive, ajouter seulement les balises <negative>
        for _, row in group.iterrows():
            neg = ET.SubElement(qapair, "negative")
            neg.text = row["Sentence"]


# Sauvegarde XML
tree = ET.ElementTree(root)
tree.write("wikiqa_dev.xml", encoding="utf-8", xml_declaration=True)
