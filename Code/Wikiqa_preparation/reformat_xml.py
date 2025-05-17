import xml.etree.ElementTree as ET

def reformat_text(text):
    # Séparer le texte en mots et les reformater
    words = text.split()
    return "	".join(words)

# Charger le fichier XML
tree = ET.parse('wikiqa_dev.xml')
root = tree.getroot()

# Reformatage de chaque QApairs
with open('dev.xml', 'w', encoding='utf-8') as f:
    for qapair in root.findall('QApairs'):
        # Récupérer l'id de QApairs
        qapair_id = qapair.get('id')

        # Écrire la balise QApairs avec l'attribut id
        f.write(f"<QApairs id='{qapair_id}'>\n")

        # Reformater et écrire la question
        question = qapair.find('question').text.strip()
        reformatted_question = reformat_text(question)
        f.write(f"<question>\n{reformatted_question}\n</question>\n")

        
        # Reformater et écrire les réponses positives et negatives

        positives = qapair.findall('positive')
        for pos in positives:
            reformatted_positive = reformat_text(pos.text.strip())
            f.write(f"<positive>\n{reformatted_positive}\n</positive>\n")

        negatives = qapair.findall('negative')
        for neg in negatives:
            reformatted_negative = reformat_text(neg.text.strip())
            f.write(f"<negative>\n{reformatted_negative}\n</negative>\n")

        # Fermer la balise QApairs
        f.write(f"</QApairs>\n")
