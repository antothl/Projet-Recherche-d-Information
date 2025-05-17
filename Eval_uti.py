import pandas as pd
from scipy.stats import spearmanr, kendalltau
import numpy as np

# Remplace par le chemin réel de ton fichier
fichier_excel = 'Eval_utilisateur_final.xlsx'

# Chargement du fichier Excel
df = pd.read_excel(fichier_excel)

# Remplace ces noms par ceux de tes colonnes
colonne1 = 'Rang du modèle'
colonne2 = 'Rang Eval Utilisateur'

# Suppression des lignes avec valeurs manquantes
df = df[[colonne1, colonne2]].dropna()

# Corrélations
corr_spearman, _ = spearmanr(df[colonne1], df[colonne2])
corr_kendall, _ = kendalltau(df[colonne1], df[colonne2])

# Moyenne d'erreur absolue
mae = np.mean(np.abs(df[colonne1] - df[colonne2]))

# Affichage des résultats
print(f"Corrélation de Spearman : {corr_spearman:.4f}")
print(f"Corrélation de Kendall  : {corr_kendall:.4f}")
print(f"Moyenne d'erreur absolue (MAE) : {mae:.4f}")


accuracy = (df[colonne1] == df[colonne2]).mean()

print(f"Accuracy : {accuracy:.4f}")