import pandas as pd
from scipy.stats import spearmanr, kendalltau
import numpy as np

fichier_excel = 'Eval_utilisateur_final.xlsx'

df = pd.read_excel(fichier_excel)

colonne1 = 'Rang du modèle'
colonne2 = 'Rang Eval Utilisateur'

df = df[[colonne1, colonne2]].dropna()

corr_spearman, _ = spearmanr(df[colonne1], df[colonne2])
corr_kendall, _ = kendalltau(df[colonne1], df[colonne2])

mae = np.mean(np.abs(df[colonne1] - df[colonne2]))

print(f"Corrélation de Spearman : {corr_spearman:.4f}")
print(f"Corrélation de Kendall  : {corr_kendall:.4f}")
print(f"Moyenne d'erreur absolue (MAE) : {mae:.4f}")


accuracy = (df[colonne1] == df[colonne2]).mean()

print(f"Accuracy : {accuracy:.4f}")