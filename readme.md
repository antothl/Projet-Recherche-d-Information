# Projet Recherche d'Information par Antoine Théologien, Adam Keddis et Gautier Cai

Réimplémentation du papier suivant : \
Chen, Ruey-Cheng and Yulianti, Evi and Sanderson, Mark and Croft, W. Bruce, On the Benefit of Incorporating External Features in a Neural Architecture for Answer Sentence Selection, Proceedings of SIGIR 17\

## Partie implémentation du papier
Instructions pour générer les données ici : https://github.com/rueycheng/QA_ExtFeats/tree/master
Pour les expérimentations, il faut run le script run_experiments contenu dans deep-qa (et réajuster en fonction des paramètres voulus)
Pour récupérer les embeddings : https://git.uwaterloo.ca/jimmylin/Castor-data/-/tree/master/embeddings/word2vec?ref_type=heads
Les fichiers de features se trouvent dans les dossiers WIKIQA et TRECQA. Pour tester avec l'ajout de features, il faut les mettre dans les dossiers TRAIN/TRAIN-ALL du dossier deep-qa, et suivre les instructions du code deepqa (après avoir généré les données) : https://github.com/aseveryn/deep-qa
