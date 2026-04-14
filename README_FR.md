===========================================================
# PROJET : Classification d'Images Intel (CNN)
# AUTEUR : ADAMA TELLY BA
# FRAMEWORKS : PyTorch & TensorFlow
===========================================================

# 1. DESCRIPTION DU PROJET
------------------------
Ce projet implémente un pipeline complet de classification d'images 
pour le dataset Intel (6 classes : buildings, forest, glacier, 
mountain, sea, street). Il comprend :
- Deux architectures CNN personnalisées (PyTorch et TensorFlow).
- Un script d'entraînement avec arguments de commande.
- Une interface Web Flask pour tester les modèles.

# 2. DÉPENDANCES (BIBLIOTHÈQUES REQUISES)
---------------------------------------
Pour faire fonctionner ce projet, vous devez installer les 
bibliothèques suivantes (Python 3.11 recommandé) :

- tensorflow (2.x) : Pour le modèle Keras et l'entraînement TF.
- torch & torchvision : Pour le modèle et les transformations PyTorch.
- flask : Pour le serveur Web et l'interface utilisateur.
- pillow : Pour la gestion et le redimensionnement des images.
- numpy : Pour les calculs mathématiques sur les images.
- argparse : Pour la gestion des arguments en ligne de commande.

Commande d'installation :
pip install tensorflow torch torchvision flask pillow numpy

# 3. STRUCTURE DES FICHIERS
-------------------------
- main.py           : Script principal pour l'entraînement.
- app.py            : Serveur Flask pour l'interface Web.
- pipeline.py       : Prétraitement et chargement des données.
- models/           : Dossier contenant les architectures CNN.
- templates/        : Dossier contenant la page index.html.
- adama_model.pth   : Modèle PyTorch entraîné.
- adama_model.keras : Modèle TensorFlow entraîné.

# 4. INSTRUCTIONS D'UTILISATION
-----------------------------
Entraînement :
python main.py --framework tensorflow
python main.py --framework pytorch

Lancement de l'interface Web :
python app.py

Accédez ensuite à l'adresse suivante dans votre navigateur : 
http://127.0.0.1:5000
===========================================================
