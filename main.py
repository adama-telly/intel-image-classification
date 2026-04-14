# --- ÉTAPE 1 : IMPORTATION DES OUTILS ---
import argparse  # Pour permettre de taper des options dans le terminal (ex: --framework)
import os        # Pour manipuler les dossiers de ton ordinateur
import torch     # L'outil PyTorch (de Meta/Facebook)
import tensorflow as tf  # L'outil TensorFlow (de Google)

# On importe nos propres fonctions que nous avons créées dans les autres fichiers
from pipeline import get_tf_data, get_torch_data
#from models.pytorch_model import IntelCNN_PyTorch
from models.pytorch_model import MonModelPyTorch
from models.tensorflow_model import creer_model_tf

def main():
    # --- ÉTAPE 2 : PRÉPARER LES ARGUMENTS DE LA COMMANDE ---
    # On crée un "menu" pour que tu puisses choisir entre pytorch et tensorflow au lancement
    parser = argparse.ArgumentParser(description="Entraînement Intel Image Classification")
    parser.add_argument('--framework', type=str, choices=['pytorch', 'tensorflow'], required=True,
                        help="Tape 'pytorch' ou 'tensorflow' après --framework")
    args = parser.parse_args()

    # Tes informations personnelles pour la sauvegarde
    data_path = "C:/Users/ADAMA T BA/Desktop/mon_projet_CV/data"
    prenom = "adama" 

    # ==========================================
    # ÉTAPE 3 : SI TU AS CHOISI PYTORCH
    # ==========================================
    if args.framework == 'pytorch':
        print("--- Début de l'entraînement PyTorch ---")
        
        # On vérifie si ton PC a une carte graphique (GPU) pour aller plus vite
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"L'ordinateur utilise : {device}")

        # On récupère les images via le pipeline PyTorch
        train_loader = get_torch_data(data_path)
        
        # On crée le modèle et on l'envoie dans le processeur (CPU ou GPU)
        model = MonModelPyTorch().to(device)
        
        # On définit la "méthode d'apprentissage"
        criterion = torch.nn.CrossEntropyLoss() # Comment on mesure l'erreur
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # L'algorithme qui corrige l'IA

        # LA BOUCLE D'APPRENTISSAGE
        model.train()
        print("L'IA est en train d'apprendre... (cela peut être long)")
        
        # On ne fait qu'une seule "époque" (un tour complet des images) pour tester
        for images, labels in train_loader:
            # On envoie les images au processeur
            images, labels = images.to(device), labels.to(device)
            
            # L'IA fait une prédiction
            outputs = model(images)
            loss = criterion(outputs, labels) # On calcule l'erreur
            
            # L'IA se corrige toute seule
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # SAUVEGARDE : On enregistre le cerveau de l'IA dans un fichier
        torch.save(model.state_dict(), f"{prenom}_model.pth")
        print(f"✅ Terminé ! Fichier créé : {prenom}_model.pth")

    # ==========================================
    # ÉTAPE 4 : SI TU AS CHOISI TENSORFLOW
    # ==========================================
    elif args.framework == 'tensorflow':
        print("--- Début de l'entraînement TensorFlow ---")
        
        # On récupère les images (train et validation)
        train_ds, val_ds = get_tf_data(data_path)
        
        # On crée le modèle TensorFlow
        model = creer_model_tf()

        # On configure l'apprentissage (Compile)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'] # On veut voir le pourcentage de réussite
        )

        # LANCEMENT DE L'ENTRAÎNEMENT
        print("L'IA commence son entraînement TensorFlow...")
        model.fit(train_ds, validation_data=val_ds, epochs=1) # 1 tour complet

        # SAUVEGARDE
        model.save(f"{prenom}_model.keras")
        print(f"✅ Terminé ! Fichier créé : {prenom}_model.keras")

# C'est ici que le script commence réellement
if __name__ == "__main__":
    main()
