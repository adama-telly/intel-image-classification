# On importe les outils nécessaires (comme on sortirait des casseroles)
import tensorflow as tf
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# --- PARAMÈTRES DE BASE ---
# On dit à l'ordi : "Toutes les images devront faire 150 pixels de large et de haut"
IMG_SIZE = (150, 150)
# On donne les images par groupes de 32 pour ne pas fatiguer la mémoire de l'ordi
BATCH_SIZE = 32

# ==========================================
# 1. LA PRÉPARATION POUR TENSORFLOW (Google)
# ==========================================
def get_tf_data(data_dir):
    # Cette fonction va dans ton dossier "data/seg_train"
    # Elle sépare automatiquement les images en deux : 
    # 80% pour apprendre (training) et 20% pour s'entraîner à l'examen (validation)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/seg_train/seg_train",
        validation_split=0.2, # 20% pour le test pendant l'entraînement
        subset="training",    # On prend la partie "Apprentissage"
        seed=123,             # Un chiffre au hasard pour que le mélange soit toujours le même
        image_size=IMG_SIZE,  # Redimensionne en 150x150
        batch_size=BATCH_SIZE # Par paquets de 32
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/seg_train/seg_train",
        validation_split=0.2,
        subset="validation",  # On prend la partie "Examen"
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # NORMALISATION : Les couleurs vont de 0 à 255. 
    # L'IA préfère les petits chiffres entre 0 et 1. 
    # On divise donc tout par 255 (Rescaling).
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    
    # On applique cette division à toutes nos images
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    return train_ds, val_ds

# ==========================================
# 2. LA PRÉPARATION POUR PYTORCH (Facebook/Meta)
# ==========================================
def get_torch_data(data_dir):
    # PyTorch utilise des "Transforms" (des transformations à la chaîne)
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),      # Redimensionne en 150x150
        transforms.RandomHorizontalFlip(), # Retourne l'image parfois (pour la diversité)
        transforms.ToTensor(),             # Transforme l'image en chiffres (0 à 1)
        # On ajuste les couleurs pour qu'elles soient "standard"
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # On va chercher les images dans le dossier
    train_set = datasets.ImageFolder(f"{data_dir}/seg_train/seg_train", transform=transform)
    
    # On crée le "chargeur" qui donnera les paquets de 32 images au modèle
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    
    return loader

# Petit message pour dire que tout est chargé dans la mémoire de Python
print("Le système de préparation des images est prêt !")
if __name__ == "__main__":
    # Test du chargement
    data_path = "C:/Users/ADAMA T BA/Desktop/mon_projet_CV/data"
    print("Test du chargement TensorFlow...")
    try:
        train_tf, val_tf = get_tf_data(data_path)
        print("✅ Pipeline TensorFlow OK !")
    except Exception as e:
        print(f"❌ Erreur TF: {e}")

    print("\nTest du chargement PyTorch...")
    try:
        train_torch = get_torch_data(data_path)
        print("✅ Pipeline PyTorch OK !")
    except Exception as e:
        print(f"❌ Erreur PyTorch: {e}")
