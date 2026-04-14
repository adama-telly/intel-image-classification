import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras import layers, models

# ==========================================
#  MON ARCHITECTURE POUR PYTORCH
# ==========================================
class MonModelPyTorch(nn.Module):
    def __init__(self):
        super(MonModelPyTorch, self).__init__()
        # Couche 1 : Cherche des formes simples (lignes, bords)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # Couche 2 : Cherche des motifs plus complexes
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Couche de Pooling : Divise la taille de l'image par 2 pour simplifier
        self.pool = nn.MaxPool2d(2, 2)
        
        # Couches finales (Décision) :
        # Après les réductions, l'image 150x150 devient 37x37
        self.fc1 = nn.Linear(64 * 37 * 37, 128)
        self.fc2 = nn.Linear(128, 6) # 6 classes (mer, montagne, etc.)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Image devient 75x75
        x = self.pool(F.relu(self.conv2(x))) # Image devient 37x37
        x = x.view(-1, 64 * 37 * 37)        # On "aplatit" l'image en une ligne
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


print("Architectures des modèles créées avec succès !")
