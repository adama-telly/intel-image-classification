import streamlit as st
import tensorflow as tf
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from models.pytorch_model import MonModelPyTorch

# 1. CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Intel Image Classification ", layout="centered")

st.title("Intel Image Classification with Adama Telly Ba")

# 2. BARRE LATÉRALE (SIDEBAR)
st.sidebar.header("Parameters")
model_choice = st.sidebar.selectbox("Choose the AI ​​model", ("TensorFlow", "PyTorch"))

# Liste des classes du dataset Intel
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# 3. INTERFACE DE TÉLÉCHARGEMENT
uploaded_file = st.file_uploader("Select a landscape image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Affichage de l'image sélectionnée
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='selected image', use_container_width=True)
    
    # 4. ANALYSE AVEC ANIMATION (SPINNER)
    with st.spinner(f"Analysis in progress with {model_choice}..."):
        try:
            if model_choice == "TensorFlow":
                # Logique TensorFlow (.keras)
                model = tf.keras.models.load_model('adama_model.keras')
                img_tf = image.resize((150, 150))
                img_array = np.array(img_tf) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                preds = model.predict(img_array, verbose=0)
                resultat = CLASSES[np.argmax(preds)]
            else:
                # Logique PyTorch (.pth)
                model = MonModelPyTorch()
                model.load_state_dict(torch.load('adama_model.pth', map_location='cpu'))
                model.eval()
                transform = transforms.Compose([
                    transforms.Resize((150, 150)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                img_pt = transform(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(img_pt)
                    _, predicted = torch.max(outputs, 1)
                    resultat = CLASSES[predicted.item()]
            
            # 5. AFFICHAGE DU RÉSULTAT FINAL
            st.success(f"Our model IA predicts : **{resultat.upper()}**")
            
        except Exception as e:
            st.error(f"Erreur lors de la classification : {e}")

# 6. INFORMATIONS BAS DE PAGE (SIDEBAR)
st.sidebar.markdown("---") 
st.sidebar.subheader("Informations")
st.sidebar.write(f"** active Modele :** {model_choice}")

if model_choice == "TensorFlow":
    st.sidebar.code("Fichier : adama_model.keras")
else:
    st.sidebar.code("Fichier : adama_model.pth")

st.sidebar.info("Projet : Intel Image Classification with **Adama Telly Ba**")
