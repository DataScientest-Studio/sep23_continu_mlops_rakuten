import streamlit as st
import requests
from PIL import Image
import io

# Titre de l'application Streamlit
st.title("Envoi de données à l'API Rakuten")

if "new_productid" not in st.session_state:
    st.session_state.new_productid = None

categories = {
            2583: "Accessoires de piscine",
            1560: "Mobilier et décoration pour maison",
            1300: "Modélisme",
            2060: "Accessoires et décorations pour maison",
            2522: "Papeterie",
            1280: "Jouets et peluches enfant",
            2403: "Livres / Magazines en lots",
            2280: "Livres documentaires",
            1920: "Linge de maison",
            1160: "Jeu de cartes à collectionner",
            1320: "Puériculture",
            10: "Littérature étrangère",
            2705: "Littérature historique",
            1140: "Figurines",
            2582: "Accessoires et décorations pour jardin",
            40: "Jeux vidéo neufs",
            2585: "Accessoires et outils de jardinage",
            1302: "Jeux de plein air",
            50: "Accessoires de gaming",
            2462: "Jeux vidéos d’occasion",
            2905: "Jeux vidéo à télécharger",
            60: "Consoles de jeux vidéo",
            2220: "Animalerie",
            1301: "Articles pour nouveaux-nés et bébés",
            1940: "Épicerie",
            1180: "Figurines à peindre et à monter",
            1281: "Jouets et peluches enfant"
        }

# Création d'un formulaire pour saisir les données
with st.form("my_form"):
    titre = st.text_input("Titre")
    description = st.text_area("Description")
    image = st.file_uploader("Téléchargez une Image", type=['jpg', 'png'])

    # Bouton de soumission du formulaire
    submitted = st.form_submit_button("Envoyer à FastAPI")

    if submitted and image:
        # Conversion de l'image en bytes pour l'envoi
        image_bytes = image.getvalue()

        # Préparation de la requête
        response = requests.post(
            "http://localhost:8000/get_prediction_input",
            files={
                "titre": (None, titre),
                "description": (None, description),
                "image": (image.name, image_bytes, image.type)
            }
        )

        # Affichage de la réponse de FastAPI
        if response.status_code == 200:
            st.success("Réponse reçue de l'API:")
            
            response_data = response.json()
            new_productid = response_data.get("new_productid", "Non spécifié")
            cat_predicted = response_data.get("predicted category", "Non spécifié")

            st.write(f"catégorie du produit : {cat_predicted}")
            st.write(f"ID du nouveau produit : {new_productid}")
            st.session_state.new_productid = new_productid
        else:
            st.error("Erreur dans la réponse de FastAPI")

# Création d'un formulaire pour saisir les données de feedback
            
with st.form("feedback_form"):
    # Supposons que votre objet Item a ces champs
    
    
    st.markdown(f"**Product id** {st.session_state.new_productid}.")
    categorie_choisie = st.selectbox("Choisissez une catégorie :", list(categories.values()))

    categorie_choisie_code = None
    for code, description in categories.items():
        if description == categorie_choisie:
          categorie_choisie_code = code
          break
    
    # Bouton de soumission du formulaire
    submitted2 = st.form_submit_button("Envoyer le Feedback")

    if submitted2:
        # Préparation des données à envoyer
        feedback_data = {
            "productID": str(st.session_state.new_productid),  # Remplacez par votre productID
            "categoryID": str(categorie_choisie_code)
        }

        # Envoi des données à l'API
        response = requests.post(
            "http://localhost:8000/Feedback",
            json=feedback_data
        )

        # Affichage de la réponse de l'API
        if response.status_code == 200:
            st.success("Feedback envoyé avec succès.")
            st.json(response.json())
        else:
            st.error("Erreur lors de l'envoi du feedback.")


