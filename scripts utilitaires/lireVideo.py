# importer opencv - librairie de traitement d'image open source 
# # si elle n'est pas installée : pip install opencv-python 
import cv2 

# Chemin vers la vidéo .mov
chemin = "../videos/BillyBassinX.mov"

# Ouvre la vidéo et crée un objet VideoCapture
cap = cv2.VideoCapture(chemin)

# Vérifie si la vidéo est bien ouverte
if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la vidéo")
    exit()

while True:
    # Lit une image extraite de la vidéo
    # ret est une variable booléenne qui indique si la lecture a réussi
    ret, image = cap.read()

    # Si plus d'images, on quitte la boucle while
    if not ret:
        break

    # Affiche l'image
    cv2.imshow("Video", image)

    # Appuyer sur 'q' pour quitter
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()