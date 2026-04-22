from ultralytics import YOLO
import os

# 1. Charge ton modèle
model = YOLO(r"C:\Users\theoc\Desktop\Projet_SYSMER_2A\runs\detect\entrainement_sysmer2\weights\best.pt")
#model = YOLO(r"C:\Users\theoc\Desktop\Projet_SYSMER_2A\best.pt")  

# 2. Chemin de ta vidéo
video_path = r"C:\Users\theoc\Desktop\Projet_SYSMER_2A\Calibrage Direct\CAM2.avi"

print("🚀 Lancement de la détection...")

# 3. Inférence avec flux (stream)
results = model.predict(
    source=video_path, 
    show=True,   
    save=False,   
    conf=0.5,
    stream=True, # Indispensable pour les vidéos longues
    device=0
)

# 4. On DOIT parcourir le générateur pour que le travail se fasse
# C'est ici que l'IA traite réellement les images
for r in results:
    # On récupère le dossier de sauvegarde depuis le dernier résultat traité
    path_sauvegarde = r.save_dir

print(f"\n✅ Terminé !")
print(f"👉 Ta vidéo annotée est ici : {os.path.abspath(path_sauvegarde)}")