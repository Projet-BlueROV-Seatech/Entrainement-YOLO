from ultralytics import YOLO

model = YOLO(r"C:\Users\theoc\Desktop\Projet_SYSMER_2A\runs\detect\entrainement_sysmer_Eau\weights\best.pt")

print("🚀 Début de l'entraînement...")

results = model.train(
    data=r"C:\Users\theoc\Desktop\Projet_SYSMER_2A\yolov5-dataset-eau\yolov5-dataset\dataset.yaml", 
    epochs=50,                           
    imgsz=640,                           
    batch=16,                            
    name="entrainement_sysmer_Eau_Sequence",
    device=0,  
    workers=0
    
)

print("✅ Entraînement terminé !")