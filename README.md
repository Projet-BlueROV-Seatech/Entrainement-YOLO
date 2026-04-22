# 🤖 SYSMER — Entraînement d'un modèle YOLOv8 sur le BlueROV

Ce guide explique comment créer un dataset d'images annoté et entraîner un modèle de détection **YOLOv8** sur un robot sous-marin (**BlueROV**), de l'extraction des frames vidéo jusqu'à la validation du modèle.

---

## 📋 Table des matières

1. [Installation de l'environnement](#1-installation-de-lenvironnement)
2. [Architecture du projet](#2-architecture-du-projet)
3. [Pipeline de préparation des données](#3-pipeline-de-préparation-des-données)
4. [Entraînement YOLO](#4-entraînement-yolo)
5. [Validation sur vidéo](#5-validation-sur-vidéo)
6. [Scripts utilitaires](#6-scripts-utilitaires)
7. [Modèles pré-entraînés](#7-modèles-pré-entraînés)
8. [Problèmes fréquents](#8-problèmes-fréquents)

---

## 1. Installation de l'environnement

### 1.1 Python

Téléchargez **Python 3.10 ou supérieur** : **https://www.python.org/downloads/**

> ⚠️ Cochez impérativement **"Add Python.exe to PATH"** lors de l'installation.

### 1.2 IDE — Spyder via Anaconda

Téléchargez **Anaconda** : **https://www.anaconda.com/download**

Lors de l'installation, cochez `☑ Add Anaconda to my PATH environment variable`.

Lancez ensuite **Spyder** depuis l'Anaconda Navigator ou depuis l'Anaconda Prompt :
```bash
spyder
```

### 1.3 Installation des bibliothèques

Ouvrez l'**Anaconda Prompt** et exécutez :

```bash
pip install ultralytics fiftyone opencv-python
```

| Bibliothèque | Rôle |
|---|---|
| `ultralytics` | Entraînement et inférence YOLOv8 |
| `fiftyone` | Gestion du dataset et annotation |
| `opencv-python` | Lecture et extraction des frames vidéo |

### 1.4 (Optionnel) Accélération GPU

Pour utiliser un GPU NVIDIA, installez PyTorch avec CUDA **avant** les autres bibliothèques. Consultez le configurateur : **https://pytorch.org/get-started/locally/**

Vérifiez ensuite que le GPU est bien détecté :
```python
import torch
print(torch.cuda.is_available())  # Doit afficher True
```

---

## 2. Architecture du projet

Organisez votre dossier de travail comme suit. Les noms en *italique* sont des **exemples que vous pouvez modifier** librement.

```
Projet_SYSMER/
│
├── videos/
│   └── ma_video.avi              ← Votre vidéo source
│
├── dataset/
│   └── images/
│       └── mon_dataset/          ← Frames extraites (créé automatiquement)
│
├── yolov5-dataset/               ← Dataset formaté pour YOLO (créé automatiquement)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── dataset.yaml
│
├── runs/
│   └── detect/
│       └── mon_entrainement/
│           └── weights/
│               └── best.pt       ← Modèle entraîné (créé automatiquement)
│
└── scripts entrainement/
    ├── exportImages.py
    ├── createDataset.py
    ├── labelIt.py
    ├── split.py
    ├── convert2yolo.py
    ├── train.py
    └── test_video.py             ← À venir
```

> 💡 Chaque script contient en haut du fichier les variables à personnaliser (noms de dossiers, de datasets, etc.). Vous n'avez pas à modifier le reste du code.

---

## 3. Pipeline de préparation des données

Les scripts doivent être exécutés **dans l'ordre suivant** :

```
exportImages → createDataset → annotation manuelle → split → convert2yolo
```

---

### Étape 3.1 — Extraction des frames (`exportImages.py`)

Ce script lit votre vidéo source et extrait une image toutes les **10 frames**.

**Variables à adapter en haut du script :**
```python
video_path = "videos/ma_video.avi"        # ← Chemin vers votre vidéo
out_dir    = "dataset/images/mon_dataset"  # ← Nom du dossier de sortie
step       = 10                            # ← 1 frame extraite toutes les X frames
```

**Exécution :**
```bash
python "scripts entrainement/exportImages.py"
```

✅ Résultat : les frames `.jpg` apparaissent dans `dataset/images/mon_dataset/`.

---

### Étape 3.2 — Création de la base de données (`createDataset.py`)

Ce script importe les images dans **FiftyOne** et ouvre l'interface d'annotation dans le navigateur.

**Variables à adapter :**
```python
fo.Dataset.from_images_dir(
    "dataset/images/mon_dataset",  # ← Doit correspondre au out_dir de l'étape 3.1
    name="mon_dataset_fiftyone",   # ← Nom libre pour identifier ce dataset dans FiftyOne
    persistent=True                # ← Ne pas modifier : sauvegarde la base de données
)
```

**Exécution :**
```bash
python "scripts entrainement/createDataset.py"
```

✅ FiftyOne s'ouvre dans votre navigateur à `http://localhost:5151`.

> ⚠️ **Ne fermez pas le terminal** pendant l'annotation : le script maintient la session ouverte.

---

### Étape 3.3 — Annotation manuelle

Dans l'interface web de FiftyOne :

1. Cliquez sur une image pour l'ouvrir
2. Sélectionnez l'outil **Bounding Box** (rectangle)
3. Tracez un cadre autour du robot
4. Dans le champ classe, saisissez le nom de votre objet — par exemple `bluerov` — et **utilisez toujours le même nom** sur toutes les images
5. Répétez pour toutes les images où le robot est visible

> 💡 Pour **reprendre une session d'annotation** sans écraser les données existantes, utilisez `labelIt.py` plutôt que `createDataset.py` :
> ```bash
> python "scripts entrainement/labelIt.py"
> ```
> Adaptez le `name` dans ce script pour qu'il corresponde au dataset créé à l'étape 3.2.

Pour vérifier le nombre d'images annotées à tout moment :
```bash
python "scripts verification/checkLabels.py"
```

---

### Étape 3.4 — Répartition train / val (`split.py`)

Ce script mélange les images annotées et leur attribue les tags `train` (80 %) et `val` (20 %).

**Variables à adapter :**
```python
ds          = fo.load_dataset("mon_dataset_fiftyone") # ← Nom du dataset (étape 3.2)
LABEL_FIELD = "bluerov"                               # ← Nom de la classe (étape 3.3)
```

**Exécution :**
```bash
python "scripts entrainement/split.py"
```

✅ Résultat affiché :
```
train: XXX
val:   XXX
```

Pour vérifier la répartition :
```bash
python "scripts verification/checksplit.py"
```

---

### Étape 3.5 — Export au format YOLO (`convert2yolo.py`)

Ce script crée le dossier `yolov5-dataset/` avec l'arborescence et le fichier `dataset.yaml` requis par Ultralytics.

**Variables à adapter :**
```python
dataset     = fo.load_dataset("mon_dataset_fiftyone") # ← Nom du dataset (étape 3.2)
export_dir  = "./yolov5-dataset/"                     # ← Dossier de sortie
label_field = "bluerov"                               # ← Nom de la classe (étape 3.3)
classes     = ["bluerov"]                             # ← Liste des classes
```

**Exécution :**
```bash
python "scripts entrainement/convert2yolo.py"
```

✅ Résultat : création du dossier `yolov5-dataset/` prêt pour l'entraînement.

---

## 4. Entraînement YOLO (`train.py`)

### 4.1 Adapter les paramètres

Ouvrez `train.py` et modifiez les variables suivantes :

```python
from ultralytics import YOLO

# Modèle de départ :
# - "yolov8n.pt" pour partir de zéro (téléchargé automatiquement)
# - Chemin vers un best.pt existant pour continuer un entraînement
model = YOLO("yolov8n.pt")

results = model.train(
    data=r"C:\Users\VotreNom\Projet_SYSMER\yolov5-dataset\dataset.yaml",
    epochs=50,              # Nombre de passages sur le dataset
    imgsz=640,              # Taille des images en pixels
    batch=16,               # Réduire à 8 ou 4 si erreur mémoire GPU
    name="mon_entrainement",# ← Nom du dossier de résultats dans runs/detect/
    device=0,               # 0 = GPU NVIDIA, "cpu" = processeur
    workers=0
)
```

### 4.2 Lancer l'entraînement

```bash
python "scripts entrainement/train.py"
```

L'entraînement peut durer de **quelques minutes** (GPU) à **plusieurs heures** (CPU) selon la taille du dataset et le nombre d'epochs.

✅ Le meilleur modèle est automatiquement sauvegardé dans :
```
runs/detect/mon_entrainement/weights/best.pt
```

---

## 5. Validation sur vidéo (`test_video.py`)

> 🚧 **Ce script est en cours d'ajout.** Il sera placé dans `scripts entrainement/test_video.py`.

En attendant, vous pouvez tester votre modèle avec ce code minimal à copier dans un nouveau fichier :

```python
from ultralytics import YOLO
import cv2

# Chargez votre modèle entraîné
model = YOLO(r"runs/detect/mon_entrainement/weights/best.pt")

# Ouvrez la vidéo de test
cap = cv2.VideoCapture("videos/ma_video_test.avi")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("Détection BlueROV", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

**Commandes :** appuyez sur `Q` pour quitter l'affichage.

---

## 6. Scripts utilitaires

### `list.py` — Lister les datasets FiftyOne existants

Utile pour retrouver le nom exact d'un dataset créé précédemment :
```bash
python "scripts entrainement/list.py"
```

### `copyDataset.py` — Sauvegarder un dataset

Crée une copie de sauvegarde avant de modifier un dataset annoté. À utiliser avant de relancer `split.py` ou `convert2yolo.py` :
```bash
python "scripts entrainement/copyDataset.py"
```

### `checkLabels.py` — Vérifier les annotations

Affiche le nombre d'images annotées dans le dataset :
```bash
python "scripts verification/checkLabels.py"
```

### `checksplit.py` — Vérifier la répartition train/val

Affiche les compteurs train / val et les classes détectées :
```bash
python "scripts verification/checksplit.py"
```

---

## 7. Modèles pré-entraînés

Deux modèles sont fournis dans le dossier `models_yolo/` et peuvent être utilisés directement pour des tests ou comme point de départ pour un nouvel entraînement :

| Fichier | Contexte |
|---|---|
| `best_air.pt` | Entraîné sur des séquences filmées **hors de l'eau** |
| `best_eau.pt` | Entraîné sur des séquences filmées **dans l'eau** |

Pour utiliser un modèle existant comme base dans `train.py`, remplacez `"yolov8n.pt"` par le chemin vers le `.pt` souhaité.

---

## 8. Problèmes fréquents

**`ModuleNotFoundError: No module named 'fiftyone'`**
→ Exécutez `pip install fiftyone` depuis l'Anaconda Prompt.

**FiftyOne ne s'ouvre pas dans le navigateur**
→ Ouvrez manuellement `http://localhost:5151`. Vérifiez que le pare-feu Windows ne bloque pas le port 5151.

**`Dataset with name '...' not found`**
→ Le nom dans le script ne correspond pas au dataset créé. Lancez `list.py` pour voir les noms disponibles.

**Erreur mémoire GPU (`CUDA out of memory`)**
→ Réduisez le paramètre `batch` dans `train.py` : passez de `16` à `8` ou `4`.

**L'entraînement tourne sur CPU au lieu du GPU**
→ Vérifiez avec `torch.cuda.is_available()`. Si `False`, réinstallez PyTorch avec le bon support CUDA (voir section 1.4).

**Les annotations disparaissent après fermeture**
→ Vérifiez que `persistent=True` est bien présent dans `createDataset.py`.
