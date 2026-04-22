import shutil
import fiftyone as fo
from fiftyone import ViewField as F 

# Charge ou crée le dataset
dataset = fo.load_dataset("robot_dataset_7")

export_dir = "./yolov5-dataset7/"
label_field = "bluerov"  # for example

# The splits to export
splits = ["train", "val"]

# All splits must use the same classes list
classes = ["bluerov"]


# Export the splits
for split in splits:
    split_view = dataset.match_tags(split)
    split_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field=label_field,
        split=split,
        classes=classes,
    )

print("Export YOLO -> yolov5-dataset")


# Lance l'app et GARDE la session
#session = fo.launch_app(dataset)

# Empêche le script de se terminer


# Reload to see changes
#dataset.reload()
#LABEL_FIELD = "bluerov"

#annotated = dataset.match(
#    fo.ViewField(LABEL_FIELD).exists()
#)

#print("Images avec labels:", len(annotated))
    
#session.wait()