import fiftyone as fo

dataset = fo.Dataset.from_images_dir(
    "dataset/images/vid7",
    name="robot_dataset_7",
    persistent = True
)

print("Dataset:", dataset.name, "persistent:", dataset.persistent)

# Ouvre l'interface d'annotation (web local)
session = fo.launch_app(dataset,port=5151)
print("FiftyOne ouvert. Annote des boîtes dans le champ 'ground_truth'.")
# Empêche le script de se terminer
session.wait()
