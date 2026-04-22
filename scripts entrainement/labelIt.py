import fiftyone as fo

# Charge ou crée le dataset
dataset = fo.load_dataset("robot_dataset_2")

# Lance l'app et GARDE la session
session = fo.launch_app(dataset)

# Empêche le script de se terminer
session.wait()
