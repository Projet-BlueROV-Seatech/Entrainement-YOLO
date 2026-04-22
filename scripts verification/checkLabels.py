import fiftyone as fo

# Charge ou crée le dataset
dataset = fo.load_dataset("robot_dataset_2")
print(dataset.persistent)

# Lance l'app et GARDE la session
#session = fo.launch_app(dataset)

# Empêche le script de se terminer


# Reload to see changes
#dataset.reload()
LABEL_FIELD = "bluerov"

annotated = dataset.match(
    fo.ViewField(LABEL_FIELD).exists()
)

print("Images avec labels:", len(annotated))
    
#session.wait()