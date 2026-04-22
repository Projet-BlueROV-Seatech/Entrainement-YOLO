import fiftyone as fo

ds = fo.load_dataset("robot_dataset_7")

# Reproductible
ds.shuffle(seed=42)

# Nettoyage si relancé
ds.untag_samples(["train", "val"])

LABEL_FIELD = "bluerov"
# Train = 80%
annotated = ds.match(
    fo.ViewField(LABEL_FIELD).exists()
)
n_train = int(1 * len(annotated))
train_view = annotated.take(n_train)
train_view.tag_samples("train")

# Val = le reste
val_view = annotated.exclude_by("id", train_view.values("id"))
val_view.tag_samples("val")

print("train:", len(ds.match_tags("train")))
print("val:",   len(ds.match_tags("val")))