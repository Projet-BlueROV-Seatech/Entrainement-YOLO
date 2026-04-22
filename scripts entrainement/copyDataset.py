import fiftyone as fo

src = fo.load_dataset("robot_dataset_2")

dst = src.clone(
    name="robot_dataset_2_copy",
    persistent=True,
)

print("Copié ->", dst.name)