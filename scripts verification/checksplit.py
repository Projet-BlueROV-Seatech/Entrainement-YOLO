import fiftyone as fo

ds = fo.load_dataset("robot_dataset_2")

print(ds.distinct("bluerov.detections.label"))

print("train:", len(ds.match_tags("train")))
print("val:",   len(ds.match_tags("val")))
print("reste :", len(ds))