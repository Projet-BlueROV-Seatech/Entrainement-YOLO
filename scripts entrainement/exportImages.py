import os, cv2

video_path = "videos/output2.avi"
out_dir = "dataset/images/output2"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0
step = 10  # prends 1 frame / 10

saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_id % step == 0:
        out_path = os.path.join(out_dir, f"frame_{frame_id:06d}.jpg")
        cv2.imwrite(out_path, frame)
        saved += 1
    frame_id += 1

cap.release()
print("frames sauvées:", saved)
