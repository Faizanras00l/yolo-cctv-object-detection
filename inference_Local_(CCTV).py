from ultralytics import YOLO
import cv2, os, torch

MODEL_PATH = "cctv.pt"   # your trained weights

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device.upper())

model = YOLO(MODEL_PATH).to(device)

file_path = "test.jpg"   # change to your image or video

results = model(file_path, save=True, conf=0.5)
output_file = os.path.join(results[0].save_dir, os.path.basename(file_path))

if file_path.lower().endswith(('.jpg','.jpeg','.png')):
    img = cv2.imread(output_file)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Video saved at:", output_file)
