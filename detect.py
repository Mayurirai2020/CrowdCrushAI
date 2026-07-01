from ultralytics import YOLO

print("=" * 50)
print("Crowd Crush Prevention - Stage 1")
print("=" * 50)

model = YOLO("yolo11n.pt")

print("Model loaded successfully.")
print("Starting person detection...")

results = model.predict(
    source="videos/input.mp4",
    classes=[0],
    conf=0.30,
    save=True,
    show=False
)

print("Detection completed!")