import cv2

# Open input video
cap = cv2.VideoCapture("videos/input.mp4")

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Save output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("outputs/grid_output.mp4", fourcc, fps, (width, height))

# Grid size
ROWS = 4
COLS = 4

cell_width = width // COLS
cell_height = height // ROWS

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Draw vertical lines
    for i in range(1, COLS):
        x = i * cell_width
        cv2.line(frame, (x, 0), (x, height), (0, 255, 0), 2)

    # Draw horizontal lines
    for j in range(1, ROWS):
        y = j * cell_height
        cv2.line(frame, (0, y), (width, y), (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()

print("Grid drawn successfully!")
print("Saved in outputs/grid_output.mp4")