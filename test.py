import torch
import cv2

# Load YOLOv5 model with your custom weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best1010.pt', force_reload=True)

# Set the model to eval mode
model.eval()

# Load image
image_path = '00000012.jpg'
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run inference
results = model(img_rgb)

# Parse results
detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, class]

if len(detections) == 0:
    print("No objects detected.")
else:
    print(f"{len(detections)} object(s) detected.")
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put label text
        cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save result
    output_path = 'output.jpg'
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to {output_path}")

