from ultralytics import YOLO
import cv2
import os

# Load your trained YOLOv8 model weights
model = YOLO('runs\\detect\\train2\\weights\\best.pt')

inference_folder = 'data/images/inference'
output_folder = 'inference_results'
log_file = 'inference_results/log.txt'

os.makedirs(output_folder, exist_ok=True)

# Open log file in write mode
with open(log_file, 'w') as log:
    # Process all images
    image_files = [f for f in os.listdir(inference_folder) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        log.write("No images found in inference folder.\n")
    else:
        for idx, filename in enumerate(image_files, 1):
            image_path = os.path.join(inference_folder, filename)
            img = cv2.imread(image_path)
            if img is None:
                log.write(f"Failed to load {filename}\n")
                continue

            # Run prediction
            results = model(image_path)

            log.write(f"\nImage {idx}/{len(image_files)}: {filename}\n")
            if len(results[0].boxes) == 0:
                log.write("  No detections.\n")
            else:
                # Log detected classes and confidence
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = box.conf[0]
                    class_name = model.names[cls]
                    log.write(f"  - Detected: {class_name}, Confidence: {conf:.2f}\n")

            # Save annotated image
            annotated_img = results[0].plot()
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, annotated_img)

print(f"Inference complete. Logs saved to {log_file}")
