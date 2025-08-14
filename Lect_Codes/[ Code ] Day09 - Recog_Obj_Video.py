import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import cv2  # OpenCV library
import numpy as np
import sys  # Import the sys module to exit the program

# 1. Prepare Deep Learning Model
# ==================================
print("Loading the deep learning model... Please wait.")

# Check for GPU availability and set the device.
# Note: The DETR model is heavy and will run very slowly on a CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load the pre-trained DETR (DEtection TRansformer) model and image processor.
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(DEVICE)

print("✅ Model ready!")

# 2. Visualization Settings
# ==================================
# List of colors to use for the detected object bounding boxes.
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# Set the font for labels. (Uses default font if arial.ttf is not found)
try:
    font = ImageFont.truetype("arial.ttf", 15)
except IOError:
    font = ImageFont.load_default()

# 3. Real-time Webcam Processing
# ==================================
# Open the default webcam connected to the computer (device 0).
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam. Please check the camera connection.")

print("\n🚀 Starting real-time object detection. Press 'q' to quit.")

# Continuously process frames until the 'q' key is pressed.
while True:
    # Read the current frame (a single image) from the webcam.
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # The deep learning model uses PIL Image objects, so convert the OpenCV BGR image to RGB.
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # --- Perform Object Detection ---
    inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    # Post-process the model's output to extract objects with a confidence score > 90%.
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # --- Visualize Results ---
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        
        # Assign a color based on the label.
        color_idx = label.item() % len(COLORS)
        color = tuple(int(c * 255) for c in COLORS[color_idx])

        # Draw the bounding box around the object.
        draw.rectangle(box, outline=color, width=3)
        
        # Display the object label and confidence score as text.
        label_text = f"{model.config.id2label[label.item()]}: {score:.2f}"
        
        # Note: The 'textbbox' method requires Pillow version 8.0.0 or higher.
        try:
            text_bbox = draw.textbbox((box[0], box[1]), label_text, font=font)
            draw.rectangle(text_bbox, fill=color)
        except AttributeError:
            # Fallback for older Pillow versions
            pass
        
        draw.text((box[0], box[1]), label_text, fill="white", font=font)

    # Convert the PIL image back to the BGR format that OpenCV can display.
    processed_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Show the processed video in a window named 'Real-time Object Detection'.
    cv2.imshow('Real-time Object Detection', processed_frame)

    # Wait for a key press for 1ms, and if 'q' is pressed, exit the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("'q' pressed, terminating the program...")
        break

# 4. Cleanup
# ==================================
# Release the webcam and destroy all windows when done.
cap.release()
cv2.destroyAllWindows()
print("Program terminated successfully.")
sys.exit() # Add this line to ensure the program exits completely