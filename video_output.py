import os
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Load image processor and model from Hugging Face
processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")

# Function to process video and extract frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize frame to fit model input
        frame = cv2.resize(frame, (1024, 720))  # Updated size
        frames.append(frame)

    cap.release()
 #   print(f"Total frames extracted: {len(frames)}")  # Print the total number of frames
    return np.array(frames)

# Function to predict deepfake
def predict_deepfake(frames):
    processed_frames = [processor(images=frame, return_tensors="pt")['pixel_values'] for frame in frames]
    inputs = torch.cat(processed_frames)
    outputs = model(inputs)
    predictions = outputs.logits.argmax(dim=-1).numpy()
    return predictions

# Function to explain predictions using LIME
def explain_prediction(frame):
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        frame,
        lambda x: model(processor(images=x, return_tensors="pt").pixel_values).logits.detach().numpy(),
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    return explanation

# Function to visualize LIME explanation
def visualize_lime_explanation(frame, explanation, label):
    temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=5, hide_rest=False)
    img_boundry = mark_boundaries(temp / 255.0, mask)
    return img_boundry

# Main function to process all videos in a directory
def main(video_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        print(f"Processing video: {video_file}")      
        frames = process_video(video_path)

        predictions = predict_deepfake(frames)
        print(f"Predictions for {video_file}:", predictions)

        # Create video writer for output video
        output_video_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_lime.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (1024, 720))

        # Explain predictions for the first frame and save the explanation
        # if len(frames) > 0:
        #     print("Processing image output...")
        #     explanation = explain_prediction(frames[0])
        #     label = predictions[0]  # Get the predicted class for the first frame
        #     img_boundry = visualize_lime_explanation(frames[0], explanation, label)
            
        #     # Save the image with LIME boundaries for the first frame
        #     output_image_path = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_lime.png")
        #     cv2.imwrite(output_image_path, img_boundry * 255)  # Scale to 0-255 for saving
        #     print(f"Saved LIME explanation to: {output_image_path}")
        
        
        num=0
        # Process each frame for the video output
        for i, frame in enumerate(frames):
            num=num+1
            print(f"Processing {num} out of {len(frames)} frames...")
            explanation = explain_prediction(frame)
            label = predictions[i]  # Get the predicted class for the current frame
            img_boundry = visualize_lime_explanation(frame, explanation, label)

            # Write the frame with LIME boundaries to the video
            out.write((img_boundry * 255).astype(np.uint8))

        out.release()  # Finalize the video file
        print(f"Saved LIME explanation video to: {output_video_path}")


# Replace with your folder path containing .mp4 videos and output folder
video_folder = './input'  # Update this path
output_folder = './output'  # Update this path
main(video_folder, output_folder)
