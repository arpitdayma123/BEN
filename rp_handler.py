import runpod
import torch
import os
import requests
from huggingface_hub import hf_hub_download
import base64
import tempfile
import time

# Assuming ben_base.py is in the same directory or PYTHONPATH
from ben_base import BEN_Base

# Global model initialization
print("Initializing model...")
model = BEN_Base()
model_path = hf_hub_download("PramaLLC/BEN2", "BEN2_Base.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.loadcheckpoints(model_path)
model.to(device)
model.eval()
print("Model initialized successfully.")

def handler(event):
    """
    Handles video processing requests.
    Downloads a video from a URL, processes it using the BEN_Base model,
    and returns the processed video encoded in base64.
    """
    print("Handler started.")
    tmp_video_path = None  # Initialize to ensure it's defined in finally block

    try:
        video_url = event['input'].get('video_url')
        if not video_url:
            return {"error": "video_url not found in input"}

        print(f"Downloading video from: {video_url}")
        # Download the video using requests
        response = requests.get(video_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Create a temporary file to save the downloaded video
        # delete=False is important to allow reopening the file by its name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            for chunk in response.iter_content(chunk_size=8192):
                tmpfile.write(chunk)
            tmp_video_path = tmpfile.name  # Get the path of the temporary file
        print(f"Video downloaded to temporary file: {tmp_video_path}")

        # Define the output path for the processed video.
        # The segment_video function in ben_base.py saves its output as foreground.mp4
        # in the specified output_path.
        output_directory = "./" 
        processed_video_path = os.path.join(output_directory, "foreground.mp4")

        print(f"Starting video segmentation. Input: {tmp_video_path}, Output dir: {output_directory}")
        # Process the video using the model
        # The segment_video method will save the output to "foreground.mp4" in output_directory
        model.segment_video(video_path=tmp_video_path, output_path=output_directory)
        print(f"Video segmentation completed. Expected output: {processed_video_path}")

        # Check if the processed video file exists
        if os.path.exists(processed_video_path):
            print(f"Processed video found at: {processed_video_path}")
            # Read the processed video file in binary mode
            with open(processed_video_path, "rb") as video_file:
                video_content = video_file.read()

            # Encode the video content to base64
            encoded_video = base64.b64encode(video_content).decode('utf-8')
            print("Video encoded to base64.")

            return {
                "output_video_base64": encoded_video,
                "message": "Video processed successfully"
            }
        else:
            print(f"Error: Processed video not found at {processed_video_path}")
            # List files in output_directory for debugging
            if os.path.exists(output_directory):
                print(f"Contents of {output_directory}: {os.listdir(output_directory)}")
            else:
                print(f"Output directory {output_directory} does not exist.")
            return {"error": f"Processed video not found at {processed_video_path}"}

    except requests.exceptions.RequestException as e:
        print(f"Error downloading video: {e}")
        return {"error": f"Error downloading video: {e}"}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": f"An error occurred: {e}"}
    finally:
        # Clean up: remove the temporary input video file
        if tmp_video_path and os.path.exists(tmp_video_path):
            try:
                os.remove(tmp_video_path)
                print(f"Cleaned up temporary input file: {tmp_video_path}")
            except Exception as e:
                print(f"Error cleaning up temporary input file {tmp_video_path}: {e}")
        
        # Clean up: remove the processed video file (foreground.mp4)
        # This path is defined based on where segment_video saves its output
        final_processed_video_path = os.path.join("./", "foreground.mp4")
        if os.path.exists(final_processed_video_path):
            try:
                os.remove(final_processed_video_path)
                print(f"Cleaned up processed video file: {final_processed_video_path}")
            except Exception as e:
                print(f"Error cleaning up processed video file {final_processed_video_path}: {e}")


# Standard RunPod boilerplate
if __name__ == '__main__':
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
