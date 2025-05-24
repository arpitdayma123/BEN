import runpod
import torch
import os
import requests
from huggingface_hub import hf_hub_download
import tempfile
import time
import sys
import uuid

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
        job_input = event['input']
        video_url = job_input.get('video_url')
        if not video_url:
            return {"error": "video_url not found in input"}

        max_frames_input = job_input.get('max_frames')
        max_frames_to_use = sys.maxsize  # Default to processing all frames

        if max_frames_input is not None:
            try:
                max_frames_val = int(max_frames_input)
                if max_frames_val > 0:
                    max_frames_to_use = max_frames_val
                # If max_frames_val is 0 or negative, it will default to sys.maxsize (all frames)
                # A message for this case could be added if desired.
            except ValueError:
                # Non-integer input, default to sys.maxsize (all frames)
                print(f"Invalid value for max_frames: {max_frames_input}. Defaulting to all frames.")
        
        print(f"Using max_frames: {max_frames_to_use if max_frames_to_use != sys.maxsize else 'all'}")
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
        # Define processed_video_path consistently. It's created by model.segment_video
        processed_video_path = os.path.join(output_directory, "foreground.mp4")

        print(f"Starting video segmentation. Input: {tmp_video_path}, Output dir: {output_directory}")
        # Process the video using the model
        model.segment_video(video_path=tmp_video_path, output_path=output_directory, max_frames=max_frames_to_use)
        print(f"Video segmentation completed. Expected output: {processed_video_path}")

        if os.path.exists(processed_video_path):
            print(f"Processed video found at: {processed_video_path}. Attempting upload to BunnyCDN.")
            
            unique_filename = f"{uuid.uuid4()}.mp4"
            # Storage zone name is 'zockto' as per instruction
            upload_url = f"https://storage.bunnycdn.com/zockto/videos/{unique_filename}"
            # Access key as per instruction
            access_key = "17e23633-2a7a-4d29-9450be4d6c8e-e01f-45f4" 
            headers = {"AccessKey": access_key, "Content-Type": "video/mp4"}

            try:
                with open(processed_video_path, 'rb') as f:
                    video_data = f.read()
                
                print(f"Uploading {processed_video_path} to {upload_url}")
                upload_response = requests.put(upload_url, data=video_data, headers=headers)

                if upload_response.status_code == 201:
                    public_url = f"https://zockto.b-cdn.net/videos/{unique_filename}"
                    print(f"Video uploaded successfully. Public URL: {public_url}")
                    return {
                        "output_video_url": public_url,
                        "message": "Video processed and uploaded successfully to BunnyCDN."
                    }
                else:
                    print(f"Failed to upload video to BunnyCDN. Status: {upload_response.status_code}, Details: {upload_response.text}")
                    return {
                        "error": "Failed to upload video to BunnyCDN.",
                        "status_code": upload_response.status_code,
                        "details": upload_response.text
                    }
            except Exception as upload_exc:
                print(f"Exception during BunnyCDN upload: {upload_exc}")
                return {"error": f"Exception during BunnyCDN upload: {upload_exc}"}
        else:
            print(f"Error: Processed video not found at {processed_video_path}")
            # List files in output_directory for debugging
            if os.path.exists(output_directory):
                print(f"Contents of {output_directory}: {os.listdir(output_directory)}")
            else:
                print(f"Output directory {output_directory} does not exist.")
            return {"error": f"Processed video file not found at {processed_video_path}."} # Added period for consistency

    except requests.exceptions.RequestException as e:
        print(f"Error downloading video or during BunnyCDN upload: {e}")
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
        # This path is defined based on where segment_video saves its output.
        # Ensure this variable is defined if it wasn't earlier in the try block (e.g. if input video download failed)
        # However, in current logic, processed_video_path is defined before try-except for upload.
        # This should be os.path.join("./", "foreground.mp4")
        processed_video_to_cleanup = os.path.join("./", "foreground.mp4") 
        if os.path.exists(processed_video_to_cleanup):
            try:
                os.remove(processed_video_to_cleanup)
                print(f"Cleaned up processed video file: {processed_video_to_cleanup}")
            except Exception as e:
                print(f"Error cleaning up processed video file {processed_video_to_cleanup}: {e}")


# Standard RunPod boilerplate
if __name__ == '__main__':
    print("Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
