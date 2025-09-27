from moviepy.editor import VideoFileClip
import librosa
import pandas as pd
import numpy as np
import os
import pickle
import cv2
import subprocess

video_path = ".mov"
Segment_output_path = "/content/video_segments"
audio_output_path = "audio.wav"
model_path = "rf_best_model.pkl"

# Segment video based on the silence intervals
def extract_silent_timestamps(input_video_path: str, audio_output_path: str, top_db: int = 20 ):
    # Step 1: Extract audio from video using moviepy
    clip = VideoFileClip(input_video_path)
    audio = clip.audio
    audio.write_audiofile(audio_output_path)

    # Step 2: Load audio and detect non-silent intervals using librosa
    y, sr = librosa.load(audio_output_path, sr=None)
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)

    # Convert non-silent intervals to seconds
    non_silent_intervals_sec = [(float(start / sr), float(end / sr)) for start, end in non_silent_intervals]

    return non_silent_intervals_sec


def split_video(input_file: str, output_prefix: str, output_path: str, intervals: list):
    with VideoFileClip(input_file) as video:
        for index, (start, end) in enumerate(intervals):
            # Calculate the duration for the current segment
            duration = end - start

            # Subclip the video
            subclip = video.subclip(start, end)

            # Construct the output filename
            output_file = f"{output_prefix}_part{index+1}.mov"

            # Set the output path
            set_output_path = os.path.join(output_path, output_file)

            # Write the subclip to the file
            subclip.write_videofile(set_output_path, codec='libx264')

# Extract audio from the segments, featurize, and get prediction

# Load the video file
def extract_audio(video_path):
  video = VideoFileClip(video_path)

  # Extract and save the audio
  audio = video.audio
  audio.write_audiofile("audio1.wav")

#function to extract features from audion files
def extract_features(audio_path):
    # load the audio file
    y,sr = librosa.load(audio_path,mono=True) #load the audio file
    # extract features
    rmse = librosa.feature.rms(y=y)[0] #compute root-mean-square (RMS) value for each frame
    spec_cent = librosa.feature.spectral_centroid(y=y,sr=sr) #calculate spectral_centroid
    spec_bw = librosa.feature.spectral_bandwidth(y=y,sr=sr) #calculate spectral_bandwith
    rolloff = librosa.feature.spectral_rolloff(y=y,sr=sr) #calculate spectral_rolloff
    zcr = librosa.feature.zero_crossing_rate(y) #calculate zero crossing rate
    mfcc = librosa.feature.mfcc(y=y,sr=sr) #Mel frequency ceptral coefficients (mfcc)


    audio_dict={
        'RMSE':rmse.mean(),
        'SPECTRAL_CENTROID':spec_cent.mean(),
        'SPECTRAL_BANDWIDTH':spec_bw.mean(),
        'ROLLOFF':rolloff.mean(),
        'ZERO_CROSSING_RATE':zcr.mean()
    }

    #add the mfcc values
    for index,mfcc_set in enumerate(mfcc):
        feature_name=f"MFCC_FEATURE_{index}"
        audio_dict[feature_name]=np.mean(mfcc_set)

    return audio_dict

non_silent_intervals = extract_silent_timestamps(video_path, audio_output_path)
video_name = video_path.split("/")[-1].split(".")[0]
split_video(video_path, video_name, Segment_output_path, non_silent_intervals)

# Removing Noises
Confidence_threshold = 0.7

with open(model_path, 'rb') as file:
    model = pickle.load(file)

segments = os.listdir(Segment_output_path)
predictions = []
for segment in segments:
  segment_path = os.path.join(Segment_output_path, segment)
  extract_audio(segment_path)
  audio_dict = extract_features("audio1.wav")
  input = pd.DataFrame([audio_dict])
  prediction = model.predict_proba(input)
  if max(prediction[0]) > Confidence_threshold:
    predictions.append(np.argmax(prediction[0]))
    print(prediction)

# Record the predictions with their intervals
df = pd.DataFrame({
    'start_time': [interval[0] for interval in non_silent_intervals],
    'end_time': [interval[1] for interval in non_silent_intervals],
    'prediction': predictions
})
df.to_csv("predictions.csv", index = False)

# Overlay the prediction on the video

# Load the CSV with float seconds directly
df = pd.read_csv('predictions.csv')  # Columns: start_time, end_time, prediction

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# Process frames
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_sec = frame_idx / fps

    # Get predictions active at this time
    active_predictions = df[(df['start_time'] <= current_sec) & (df['end_time'] >= current_sec)]

    # Overlay text
    y_offset = 50
    for _, row in active_predictions.iterrows():
        text = f"Prediction: {row['prediction']}"
        cv2.putText(frame, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        y_offset += 40

    out.write(frame)
    frame_idx += 1

# Clean up
cap.release()
out.release()
print("✅ Video processing complete. Output saved as output_video.mp4")

# !ffmpeg -i output_video.mp4 -i "/content/drive/MyDrive/1:1_Chinee_Bernabe/Kulintang/Dataset/Kulintang St Scholastica College/IMG_0010.MOV" -c:v copy -map 0:v:0 -map 1:a:0 -shortest final_output_with_audio.mp4
cmd = [
    "ffmpeg",
    "-i", "output_video.mp4",
    "-i", "/content/drive/MyDrive/1:1_Chinee_Bernabe/Kulintang/Dataset/Kulintang St Scholastica College/IMG_0010.MOV",
    "-c:v", "copy",
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-shortest",
    "final_output_with_audio.mp4"
]

# Run command
result = subprocess.run(cmd, capture_output=True, text=True)