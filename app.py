from moviepy.editor import VideoFileClip
import librosa
import pandas as pd
import numpy as np
import os
import pickle
import cv2
import subprocess
import streamlit as st
from tempfile import NamedTemporaryFile

Segment_output_path = "video_segments"
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

def main():
    st.title("Kulintang Music Transcription")
    st.write("Upload a video to transcribe and overlay detections.")

    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov"])
    if uploaded_file is not None:
        with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name

        st.info("Processing video...")
        # Run the main pipeline
        try:
            # Use temp_video_path instead of video_path
            non_silent_intervals = extract_silent_timestamps(temp_video_path, audio_output_path)
            video_name = temp_video_path.split("/")[-1].split(".")[0]
            split_video(temp_video_path, video_name, Segment_output_path, non_silent_intervals)

            with open(model_path, 'rb') as file:
                model = pickle.load(file)

            segments = os.listdir(Segment_output_path)
            predictions = []
            Confidence_threshold = 0.7
            for segment in segments:
                segment_path = os.path.join(Segment_output_path, segment)
                extract_audio(segment_path)
                audio_dict = extract_features("audio1.wav")
                input = pd.DataFrame([audio_dict])
                prediction = model.predict_proba(input)
                if max(prediction[0]) > Confidence_threshold:
                    predictions.append(np.argmax(prediction[0]))

            df = pd.DataFrame({
                'start_time': [interval[0] for interval in non_silent_intervals],
                'end_time': [interval[1] for interval in non_silent_intervals],
                'prediction': predictions
            })
            df.to_csv("predictions.csv", index = False)

            # Overlay predictions on video
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                st.error(f"Cannot open video file: {temp_video_path}")
                return
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output_video:
                output_video_path = temp_output_video.name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_sec = frame_idx / fps
                active_predictions = df[(df['start_time'] <= current_sec) & (df['end_time'] >= current_sec)]
                y_offset = 50
                for _, row in active_predictions.iterrows():
                    text = f"Prediction: {row['prediction']}"
                    cv2.putText(frame, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)
                    y_offset += 40
                out.write(frame)
                frame_idx += 1
            cap.release()
            out.release()
            st.success("Video processing complete.")
            st.header("Output Preview")
            st.video(output_video_path)
            st.write("You can preview the processed video above. If satisfied, download it below:")
            with open(output_video_path, "rb") as f:
                st.download_button(
                    label="Download processed video",
                    data=f,
                    file_name="output_with_detections.mp4",
                    mime="video/mp4"
                )
        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()