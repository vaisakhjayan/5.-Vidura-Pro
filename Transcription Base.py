import json
import os
import whisper
from pathlib import Path

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def transcribe_audio():
    # Load the selected video information
    with open('JSON Files/1. Selected Video.json', 'r') as f:
        video_info = json.load(f)
    
    # Get the title which is the filename
    audio_filename = f"{video_info['title']}.wav"
    
    # Construct the full path to the audio file
    audio_path = os.path.join('/Users/superman/Desktop/Celebrity Voice Overs', audio_filename)
    
    # Check if the audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return
    
    # Load the Whisper model (small version)
    print("Loading Whisper model...")
    model = whisper.load_model("small")
    
    # Transcribe the audio with word-level timestamps
    print(f"Transcribing {audio_filename}...")
    result = model.transcribe(audio_path, word_timestamps=True)
    
    # Process segments and format with timestamps
    formatted_transcription = []
    
    for segment in result["segments"]:
        start_time = format_timestamp(segment["start"])
        text = segment["text"].strip()
        
        # Add timestamp and text
        formatted_segment = f"[{start_time}] {text}\n"
        formatted_transcription.append(formatted_segment)
        
        # Add a blank line between segments for readability
        if len(text) > 150:  # Add paragraph break for longer segments
            formatted_transcription.append("\n")
    
    # Write the formatted transcription to the output file
    with open('Transcription Base.txt', 'w', encoding='utf-8') as f:
        f.writelines(formatted_transcription)
    
    print("Transcription completed and saved to Transcription Base.txt")

if __name__ == "__main__":
    transcribe_audio()
