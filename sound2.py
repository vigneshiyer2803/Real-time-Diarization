import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
from sklearn.cluster import KMeans
import numpy as np
from pyannote.metrics.diarization import DiarizationErrorRate
import shutil
import os

# Define the cache directory for the dataset
cache_dir = os.path.expanduser("C:/Users/Dell/.cache/huggingface/datasets/librispeech_asr/clean/2.1.0/2712a8f82f0d20807a56faadcd08734f9bdd24c850bb118ba21ff33ebff0432f.incomplete\\librispeech_asr-train.100-00000-00002-of-NNNNN.arrow")

# Remove the directory containing incomplete files to resolve any file access issues
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir, ignore_errors=True)

# Load the pre-trained Wav2Vec2 model and processor
model_name = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name, output_hidden_states=True)

# Set the model to evaluation mode
model.eval()

# Load a small subset of the LibriSpeech dataset from Hugging Face
dataset = load_dataset("librispeech_asr", "clean", split="test[:0.1%]", cache_dir='D:\Python Programs\ReinforcementLearning', trust_remote_code=True)

# Simulate real-time audio streaming (process audio chunks)
def stream_audio_chunk(audio, chunk_size=16000):
    num_chunks = len(audio) // chunk_size
    
    for i in range(num_chunks):
        444444
        yield audio[i * chunk_size : (i + 1) * chunk_size]

# Basic VAD function to simulate real-time processing (this is a simplified approach)
def vad(audio_chunk, threshold=0.02):
    return torch.abs(torch.tensor(audio_chunk)).mean() > threshold

# Function to extract speaker embeddings (simplified version using Wav2Vec2 outputs)
def extract_speaker_embeddings(audio_chunk):
    with torch.no_grad():
        inputs = processor(audio_chunk, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.squeeze()
        if len(input_values.shape) == 1:
            input_values = input_values.unsqueeze(0)  # Ensure [batch_size, sequence_length]
        
        # Forward pass through the model, returning hidden states
        outputs = model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        embeddings = hidden_states[-1].mean(dim=1).squeeze().numpy()  # Using the last hidden layer
    return embeddings

# Function to perform clustering
def perform_clustering(embeddings_list, n_clusters=2):
    means = KMeans(n_clusters=n_clusters)
    means.fit(embeddings_list)
    return means.labels_

# Function to calculate Diarization Error Rate (DER)
def calculate_der(predictions, references):
    der = DiarizationErrorRate()
    return der(references, predictions)

# Transcribe real-time audio stream with speaker diarization
def transcribe_audio_stream_with_clustering(dataset):
    embeddings_list = []
    transcriptions = []
    references = []  # store the ground truth for DER calculation
    
    for example in dataset:
        audio = example['audio']['array']
        transcription = example['text']
        
        # Process each chunk of the audio
        for audio_chunk in stream_audio_chunk(audio):
            audio_chunk = torch.tensor(audio_chunk).unsqueeze(0)  # Add batch dimension
            
            if vad(audio_chunk):
                # Extract embeddings
                embeddings = extract_speaker_embeddings(audio_chunk)
                embeddings_list.append(embeddings)
                
                # Process the chunk for transcription
                inputs = processor(audio_chunk, sampling_rate=16000, return_tensors="pt", padding=True)
                input_values = inputs.input_values.squeeze()
                if len(input_values.shape) == 1:
                    input_values = input_values.unsqueeze(0)  # Ensure [batch_size, sequence_length]

                with torch.no_grad():
                    logits = model(input_values).logits
                
                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_transcription = processor.decode(predicted_ids[0])
                transcriptions.append(predicted_transcription)
                
                # Store the reference transcription (ground truth)
                references.append(transcription)

    # Perform clustering on collected embeddings
    if embeddings_list:
        labels = perform_clustering(np.array(embeddings_list), n_clusters=2)
        
        # Print transcription with speaker labels
        for idx, transcription in enumerate(transcriptions):
            print(f"Speaker {labels[idx] + 1}: {transcription}")
    
    # Calculate DER based on predictions and ground truth
    der = calculate_der(predictions=transcriptions, references=references)
    print(f"Diarization Error Rate (DER): {der}")

# Example usage with the small dataset from Hugging Face
transcribe_audio_stream_with_clustering(dataset)
