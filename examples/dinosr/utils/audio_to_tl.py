import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt


def cumulative_speech_time_vector(audio_path, device='cuda'):
    """
    Computes a vector where each element t represents the cumulative active speech time up to sample t.
    
    Args:
        audio_path (str): Path to the input audio file.
        device (str): 'cuda' for GPU or 'cpu' for CPU.
    
    Returns:
        cumulative_speech_time (Tensor): A tensor of shape (T,) with cumulative speech time in seconds.
    """
    # Load the Silero VAD model
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False,
                                  verbose=False)
    (get_speech_timestamps, _, _, _, _) = utils

    # Move model to the desired device
    model.to(device)
    model.eval()

    # Load the audio file
    wav, sample_rate = torchaudio.load(audio_path)
    wav = wav.to(device)

    # If stereo, convert to mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Resample if sample rate is not 16 kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
        wav = resampler(wav)
        sample_rate = 16000

    # Flatten the waveform
    wav = wav.squeeze(0)

    # Apply VAD to get speech timestamps
    with torch.no_grad():
        speech_timestamps = get_speech_timestamps(wav, model,
                                                  sampling_rate=sample_rate,
                                                  return_seconds=False,
                                                  threshold=0.75)

    # Initialize a binary mask for speech (1) and silence (0)
    T = wav.shape[0]
    speech_mask = torch.zeros(T, dtype=torch.bool, device=device)

    # Mark speech segments in the mask
    for segment in speech_timestamps:
        start_idx = segment['start']
        end_idx = segment['end']
        speech_mask[start_idx:end_idx] = True

    # Compute the cumulative sum over the speech mask
    cumulative_speech_samples = torch.cumsum(speech_mask.int(), dim=0)

    # Convert cumulative speech samples to time (in seconds)
    cumulative_speech_time = cumulative_speech_samples / sample_rate

    return cumulative_speech_time.cpu()  # Move to CPU if needed


def calculate_speech_silence_durations(audio_path, device='cuda'):
    """
    Calculates the durations of speech and silence in the audio file using Silero VAD.
    
    Args:
        audio_path (str): Path to the input audio file.
        device (str): 'cuda' for GPU or 'cpu' for CPU.
    
    Returns:
        total_speech_duration (float): Total duration of detected speech in seconds.
        total_silence_duration (float): Total duration of silence in seconds.
    """
    # Load the Silero VAD model
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=False,
                                  onnx=False,
                                  verbose=False)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Move model to the desired device
    model.to(device)
    model.eval()

    # Load the audio file
    # read_audio returns a torch.Tensor
    wav, sample_rate = torchaudio.load(audio_path)
    wav = wav.to(device)

    # If stereo, convert to mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Resample if sample rate is not 16 kHz
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
        wav = resampler(wav)
        sample_rate = 16000

    # Apply VAD to get speech timestamps
    with torch.no_grad():
        speech_timestamps = get_speech_timestamps(wav.squeeze(0), model,
                                                  sampling_rate=sample_rate,
                                                  return_seconds=True, 
                                                  threshold=0.75)

    # Calculate total durations
    total_audio_duration = wav.shape[1] / sample_rate
    total_speech_duration = sum([t['end'] - t['start'] for t in speech_timestamps])
    total_silence_duration = total_audio_duration - total_speech_duration


    return total_audio_duration, total_speech_duration, total_silence_duration


if __name__ == '__main__':
    import os
    import csv
    import tqdm

    device = "cuda"
    datapath = "/workspace/fairseq/data/database.csv"

    if not torch.cuda.is_available() and device == 'cuda':
        print("CUDA is not available. Using CPU instead.")
        device = 'cpu'

    x = []
    y = []
    total_audio = 0
    total_speech = 0
    total_chars = 0
    with open(datapath, "r") as f:
        reader = csv.reader(f)
        idx = 0

        progress_bar = tqdm.tqdm(reader)

        for flac_file, txt_file in progress_bar:
            idx += 1

            audio_len, speech_len, silence_len = calculate_speech_silence_durations(flac_file, device=device)

            with open(txt_file, "r") as f:
                txt = f.read()
                txt = txt.replace("\n", "")
                txt_len = len(txt.split())
            if audio_len > 30:
                continue
            total_audio += audio_len
            total_speech += speech_len
            total_chars += txt_len
            progress_bar.set_postfix({"speech_factor": total_chars / total_speech, "audio_factor": total_chars / total_audio})
