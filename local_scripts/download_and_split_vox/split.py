import glob
import json
import os
import torch
import torchaudio
import progressbar

def read_audio(path):
    """Reads an audio file and returns the waveform and sample rate."""
    waveform, sample_rate = torchaudio.load(path)
    if waveform.size(0) != 1:
        # Convert to mono by averaging channels
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate not in (8000, 16000, 32000, 48000):
        # Resample to 16kHz
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    return waveform, sample_rate

def write_audio(path, waveform, sample_rate):
    """Writes a waveform to an audio file in FLAC format."""
    # Normalize the waveform to (-1.0, 1.0)
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    torchaudio.save(path, waveform, sample_rate, format='flac')

def trim_waveform(waveform, sample_rate, start, end):
    """Trims a waveform to a specified time interval."""
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    return waveform[:, start_sample:end_sample]

def get_flac_path_from_json(json_path):
    """Returns the path to the FLAC file corresponding to a JSON file."""
    return os.path.splitext(json_path)[0] + '.flac'

def get_vac(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        return data['voice_activity']

def get_new_path(root_dir, json_file, filename):
    new_dir = os.path.join(root_dir, os.path.splitext(json_file)[0])
    os.makedirs(new_dir, exist_ok=True)
    new_path = os.path.join(new_dir, filename)
    return new_path

def main():
    json_files = glob.glob("unlab_60k/**/*.json", recursive=True)
    new_root_dir = "new_out"
    bar = progressbar.ProgressBar(maxval=len(json_files))
    bar.start()
    for i, j_file in enumerate(json_files):
        waveform, sample_rate = read_audio(get_flac_path_from_json(j_file))
        vac = get_vac(j_file)
        for j, segment in enumerate(vac):
            start = segment[0]
            end = segment[1]
            trimmed_waveform = trim_waveform(waveform, sample_rate, start, end)
            new_path = get_new_path(new_root_dir, j_file, f"segment_{j}.flac")
            write_audio(new_path, trimmed_waveform, sample_rate)
        bar.update(i + 1)
    bar.finish()

if __name__ == '__main__':
    main()