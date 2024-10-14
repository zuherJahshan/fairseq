import whisper
import torch
import pandas as pd
import whisper
import torchaudio
from tqdm import tqdm
import glob
import torch.nn.functional as F
import os
from torchaudio.transforms import Resample

# Build a dataset which will load all the flac audiofiles
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TqdmWithEMATime(tqdm):
    def __init__(self, *args, ema_alpha=0.1, **kwargs):
        """
        A custom tqdm class that uses an exponentially weighted moving average
        for time estimation.
        
        Args:
            ema_alpha (float): The smoothing factor for EMA. 0 < ema_alpha <= 1.
                              Higher values give more weight to recent steps.
        """
        super().__init__(*args, **kwargs)
        self.ema_alpha = ema_alpha
        self.ema_time = None

    def update(self, n=1):
        """
        Update the EMA-based time per iteration and adjust the tqdm display.
        
        Args:
            n (int): The number of iterations to update by (default is 1).
        """
        if self.ema_time is None:
            self.ema_time = self.last_print_t - self.start_t
        else:
            step_time = (self.last_print_t - self.start_t) / self.n
            self.ema_time = (self.ema_alpha * step_time) + ((1 - self.ema_alpha) * self.ema_time)
        
        remaining_steps = self.total - self.n
        if remaining_steps > 0:
            eta = self.ema_time * remaining_steps
            self.set_postfix(ETA=f'{eta:.2f}s')
        
        super().update(n)

class audioDataset(torch.utils.data.Dataset):
    def __init__(self, audiofiles, device):
        self.audiofiles = audiofiles
        self.device = device
        self.target_sample_rate = 16_000

    def __len__(self):
        return len(self.audiofiles)

    def __getitem__(self, idx):
        audiofile = self.audiofiles[idx]
        audio, sample_rate = torchaudio.load(audiofile)
        
        # Resample the audio if necessary
        if sample_rate != self.target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            audio = resampler(audio)

        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        return audiofile, mel

"""
def pad_collate_fn(batch):
    max_time = torch.max(torch.tensor([mel.shape[-1] for mel in batch]))
    padded_batch = torch.concatenate(
        [F.pad(mel, (0, max_time - mel.shape[-1])) for mel in batch],
        dim=0,
    )
    return padded_batch
"""

def get_transcribed_file(audiofile):
    return audiofile.replace('.flac', '.txt').replace('train_val', 'results')

print("Loading all audiofiles...")
all_audiofiles = glob.glob('/workspace/fairseq/data/train_val/**/*.flac', recursive=True)
all_resultfiles = glob.glob('/workspace/fairseq/data/results/**/*.txt', recursive=True)

print("Searching for audiofiles not transcribed yet...")
audiofiles = []
transcribed_files = 0
for audiofile in tqdm(all_audiofiles):
    resultfile = get_transcribed_file(audiofile)
    if not os.path.exists(resultfile):
        audiofiles.append(audiofile)
    else:
        transcribed_files = transcribed_files + 1

print(f"Found {transcribed_files} transcribed audiofiles, files yet to be transcribed: {len(audiofiles) - transcribed_files}")

dataset = audioDataset(audiofiles, device)
loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)


model_name = 'medium.en'
print("Loading the model...")
model = whisper.load_model(model_name, device=device)
print(f"Model {model_name} successfully loaded.")

print("Loading decoding options...")
options = whisper.DecodingOptions(language="en", without_timestamps=True)

for audiofiles, mels in tqdm(loader):
    try:
        results = model.decode(mels, options)
        for (audiofile, result) in zip(audiofiles, results):
            # the name of the file is dir1/dir2/.../file.flac.
            # Write the result to results/dir1/dir2/.../file.txt
            result_file = get_transcribed_file(audiofile)
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            with open(result_file, 'w') as f:
                f.write(result.text)
    except Exception as e:
        print(f"Warning, could not transcribe batch: {e}")
