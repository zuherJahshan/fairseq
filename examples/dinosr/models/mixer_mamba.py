import torch
from mamba_ssm import MambaLMHeadModel, MambaConfig
import glob
from tqdm import tqdm
import whisper
import torchaudio
import os
from transformers import GPT2Tokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, base_audio, base_transcripts, device, tokenizer = None):
        print("Loading dataset filepaths...")
        # audiofiles = glob.glob(f'base_audio/**/*.flac', recursive=True)
        audiofiles = glob.glob(f'{base_audio}/*.flac', recursive=True)
        self.device = device

        self.file_dataset = []

        # Check that for every audio file there is a corresponding transcript file
        for audiofile in tqdm(audiofiles):
            transcript_file = audiofile.replace(base_audio, base_transcripts).replace('.flac', '.txt')
            if not os.path.exists(transcript_file):
                raise ValueError(f"Transcript file for {audiofile} not found, in {transcript_file}")
            else:
                self.file_dataset.append((audiofile, transcript_file))
        print("Dataset filepaths loaded successfully")

        print("Loading tokenizer")
        if tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
        else:
            self.tokenizer = tokenizer
        print("Tokenizer loaded successfully")

    def __len__(self):
        return len(self.file_dataset)

    def __getitem__(self, idx):
        audiofile, transcript_file = self.file_dataset[idx]
        
        # Load the audiofile
        audio, sample_rate = torchaudio.load(audiofile)
        assert sample_rate == 16_000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)

        # Load the transcript file and tokenize it.
        with open(transcript_file, 'r') as f:
            transcript = f.read()
        tokens = self.tokenizer.encode(transcript, return_tensors="pt").to(self.device)
        return audio, tokens.flatten(), tokens.size()[-1] + 1 # the +1 is to account for the <EOS> token


def collate_fn(batch):
    audios, tokens, lengths = zip(*batch)
    max_length = max(lengths)
    # pad with <EOS> token (GPT2 uses the last token as the EOS token - which is 50256)
    padded_tokens = torch.stack(
        [torch.cat([tokens[i], torch.tensor([50256] * (max_length - lengths[i]))]) for i in range(len(tokens))]
    )
    # audios have a shape of (audio_length), we need to build a tensor of shape (batch_size, audio_length)
    audios = torch.stack(audios)
    return audios, padded_tokens, lengths


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    base_audio = '/workspace/fairseq/data/train_val/unlab_60k/large/17/patient_spider_64kb_mp3/patient_spider_whitman_gm_64kb/'
    trainscript_base = '/workspace/fairseq/data/results/unlab_60k/large/17/patient_spider_64kb_mp3/patient_spider_whitman_gm_64kb/'
    
    dataset = Dataset(base_audio, trainscript_base, device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    for audios, tokens, lengths in tqdm(loader):
        # print(audios.shape, tokens.shape, lengths)
        print(tokens)


if __name__ == '__main__':
    main()


