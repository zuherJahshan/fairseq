import torch
import torch.nn as nn
from mamba_ssm import MambaLMHeadModel, MambaConfig
import glob
from tqdm import tqdm
import whisper
import torchaudio
import os
from transformers import GPT2Tokenizer
import csv
from fairseq.models.wav2vec import ConvFeatureExtractionModel

class Dataset(torch.utils.data.Dataset):
    def __init__(self, database_file, device, tokenizer = None):
        print("Loading dataset file...")
        # audiofiles = glob.glob(f'base_audio/**/*.flac', recursive=True)
        
        """
        the database_file is a csv file that contains two columns,
        the first column is the path to the audio file,
        and the second column is the path to the transcript file.
        Load this file as a list of tuples,
        where the first element of the tuple is the path to the audio file,
        and the second element is the path to the transcript file.
        """
        self.file_dataset = []
        with open(database_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.file_dataset.append((row[0], row[1]))

        self.device = device

        print("Loading tokenizer...")
        if tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=True)
        else:
            self.tokenizer = tokenizer
        print("Initialization done successfully")

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
    return audios, padded_tokens


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    dataset_file = "/workspace/fairseq/data/database.csv"
    
    dataset = Dataset(dataset_file, device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn, num_workers=8)

    for audios, tokens, lengths in tqdm(loader):
        # print(audios.shape, tokens.shape, lengths)
        pass


class mambaModel(nn.Module):
    def __init__(self, config=None):
        """
        input: config - a yaml file defining the configuration of the model
        output: None
        Description: This function initializes the model with the given configuration
        """
        self.cfg = config
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=eval(self.cfg.feature_enc_layers),
            dropout=0.0,
            mode=self.cfg.extractor_mode,
            conv_bias=self.cfg.conv_bias,
        )
        self.vocab = []
        with open(self.cfg.vocab_file, 'r') as fvocab:
            # Open as a csv file
            reader = csv.reader(fvocab)
            # all vocabulary could be found in the same row
            for row in reader:
                self.vocab = self.vocab + row
        
        mamba_cfg = MambaConfig()
        mamba_cfg.d_model = self.cfg.encoder_embed_dim
        mamba_cfg.vocab_size = len(self.vocab)
        mamba_cfg.n_layer = self.cfg.encoder_layers
        self.encoder = MambaLMHeadModel(mamba_cfg)

    def forward(self, audios, tokens):
        pass

if __name__ == '__main__':
    main()
