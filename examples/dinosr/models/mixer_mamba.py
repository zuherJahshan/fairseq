
from multiprocessing import set_start_method
# Set the start method to spawn
try:
    set_start_method('spawn')
except RuntimeError:
    pass
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import MambaLMHeadModel, MambaConfig
import glob
from tqdm import tqdm
import whisper
import torchaudio
import os
from transformers import GPT2Tokenizer
import csv
from fairseq.models.wav2vec import ConvFeatureExtractionModel
import yaml
from dataclasses import (
    dataclass,
    field
)
from einops import rearrange
from enum import Enum


@dataclass
class SpecialTokens(str, Enum):
    SOS = "<SOS>"
    EOS = "<EOS>"
    BLANK = "<BLANK>"


class CharTokenizer:
    def __init__(self, vocab_file, device):
        # Load vocab from the CSV file and add special tokens
        chars = []
        with open(vocab_file, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                chars = chars + row
        
        # Get rid of redundant character, and sort the characters
        chars = list(set(chars))
        chars.sort()


        # Merge vocab with special tokens and prepare reverse vocab
        self.vocab = {
            SpecialTokens.SOS.value: 0,
            SpecialTokens.BLANK.value: len(chars) + 1,
            SpecialTokens.EOS.value: len(chars) + 2
        }
        for idx, char in enumerate(chars):
            self.vocab[char] = idx + 1

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        self.device = device


    def encode(self, t: str):
        """
        Encodes a string of characters into a one-hot encoded tensor.
        Args:
            t (str): string
        Returns:
            Tensor of shape [str_len, vocab_size]
        """
        vocab_size = self.size()
        
        indices = torch.tensor([self.vocab[SpecialTokens.SOS.value]] + [self.vocab[char] for char in t] + [self.vocab[SpecialTokens.EOS.value]], device=self.device)

        # Convert to one-hot and return
        #return F.one_hot(indices, num_classes=vocab_size)
        return indices


    def decode(self, t):
        raise NotImplementedError


    def size(self):
        """
        Returns the size of the vocabulary (including <SOS> and <EOS> tokens).
        """
        return len(self.vocab)


    def get_blank_index(self):
        return self.vocab[SpecialTokens.BLANK.value]


    def get_sos_index(self):
        return self.vocab[SpecialTokens.SOS.value]


    def get_eos_index(self):
        return self.vocab[SpecialTokens.EOS.value]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, database_file, vocab_file, device, **kwargs):
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
        self.tokenizer = CharTokenizer(vocab_file, device)
        print("Initialization done successfully")

        self.target_sample_rate = 16_000


    def __len__(self):
        return len(self.file_dataset)


    def __getitem__(self, idx):
        audiofile, transcript_file = self.file_dataset[idx]
        
        # Load the audiofile
        audio, sample_rate = torchaudio.load(audiofile)

        # Resample the audio if necessary
        if sample_rate != self.target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            audio = resampler(audio)

        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)

        # Load the transcript file and tokenize it.
        with open(transcript_file, 'r') as f:
            transcript = f.read()
        tokens = self.tokenizer.encode(transcript)
        return audio, tokens, tokens.shape[0] + 1 # the +1 is to account for the <EOS> token


    def get_vocab_size(self):
        return self.tokenizer.size()


    def get_blank_index(self):
        return self.tokenizer.get_blank_index()


    def get_sos_index(self):
        return self.tokenizer.get_sos_index()


    def get_eos_index(self):
        return self.tokenizer.get_eos_index()


def collate_fn(batch):
    audios, tokens, lengths = zip(*batch)
    max_length = max(lengths)

    # the tokens is a list of tensors of shape [T, vocab_size] where the last dimension is always the same for all
    # tensors and is equal to vocab_size. And the [0] dimension is the length of the transcript.
    # Pad the tokens to have the same length and stack them to form [B,T, vocab_size]
    #padded_tokens = torch.stack([F.pad(token, (0, 0, 0, max_length - token.size(0))) for token in tokens])

    padded_tokens = []
    for token in tokens:
        padding = torch.ones((max_length - token.shape[0]), device=token.device) * 482 # TODO:  should be changed
        padded_tokens.append(torch.cat((token, padding), dim=0))

    padded_tokens = torch.stack(padded_tokens)


    token_lens = torch.tensor(lengths, device=padded_tokens.device, dtype=torch.long)

    # audios have a shape of (audio_length), we need to build a tensor of shape (batch_size, audio_length)
    audios = torch.stack(audios)
    return audios, padded_tokens, token_lens


@dataclass
class MixerMambaConfig:
    conv_feature_enc: str = field(default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]")
    extractor_mode: str = field(default="layer_norm")
    conv_bias: bool = field(default=False)
    encoder_embed_dim: int = field(default=768)
    vocab_file: str = field(default="/workspace/fairseq/data/vocab.csv")
    fe_dropout: float = field(default=0.0)
    encoder_layers: int = field(default=12)


class mambaModel(nn.Module):

    def __init__(self, config, vocab_size):
        """
        input: config - a yaml file defining the configuration of the model
        output: None
        Description: This function initializes the model with the given configuration
        """
        super(mambaModel, self).__init__()
        self.cfg = config
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=eval(self.cfg.conv_feature_enc),
            dropout=self.cfg.fe_dropout,
            mode=self.cfg.extractor_mode,
            conv_bias=self.cfg.conv_bias,
        )

        mamba_cfg = MambaConfig()
        mamba_cfg.d_model = self.cfg.encoder_embed_dim
        mamba_cfg.vocab_size = vocab_size
        mamba_cfg.n_layer = self.cfg.encoder_layers
        self.encoder = MambaLMHeadModel(mamba_cfg)

    def forward(self, audios):
        """
            input - audios: a tensor of shape (batch_size, audio_length)
            tokens: a tensor of shape (batch_size, transcript_length)
        """
        # Apply feature extraction as defined by fairseq (will output [B,D,T]) and reaganize it to [B,T,D]
        features = self.feature_extractor(audios)
        features = rearrange(features, 'b d t -> b t d')

        # features have a shape of (batch_size, feature_length, feature_dim)
        return self.encoder(features)


def main() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_file = "/workspace/fairseq/data/small_database.csv"

    # read the yaml file inside ./../config/s2t.yaml
    cfg = yaml.safe_load(open("./../config/s2t.yaml"))

    dataset = Dataset(**cfg["dataset"], device=device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["dataset"]["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=cfg["dataset"]["num_workers"])

    mixer_mamba_cfg = MixerMambaConfig(**cfg["model"])

    model = mambaModel(mixer_mamba_cfg, dataset.get_vocab_size()).to(device)

    optimizer = getattr(torch.optim, cfg["optimizer"]["name"])(model.parameters(), **cfg["optimizer"]["params"])
    criterion = getattr(nn, cfg["criterion"]["name"])(**cfg["criterion"]["params"])

    for epoch in range(10):
        print(f"Epoch {epoch}")
        epoch_loss = 0

        progress_bar = tqdm(loader)

        for step, (audios, tokens, token_lens) in enumerate(progress_bar):
            # the model output have no softmax applied to it.
            logits, _ = model(audios)
            log_probs = F.log_softmax(logits, dim=-1)

            # Calculate loss
            input_lens = torch.full((log_probs.shape[0],), log_probs.shape[1], device=log_probs.device, dtype=torch.long)
            log_probs = rearrange(log_probs, 'b t v -> t b v')
            loss = criterion(log_probs, tokens, input_lens, token_lens)

            epoch_loss += loss.item()

            # Zero out the gradient state of the model parameters.
            optimizer.zero_grad()

            # Perform a backprop to calculate the new gradients
            loss.backward()

            # Perform an optimizer step
            optimizer.step()

            progress_bar.set_postfix(loss=(epoch_loss / (step + 1)))


if __name__ == '__main__':
    main()
