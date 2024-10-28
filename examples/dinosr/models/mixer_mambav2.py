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
from tqdm import tqdm
import whisper
import torchaudio
import csv
from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models.wav2vec import ConvFeatureExtractionModel
import yaml
from dataclasses import (
    dataclass,
    field
)
from einops import rearrange
from enum import Enum
import unicodedata
import string
import editdistance
from abc import ABC, abstractmethod
import os

class ModelMetric(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def update(self, step, loss, logits, labels, **kwargs):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_value(self):
        pass

    @abstractmethod
    def new_val_better(self, old_val, new_val):
        pass


def remove_punctuation(text):
    """
    Removes all punctuation from the text, including Unicode punctuation.
    """
    # Use a list comprehension to filter out punctuation characters
    return ''.join(
        ch for ch in text if ch not in string.punctuation and unicodedata.category(ch)[0] != 'P'
    )


def calculate_wer(references, queries):
    total_wer = 0
    for reference, query in zip(references, queries):
        # Remove punctuation from both reference and query
        reference = remove_punctuation(reference)
        query = remove_punctuation(query)

        # Convert to lowercase
        reference = reference.lower()
        query = query.lower()

        # Tokenize the strings into words based on spaces
        ref_words = reference.strip().split()
        hyp_words = query.strip().split()

        # Compute the edit distance between the word lists
        distance = editdistance.eval(hyp_words, ref_words)

        # Calculate WER, and add it to the total_wer
        wer = distance / len(ref_words) if len(ref_words) > 0 else float('inf')
        wer = min(1.0, wer)
        total_wer += wer

        # Raise an exception if the length of the ref_words is zero
        if not ref_words:
            print("Reference words are empty")

    return total_wer / len(references)


class LossMetric(ModelMetric):
    def __init__(self, name="loss", memory=100):
        # Call base class init
        super().__init__()
        self.name = name
        self.memory = memory
        self.loss = None


    def update(self, step, loss, logits, labels, **kwargs):
        if  self.loss:
            self.loss = loss * (1 / self.memory) + (1 - 1 / self.memory) * self.loss
        else:
            self.loss = loss
        return self.loss


    def get_name(self):
        return self.name

    def get_value(self):
        return self.loss

    def new_val_better(self, old_val, new_val):
        return new_val < old_val


class WERMetric(ModelMetric):
    def __init__(self, tokenizer, name="wer", memory=100):
        super().__init__()
        self.name = name
        self.memory = memory
        self.wer = None
        self.tokenizer = tokenizer


    def update(self, step, loss, logits, labels, **kwargs):

        ref = labels
        hyp = self.tokenizer.decode(logits, from_logits=True, ctc=True)
        wer = calculate_wer(ref, hyp)

        if self.wer:
            self.wer = wer * (1 / self.memory) + (1 - 1 / self.memory) * self.wer
        else:
            self.wer = wer
        return self.wer


    def get_name(self):
        return self.name


    def get_value(self):
        return self.wer


    def new_val_better(self, old_val, new_val):
        return new_val < old_val


class MetricRecorder:
    def __init__(self):
        self.metrics = []
        self.step = -1

    def register_metric(self, metric: ModelMetric):
        self.metrics.append(metric)

    def update(self, step, loss, preds, labels):
        results = {}
        for metric in self.metrics:
            value = metric.update(step, loss, preds, labels)
            results[metric.get_name()] = value
        return results


class Checkpointer:
    def __init__(self, metric: ModelMetric, path: str):
        self.path = path
        self.metric = metric
        self.best_val = None

    def try_save():
        pass


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

        indices = torch.tensor([self.vocab[SpecialTokens.SOS.value]] + [self.vocab.get(char, self.vocab[SpecialTokens.BLANK.value]) for char in t] + [self.vocab[SpecialTokens.EOS.value]], device=self.device)

        # Convert to one-hot and return
        #return F.one_hot(indices, num_classes=vocab_size)
        return indices


    def decode(self, t: torch.Tensor, from_logits: bool = False, ctc: bool = False):
        """
        Decodes a batch of sequences of indices into a list of strings,
        applying CTC decoding rules (removing blanks and merging repeating characters).

        Args:
            t (Tensor): Tensor of shape [B, T], where each element is an index.

        Returns:
            List[str]: A list of decoded strings of length B.
        """
        # Map indices to characters
        index_to_char = self.reverse_vocab

        blank_index = self.get_blank_index()
        eos_index = self.get_eos_index()
        sos_index = self.get_sos_index()

        batch_size = t.size(0)
        decoded_strings = []

        if from_logits:
            t = torch.argmax(t, dim=-1)

        for b in range(batch_size):
            sequence = t[b]
            decoded_str = []
            previous_char = None

            for idx in sequence:
                idx = idx.item()  # Get the integer value from the tensor element

                char = index_to_char.get(idx, '')

                # Stop decoding if EOS token is reached
                if idx == eos_index:
                    break

                # Skip blanks and SOS tokens
                if idx == blank_index or idx == sos_index:
                    previous_char = char
                    continue

                # Skip if the same as previous character, and this is ctc
                if char == previous_char and ctc:
                    continue

                decoded_str.append(char)
                previous_char = char

            decoded_strings.append(''.join(decoded_str))

        return decoded_strings


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

        return audio, tokens, tokens.shape[0], transcript


    def get_tokenizer(self):
        return self.tokenizer


    def get_vocab_size(self):
        return self.tokenizer.size()


    def get_blank_index(self):
        return self.tokenizer.get_blank_index()


    def get_sos_index(self):
        return self.tokenizer.get_sos_index()


    def get_eos_index(self):
        return self.tokenizer.get_eos_index()


def collate_fn(batch):
    audios, tokens, lengths, transcripts = zip(*batch)

    # collate only audios, tokens and lengths that have length > 0
    audios = [audio for idx, audio in enumerate(audios) if lengths[idx] > 0]
    tokens = [token for idx, token in enumerate(tokens) if lengths[idx] > 0]
    lengths = [length for length in lengths if length > 0]

    # Concatenate the tokens and create target lengths
    targets = torch.cat(tokens)
    target_lens = torch.tensor(lengths, dtype=torch.long, device=targets.device)

    # audios have a shape of (audio_length), we need to build a tensor of shape (batch_size, audio_length)
    audios = torch.stack(audios)
    return audios, targets, target_lens, transcripts


@dataclass
class MixerMambaConfig:
    conv_feature_enc: str = field(default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]")
    extractor_mode: str = field(default="layer_norm")
    conv_bias: bool = field(default=False)
    encoder_embed_dim: int = field(default=768)
    vocab_file: str = field(default="/workspace/fairseq/data/vocab.csv")
    fe_dropout: float = field(default=0.0)
    encoder_layers: int = field(default=12)
    mask_prob: int = field(default=0.5)
    mask_length: int = field(default=10)
    top_k_layers: int = field(default=8)
    codebook_size: int = field(default=256)
    n_codebooks: int = field(default=8)
    codebook_init_decay: float = field(default=0.9) # TODO: set it up correctly
    ema_decay: float = field(default=0.999) # TODO: set it up correctly


class DinoSRModel(nn.Module):
    def __init__(self, cfg: MixerMambaConfig, student: nn.Module):
        super(DinoSRModel, self).__init__()
        self.cfg = cfg

        self.student = student

        self.mask_prob = cfg.mask_prob
        self.mask_length = cfg.mask_length
        self.mask_emb = nn.Parameter(torch.randn(cfg.encoder_embed_dim))

        self.codebook_size = cfg.codebook_size
        self.n_codebooks = cfg.n_codebooks
        self.codebook_decay = cfg.codebook_init_decay
        self.heads = torch.nn.ModuleList([
            nn.Linear(
                cfg.encoder_embed_dim,
                self.codebook_size
            ) for _ in range(self.n_codebooks)
        ])
        self.ema_decay = cfg.ema_decay

        codebooks = torch.randn(self.n_codebooks, cfg.encoder_embed_dim, self.codebook_size)
        codebooks = F.instance_norm(codebooks).transpose(1, 2)
        self.codebooks = {
            i: codebooks[i] for i in range(self.n_codebooks)
        }
        self.codebook_cnts = {
            i: torch.ones(self.codebook_size) for i in range(self.n_codebooks)
        }

        ema_cfg = EMAModuleConfig(
            ema_decay=self.ema_decay,
            ema_fp32=True,
        )
        self.teacher = EMAModule(student, ema_cfg)


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Get the state_dict from the superclass
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        # Remove the student model's state_dict to avoid duplication
        student_keys = [key for key in state.keys() if key.startswith(prefix + 'student.')]
        for key in student_keys:
            del state[key]
        
        # Save codebooks and codebook counts
        for i in self.codebooks:
            state[prefix + f'codebooks.{i}'] = self.codebooks[i]
            state[prefix + f'codebook_cnts.{i}'] = self.codebook_cnts[i]
        
        # Save the mask embedding
        state[prefix + 'mask_emb'] = self.mask_emb

        # Save the teacher model's state_dict with a prefix
        teacher_state = self.teacher.model.state_dict(destination=destination, prefix=prefix + 'teacher_model.', keep_vars=keep_vars)
        if self.teacher.config.ema_fp32:
            for key in self.teacher.fp32_params:
                teacher_state[prefix + f'teacher_fp32_params.{key}'] = self.teacher.fp32_params[key]
        state.update(teacher_state)

        return state


    def load_state_dict(self, state_dict, strict=True):
        # Load codebooks and codebook counts
        for i in self.codebooks:
            self.codebooks[i] = state_dict.pop(f'codebooks.{i}')
            self.codebook_cnts[i] = state_dict.pop(f'codebook_cnts.{i}')

        # Load the mask embedding
        self.mask_emb.data.copy_(state_dict.pop('mask_emb'))

        # Load the teacher model's state_dict
        teacher_model_state_dict = {
            k.replace('teacher_model.', ''): v
            for k, v in state_dict.items()
            if k.startswith('teacher_model.')
        }
        self.teacher.model.load_state_dict(teacher_model_state_dict, strict=strict)

        # Remove teacher model entries from state_dict
        for key in list(state_dict.keys()):
            if key.startswith('teacher_model.'):
                del state_dict[key]

        # Load teacher's fp32_params if ema_fp32 is True
        if self.teacher.config.ema_fp32:
            self.teacher.fp32_params = {}
            fp32_keys = [k for k in state_dict.keys() if k.startswith('teacher_fp32_params.')]
            for key in fp32_keys:
                param_name = key.replace('teacher_fp32_params.', '')
                self.teacher.fp32_params[param_name] = state_dict.pop(key)
        
        # Remove any remaining teacher_fp32_params entries from state_dict
        for key in list(state_dict.keys()):
            if key.startswith('teacher_fp32_params.'):
                del state_dict[key]

        # Remove student keys from state_dict to prevent duplication
        student_keys = [key for key in state_dict.keys() if key.startswith('student.')]
        for key in student_keys:
            del state_dict[key]

        # Call the superclass's load_state_dict to load remaining parameters
        super().load_state_dict(state_dict, strict=False)


    def ema_step(self):
        self.teacher.step(self.student)


    def to(self, device, *args, **kwargs):
        self.codebooks = {
            i: codebook.to(device) for i, codebook in self.codebooks.items()
        }
        self.codebook_cnts = {
            i: codebook_cnt.to(device) for i, codebook_cnt in self.codebook_cnts.items()
        }
        # Handle teacher movement
        self.teacher.model.to(device)
        if self.teacher.config.ema_fp32:
            for key in self.teacher.fp32_params:
                self.teacher.fp32_params[key] = self.teacher.fp32_params[key].to(device)

        return super().to(device, *args, **kwargs)


    def forward(self, x):
        features = x # x are the features after CNN feature extraction was applied to them.
        B, T, _ = features.shape

        # Copy the features for the teacher before applying the mask
        pre_encoder_features = features.clone().to(x.device)

        mask_indices = compute_mask_indices(
            (B,T),
            padding_mask=None,
            mask_prob=self.mask_prob,
            mask_length=self.mask_length,
        )
        mask_indices = torch.from_numpy(mask_indices).to(x.device)

        x, layer_results = self.student(x)
        layer_results = [
            l[mask_indices] for l in layer_results # Changes the shape from [B,T,D] to [mask(B*T), D]
        ]

        with torch.no_grad():
            self.teacher.model.eval()

            # Calculate the teacher features
            _, target_layer_results = self.teacher.model.extract_features(
                pre_encoder_features,
                min_layer=self.cfg.encoder_layers - self.cfg.top_k_layers,
            )

            # Apply instance normalization to the teacher features
            target_layer_results = [rearrange(tl, 'b t d -> b d t') for tl in target_layer_results]
            target_layer_results = [
                F.instance_norm(tl) for tl in target_layer_results
            ]
            target_layer_results = [
                rearrange(tl, 'b d t -> b t d') for tl in target_layer_results
            ]

            # Filter out target_layer_results
            target_layer_results = [
                tl[mask_indices] for tl in target_layer_results # Changes the shapr from [B,T,D] to [mask(B*T), D]
            ]

            x = x[mask_indices]

        losses = 0.0
        for i, (target, lr) in enumerate(zip(target_layer_results, layer_results)):
            with torch.no_grad():
                codebook = self.codebooks[i] / self.codebook_cnts[i].unsqueeze(1)

                # Calculate teacher preds distance from codebook words. The total shape of neg_l2_dist is [B*T, CS]
                neg_l2_dist = - (torch.sum(target**2, dim=1, keepdim=True) # shape [B*T, 1]
                                + torch.sum(codebook**2, dim=1) # shape [CS]
                                - 2 * torch.matmul(target, codebook.t()))

                # Create onehot targets according to the maximum negative l2 distance
                onehot_target = torch.zeros_like(neg_l2_dist)
                onehot_target[range(len(neg_l2_dist)), neg_l2_dist.argmax(dim=1)] = 1.0

            # Calculate loss
            pred = self.heads[i](lr)
            pred = F.log_softmax(pred, dim=-1)
            loss = torch.sum(-onehot_target*pred, dim=-1)
            losses += losses + loss

            # Update codebook
            count = onehot_target.sum(0)
            memory = torch.matmul(onehot_target.t(), target)
            alpha = torch.ones_like(count).unsqueeze(1)
            alpha[count!=0] = self.codebook_decay
            self.codebook_cnts[i] = alpha.squeeze(1) * self.codebook_cnts[i] + (1-alpha).squeeze(1) * count
            self.codebooks[i] = alpha * self.codebooks[i] + (1-alpha) * memory

        return (losses / self.n_codebooks).sum()


class MambaModel(nn.Module):
    def __init__(self, config, vocab_size):
        """
        input: config - a yaml file defining the configuration of the model
        output: None
        Description: This function initializes the model with the given configuration
        """
        super(MambaModel, self).__init__()
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
        self.dinosr = DinoSRModel(config, self.encoder)


    def run_dinosr(self, audios):
        features = self.feature_extractor(audios)
        features = rearrange(features, 'b d t -> b t d')
        return self.dinosr(features)


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Get the state_dict from the superclass
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Remove the student model's state_dict from dinosr to avoid duplication
        dinosr_state = self.dinosr.state_dict(destination=destination, prefix=prefix + 'dinosr.', keep_vars=keep_vars)
        state.update(dinosr_state)

        return state


    def load_state_dict(self, state_dict, strict=True):
        # Load the dinosr state dict
        dinosr_keys = [k for k in state_dict.keys() if k.startswith('dinosr.')]
        dinosr_state_dict = {k.replace('dinosr.', ''): state_dict[k] for k in dinosr_keys}
        self.dinosr.load_state_dict(dinosr_state_dict, strict=strict)

        # Remove dinosr entries from state_dict
        for key in dinosr_keys:
            del state_dict[key]

        # Ensure that self.encoder and self.dinosr.student share parameters
        self.dinosr.student = self.encoder

        # Call the superclass's load_state_dict to load remaining parameters
        super().load_state_dict(state_dict, strict=False)


    def ema_step(self):
        self.dinosr.ema_step()


    def to(self, device, *args, **kwargs):
        self.dinosr.to(device)
        return super().to(device, *args, **kwargs)


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


def lr_lambda(current_step, warmup_steps, init_step=0):
    current_step = max(1, current_step+init_step)
    if current_step < warmup_steps:
        return current_step / warmup_steps
    else:
        return (warmup_steps ** 0.5) / (current_step ** 0.5)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_file = "/workspace/fairseq/data/database.csv"

    # read the yaml file inside ./../config/s2t.yaml
    cfg = yaml.safe_load(open("./../config/s2t.yaml"))

    dataset = Dataset(**cfg["dataset"], device=device)
    tokenizer = dataset.get_tokenizer()
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg["dataset"]["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=cfg["dataset"]["num_workers"])

    # Instantiate the metrics, and register them
    metric_recorder = MetricRecorder()
    metric_recorder.register_metric(LossMetric())
    metric_recorder.register_metric(WERMetric(tokenizer))

    mixer_mamba_cfg = MixerMambaConfig(**cfg["model"])

    model = MambaModel(mixer_mamba_cfg, dataset.get_vocab_size()).to(device)
    if os.path.exists("best_model.pth"):
        print("Loading the model.")
        model.load_state_dict(torch.load("best_model.pth"))
        print("Model loaded successfully.")

    # Print the model size
    print(f"Model size: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = getattr(torch.optim, cfg["optimizer"]["name"])(model.parameters(), **cfg["optimizer"]["params"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, cfg["optimization"]["warmup_steps"], cfg["optimization"]["init_step"]))
    criterion = getattr(nn, cfg["criterion"]["name"])(**cfg["criterion"]["params"])

    update_freq = cfg["optimization"]["update_freq"]

    best_loss = None


    for epoch in range(50):
        print(f"Epoch {epoch}")

        progress_bar = tqdm(loader)

        optimizer.zero_grad()

        for step, (audios, tokens, token_lens, transcripts) in enumerate(progress_bar):
            # the model output have no softmax applied to it.
            logits, _ = model(audios)
            dinosr_loss = model.run_dinosr(audios) / update_freq
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Logits are NaN")
                continue
            log_probs = F.log_softmax(logits, dim=-1)

            # Calculate loss
            input_lens = torch.full((log_probs.shape[0],), log_probs.shape[1], device=log_probs.device, dtype=torch.long)
            log_probs = rearrange(log_probs, 'b t v -> t b v')
            loss = criterion(log_probs, tokens, input_lens, token_lens) / update_freq + dinosr_loss * 0.0001

            # Perform a backprop to calculate the new gradients
            if torch.isnan(loss):
                print("Loss is NaN")
                continue
            loss.backward()

            # Update and calculate the metrics
            metrics = metric_recorder.update(step, loss.item(), logits, transcripts)

            if (step + 1) % update_freq == 0:
                # Clip the gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Perform an optimizer step
                optimizer.step()
                model.ema_step()
                scheduler.step()

                # Zero out the gradient state of the model parameters.
                optimizer.zero_grad()

                # TODO:  This is temporary and should change
                if not best_loss:
                    best_loss = metrics["loss"]
                    # save the model
                    torch.save(model.state_dict(), "tmp.pth")
                    os.system("cp tmp.pth best_model.pth")
                elif best_loss > metrics["loss"]:
                    best_loss = metrics["loss"]
                    torch.save(model.state_dict(), "tmp.pth")
                    os.system("cp tmp.pth best_model.pth")

            progress_bar.set_postfix(**metrics)

            if step % cfg["generation"]["freq"] == 0:
                print(tokenizer.decode(logits, from_logits=True, ctc=True))


if __name__ == '__main__':
    main()

