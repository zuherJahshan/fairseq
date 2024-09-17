import os
import sys
import torch
import torchaudio
import webrtcvad
import progressbar
import glob

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

def frame_generator(frame_duration_ms, waveform, sample_rate):
    """Generates audio frames from the waveform."""
    frame_length = int(sample_rate * (frame_duration_ms / 1000.0))
    total_length = waveform.size(1)
    num_frames = total_length // frame_length
    for i in range(num_frames):
        start = i * frame_length
        end = start + frame_length
        frame = waveform[:, start:end]
        timestamp = i * (frame_duration_ms / 1000.0)
        duration = frame_duration_ms / 1000.0
        yield frame, timestamp, duration
    # Discard any remaining incomplete frame at the end

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Detects speech segments and yields the waveform for each segment."""
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = []
    triggered = False

    voiced_frames = []
    for frame, timestamp, duration in frames:
        # Convert the frame to 16-bit integers
        frame_int16 = (frame * 32768).short()
        frame_bytes = frame_int16.numpy().tobytes()

        if len(frame_bytes) == 0:
            continue  # Skip empty frames

        is_speech = vad.is_speech(frame_bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * num_padding_frames:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * num_padding_frames:
                triggered = False
                yield torch.cat(voiced_frames, dim=1)
                ring_buffer.clear()
                voiced_frames = []
    if voiced_frames:
        yield torch.cat(voiced_frames, dim=1)



def main(audiofiles, base_dir="output"):
    bar = progressbar.ProgressBar(maxval=len(audiofiles))
    bar.start()

    for i, audio_file in enumerate(audiofiles):
        if not os.path.exists(audio_file):
            sys.stderr.write(f"File '{audio_file}' not found.\n")
            sys.exit(1)
        # Generate output directory path resembling the input path
        basedir = os.path.join(base_dir, os.path.dirname(audio_file))
        filename = os.path.splitext(os.path.basename(audio_file))[0]
        output_dir = os.path.join(basedir, filename)
    
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        waveform, sample_rate = read_audio(audio_file)
        vad = webrtcvad.Vad(2)  # Aggressiveness from 0 to 3
    
        frame_duration_ms = 30  # Duration of each frame in milliseconds
        padding_duration_ms = 300  # Duration for VAD padding
    
        frames = list(frame_generator(frame_duration_ms, waveform, sample_rate))
        segments = list(vad_collector(sample_rate, frame_duration_ms,
                                      padding_duration_ms, vad, frames))
    
        for j, segment in enumerate(segments):
            segment_duration = segment.size(1) / sample_rate
            segment_filename = f'segment_{j:03d}.flac'
            segment_path = os.path.join(output_dir, segment_filename)
            write_audio(segment_path, segment, sample_rate)
        bar.update(i + 1)
    bar.finish()

if __name__ == '__main__':
    # build a list of all files inside some subdirectory in the unlkab_60k directory
    audiofiles = glob.glob("unlab_60k/**/*.flac", recursive=True)
    len(audiofiles)
    
    main(audiofiles)

