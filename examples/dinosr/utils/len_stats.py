import os
import glob
import csv
import torchaudio
import matplotlib.pyplot as plt
import tqdm

datapath = "/workspace/fairseq/data/database.csv"

x = []
y = []

with open(datapath, "r") as f:
    reader = csv.reader(f)
    idx = 0
    for flac_file, txt_file in tqdm.tqdm(reader):
        idx += 1
        # Check the number of samples taken in the flac file
        audio, sample_rate = torchaudio.load(flac_file)
        if sample_rate != 16_000:
            continue
        with open(txt_file, "r") as f:
            txt = f.read()
            # txt = txt.replace("\n", "")
            # txt_len = len(txt)
            txt_len = len(txt.split())
        audio_len = audio.size(1)
        if audio_len > 480_000:
            continue
        y.append(txt_len)
        x.append(audio_len)
        if idx % 4096 == 0:
            plt.scatter(x, y)
            plt.xlabel("Audio length")
            plt.ylabel("Text length")
            plt.savefig("len_stats.png")
            # clean the plot
            plt.clf()



