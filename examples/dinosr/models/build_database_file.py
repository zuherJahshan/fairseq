import os
import sys
import glob
from tqdm import tqdm
import csv
import argparse

def main():
    # Recieve from the user the dir that contains the audio, and the dir that contains the transcripts
    parser = argparse.ArgumentParser(description='Build the database file')
    parser.add_argument('--audio_dir', '-a', type=str, help='The directory that contains the audio files, recursively', required=True)
    parser.add_argument('--transcript_dir', '-t', type=str, help='The directory that contains the transcripts files, recursively', required=True)
    parser.add_argument('--output_file', '-o', type=str, default="./database", required=False, help='The output file that will contain the database')
    # add optional argument for the audio file type
    parser.add_argument('--audio_type', type=str, default='flac', required=False, help='The type of the audio files')
    args = parser.parse_args()

    # Get the audio files
    audio_files = glob.glob(os.path.join(args.audio_dir, '**', f'*.{args.audio_type}'), recursive=True)
    # Get the transcript files

    # The structure of the files should be s.t.
    # every .flac file have a corresponding .txt file in the same relative path
    # path (relative to the audio_dir and transcript_dir).
    # To build the database file successfully, we need to check that every audio has a transcript and vice versa.
    with open(args.output_file + '.csv', 'w') as f:
        writer = csv.writer(f)
        for audio_file in tqdm(audio_files):
            # Get the relative path of the audio file
            relative_path = os.path.relpath(audio_file, args.audio_dir)
            # Check if path exists in the transcript files
            transcript_file = os.path.join(args.transcript_dir, relative_path.replace(f'.{args.audio_type}', '.txt'))
            if os.path.exists(transcript_file):
                # insert them to the database file
                writer.writerow([audio_file, transcript_file])


if __name__ == '__main__':
    main()
