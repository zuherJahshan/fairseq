import csv
from tqdm import tqdm

def contains_mandarin(text):
    return any('\u4e00' <= char <= '\u9fff' for char in text)

def main():
    database_file = '/workspace/fairseq/data/database.csv'
    detected_chars = set()
    # Open the file and read the lines.
    with open(database_file, 'r') as file:
        reader = csv.reader(file)
        for line in tqdm(reader):
            transcription_file = line[1]
            # Open the file and get the transcription.
            with open(transcription_file, 'r') as tfile:
                transcription = tfile.read()
                # check if containning a mandarin characters
                if contains_mandarin(transcription):
                    print(transcription_file)
                # Get the characters.
                chars = set(transcription)
                detected_chars = detected_chars.union(chars)
    
    # Write the characters to a file.
    with open('chars.csv', 'w') as cfile:
        writer = csv.writer(cfile)
        writer.writerow(list(detected_chars))

    return detected_chars

if __name__ == '__main__':
    detected_chars = main()
    print("The detected characters are: ", detected_chars)
