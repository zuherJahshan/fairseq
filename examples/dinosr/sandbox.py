import unicodedata
import string
import editdistance

def remove_punctuation(text):
    """
    Removes all punctuation from the text, including Unicode punctuation.
    """
    # Use a list comprehension to filter out punctuation characters
    return ''.join(
        ch for ch in text 
        if ch not in string.punctuation and unicodedata.category(ch)[0] != 'P'
    )

def calculate_wer(references, queries):
    # Remove punctuation from both reference and query
    wer = 0
    for reference, query in zip(references, queries):
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
        
        # Calculate WER
        wer += distance / len(ref_words) if ref_words else float('inf')
    
    return wer / len(references)

# Example usage
reference = "Hello, World!"
query = "hello World!"

wer = calculate_wer(reference, query)
print(f"Word Error Rate (WER): {wer:.2%}")

