import os
import json
import re
import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Define input and output paths
input_file = "output/formatted_data.json"
output_file = "output/formatted_data_cleaned.json"
removed_chars_file = "output/removed_characters.log"

# Ensure NLTK resources are downloaded
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Define allowed punctuations from NLTK
allowed_punctuations = set(string.punctuation)

# Function to remove non-ASCII characters (including extended ASCII) and log removed characters
def remove_non_ascii(text, removed_counter):
    temp_removed = ""

    # Iterate through the text to identify and process non-ASCII sequences
    for char in text:
        if ord(char) >= 256 and char not in allowed_punctuations:  # Non-ASCII character not in allowed punctuation
            temp_removed += char
        else:
            if temp_removed:  # End of a consecutive non-ASCII sequence
                removed_counter[temp_removed] += 1
                temp_removed = ""

    if temp_removed:  # Capture any remaining non-ASCII sequence
        removed_counter[temp_removed] += 1

    # Remove non-ASCII characters except allowed punctuations
    cleaned_text = "".join(char for char in text if ord(char) < 256 or char in allowed_punctuations)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Load the formatted dataset
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare a counter to store removed characters
removed_counter = Counter()

# Process the data and clean non-ASCII characters
cleaned_data = []
for entry in data:
    cleaned_entry = {
        "id": entry["id"],
        "original_id": entry["original_id"],
        "table_title": remove_non_ascii(entry["table_title"], removed_counter),
        "caption": remove_non_ascii(entry["caption"], removed_counter),
        "table": [[remove_non_ascii(cell, removed_counter) for cell in row] for row in entry["table"]],
        "sentence_context": [remove_non_ascii(sentence, removed_counter) for sentence in entry["sentence_context"]]
    }
    cleaned_data.append(cleaned_entry)

# Save the cleaned dataset
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

# Save the removed characters to a log file as counts, sorted by count in descending order
with open(removed_chars_file, "w", encoding="utf-8") as log_f:
    log_f.write("Character/Sequence\tCount\n")
    for seq, count in removed_counter.most_common():
        log_f.write(f"{seq}\n{count}\n\n")

print(f"Cleaned dataset saved to {output_file}")
print(f"Removed characters log saved to {removed_chars_file}")
