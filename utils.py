
import os
import shutil
import re
import glob
import random


def extract_random_chunk(file_path, new_file, chunk_size=8000):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    print(len(content.split()))
    # Find all sentence-ending positions
    sentence_end_positions = [match.end()
                              for match in re.finditer(r'[.;]', content)]

    if not sentence_end_positions:
        raise ValueError(
            "No sentence-ending periods or question marks found in the document.")
    # max_position = max(sentence_end_positions) - chunk_size
    target = 557830  # sentence_end_positions.index(max_position)
    subset_random = sentence_end_positions[:target]
    # Randomly select a sentence-ending position to start the chunk
    start_position = random.choice(subset_random)

    # Find the nearest word boundary after the selected position
    start_boundary = content.rfind(' ', 0, start_position) + 1

    # Extract words from the selected position to meet the specified chunk size
    words = re.findall(r'\S+', content[start_boundary:])
    selected_chunk = ' '.join(words[:chunk_size])
    print(len(selected_chunk.split()))
    with open(new_file, "w", encoding="utf-8") as save_file:
        save_file.write(selected_chunk)


def count_words(text):
    return len(text.split())


def replace_named_entities(string1):
    """reads lists of named entities and substitutes them in text with '*'
    """
    named_entities = open(
        'outputs/NER_tokens.txt', 'r').read().splitlines()

    pattern = re.compile(r'\w+')
    cleanstring = ['*' if any(m.group() in named_entities for m in pattern.finditer(
        s)) else s for s in string1.split(" ")]

    return ' '.join(cleanstring)


def cut_string_in_half(words):
    halfway_point = len(words) // 2
    first_half = words[:halfway_point]
    second_half = words[halfway_point:]
    return first_half, second_half


def make_dataset_dictionary(folder, max_length=4300):
    """Make dictionary of AUTHOR-PROFILES"""
    authors = []
    text_name = []
    contents = []
    for file in glob.glob(folder + "/*.txt"):
        content = open(file, "r", encoding="utf-8").read()
        words = re.findall(r'\S+', content)
        author, text = file.split("/")[-1].split(".")[0].split("_")
        chunks = chunk_text(words, max_length, exact=True)
        for c in chunks:
            contents.append(' '.join(c))
            text_name.append(text)
            authors.append(author)
    print(text_name)
    print(len(text_name))
    return authors, text_name, contents


def chunk_text(words, chunk_size, exact=True):
    # words = re.findall(r'\S+', input_text)
    chunks = [words[i:i + chunk_size]
              for i in range(0, len(words), chunk_size)]
    if exact:
        chunks = [chunk for chunk in chunks if len(chunk) == chunk_size]
    else:
        chunks = [chunk for chunk in chunks if len(chunk) <= chunk_size]

    return chunks


def combine_authors_profiles(input_directory, output_directory):
    # Get a list of all files in the specified input_directory
    files = os.listdir(input_directory)
    # Create a dictionary to store content for each author
    trigram_files = {}
    # Iterate through each file in the input_directory
    for filename in files:
        # Extract the first three characters as the trigram
        trigram = filename[:3]
        # Append the filename to the corresponding trigram key in the dictionary
        trigram_files.setdefault(trigram, []).append(filename)
    print("Remember, escluding psPlato docs!!")
    # Iterate through the dictionary and combine files with the same trigram
    for trigram, filenames in trigram_files.items():
        if len(filenames) > 1 and trigram != "PsP":
            # Combine files only if there is more than one file with the same trigram
            output_filename = f"{trigram}_combined.txt"
            # Open the output file in binary write mode
            with open(output_filename, 'wb') as output_file:
                for filename in filenames:
                    # Open each input file in binary read mode and write its content to the output file
                    with open(os.path.join(output_directory, filename), 'rb') as input_file:
                        shutil.copyfileobj(input_file, output_file)
            print(
                f"Combined files with trigram '{trigram}' into '{output_filename}'")


def match_trigrams_to_full_names(trigram_names, full_names):
    matched_names = []

    for trigram in trigram_names:
        # Check if the trigram matches the first three characters of any full name
        matches = [full_name for full_name in full_names if full_name.lower(
        ).startswith(trigram.lower())]

        if matches:
            matched_names.extend(matches)
    print(len(matched_names))
    return matched_names