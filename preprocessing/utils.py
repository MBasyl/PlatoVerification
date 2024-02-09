import pandas as pd
import numpy as np
import os
import shutil
import glob
import re


def replace_named_entities(string1):
    """reads lists of named entities and substitutes them in text with '*'
    """
    named_entities = open(
        'outputs/processing_lists/NER_tokens.txt', 'r').read().splitlines()

    pattern = re.compile(r'\w+')
    cleanstring = ['*' if any(m.group() in named_entities for m in pattern.finditer(
        s)) else s for s in string1.split(" ")]

    return ' '.join(cleanstring)


def chunk_text(input_text, chunk_size, exact=True):
    words = re.findall(r'\S+', input_text)
    chunks = [words[i:i + chunk_size]
              for i in range(0, len(words), chunk_size)]
    if exact:
        chunks = [chunk for chunk in chunks if len(chunk) == chunk_size]
    else:
        chunks = [chunk for chunk in chunks if len(chunk) <= chunk_size]

    return chunks


def combine_plato_profiles(work_list, input_input_directory, new_file_name):
    """combine Plato files according to chronological order for profile processing"""
    output_file_name = f"PlatoProfile/{new_file_name}.txt"
    os.makedirs("PlatoProfile", exist_ok=True)
    for file in glob.glob(input_input_directory + '/*.txt'):
        filename = file.split("/")[-1]
        author = filename.split("_")[0]
        title = filename.split("_")[1].split(".")[0]
        if author == 'Pla':
            if title in work_list:
                # Open the new file in append mode
                with open(output_file_name, 'a') as output_file:
                    try:
                        # Open the corresponding input file with the current title
                        with open(f"{file}", 'r') as input_file:
                            print("working on", file)
                            # Read the content of the input file and write it to the output file
                            content = input_file.read()
                            output_file.write(content)
                    except FileNotFoundError:
                        # If the input file is not found, print a warning
                        print(f"Warning: File not found.")


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


def word_count(file):
    with open(file) as f:
        # get string from file name
        title0 = file.split("/")[1]
        title = title0.split(".")[0]
        word_count = len(f.read().split())
        chunk_size = 1000
        # make dictionary
        stats_dictionary = {"Text_id": title,
                            "Word count": word_count}
        f.close()
    # divide each word_count by 1000 and round down the decimals
    stats_dictionary["Number of chunks"] = np.floor(
        stats_dictionary["Word count"] / chunk_size)

    return stats_dictionary


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


if __name__ == "__main__":

    input_corpus_directory = ""
    output_corpus_directory = "data/"

    os.makedirs(output_corpus_directory, exist_ok=True)

    early = ["1Eu", "2Eu", "Cha", "Cra", "Cri", "Pro"
             "Gor", "Him", "Ion", "Lac", "Lys"]
    mid = ["1Ph", "2Ph", "Sym", "Men", "Rep"]
    late = ["Crt", "Phi", "Sop", "Sta", "Tht", "Tim", "Par"]  # Law
    combine_plato_profiles(work_list=mid, input_input_directory=input_corpus_directory,
                           new_file_name="Plato_Mid")

    # Call the function to combine files by trigram
    # combine_authors_profiles(input_directory_path, output_directory)
