import os
import re
import shutil
import glob
from preprocessing.utils import combine_authors_profiles, combine_plato_profiles, chunk_text
# from sklearn.model_selection import train_test_split


# SCRIPT TO RUN RUZICKA'S GI method

def group_and_chunk(input_directory, output_directory, chunk_size=1000):
    """group single docs in author subdirectories"""
    # Create subdirectories for authors
    authors = set()
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            author = filename.split("_")[0]
            authors.add(author)
            author_dir = os.path.join(output_directory, author)
            os.makedirs(author_dir, exist_ok=True)

    # Process each text file
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            author, title = filename.split("_")[:2]
            title = title.replace(".txt", "")
            input_file_path = os.path.join(input_directory, filename)
            output_dir = os.path.join(output_directory, author)

            with open(input_file_path, "r", encoding="utf-8") as file:
                content = file.read()

            chunks = chunk_text(content, chunk_size)

            # Write chunks to separate files
            for i, chunk in enumerate(chunks):
                output_file_path = os.path.join(
                    output_dir, f"{title}_{i + 1}.txt")
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    output_file.write(" ".join(chunk))

            # Log the number of words dropped
            words_dropped = len(content.split()) - (len(chunks) * chunk_size)
            # with open(os.path.join(output_directory, "words_dropped_log.txt"), "a", encoding="utf-8") as log_file:
            #    log_file.write(
            #        f"{filename}_{i + 1}: {words_dropped} words dropped\n")


def split_train_test(input_directory, output_directory='data/GIprofiles', test_size=0.2):
    """split author subdirectories in test and train"""
    # Create subdirectories for 'test' and 'train'
    os.makedirs(output_directory, exist_ok=True)

    os.makedirs(os.path.join(output_directory, 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'train'), exist_ok=True)

    # List all authors
    authors = [author for author in os.listdir(
        input_directory) if os.path.isdir(os.path.join(input_directory, author))]
    test_authors = [auth for auth in authors if auth in [
        'Pla', 'Disputed']]  # Ano*
    train_authors = [auth for auth in authors if auth not in test_authors]
    # Split the authors into train and test
    # train_authors, test_authors = train_test_split(
    #    authors, test_size=test_size, random_state=42)

    # Move files to 'train' and 'test' directories
    for author in train_authors:
        source_dir = os.path.join(input_directory, author)
        dest_dir = os.path.join(output_directory, 'train', author)
        shutil.copytree(source_dir, dest_dir)

    for author in test_authors:
        source_dir = os.path.join(input_directory, author)
        dest_dir = os.path.join(output_directory, 'test', author)
        shutil.copytree(source_dir, dest_dir)


if __name__ == "__main__":
    input_corpus_directory = "data/GI"
    output_corpus_directory = "data/GI/preprocess"

    os.makedirs(output_corpus_directory, exist_ok=True)
    group_and_chunk(input_corpus_directory,
                    output_corpus_directory, chunk_size=5000)
    split_train_test(output_corpus_directory)
