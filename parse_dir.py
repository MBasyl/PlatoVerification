import os
import re
import shutil
# from sklearn.model_selection import train_test_split


def chunk_text(input_text, chunk_size):
    words = re.findall(r'\S+', input_text)
    chunks = [words[i:i + chunk_size]
              for i in range(0, len(words), chunk_size)]

    chunks = [chunk for chunk in chunks if len(chunk) == chunk_size]

    return chunks


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


def split_train_test(input_directory, output_directory='data/GIplain', test_size=0.2):
    """split author subdirectories in test and train"""
    # Create subdirectories for 'test' and 'train'
    os.makedirs(output_directory, exist_ok=True)

    os.makedirs(os.path.join(output_directory, 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'train'), exist_ok=True)

    # List all authors
    authors = [author for author in os.listdir(
        input_directory) if os.path.isdir(os.path.join(input_directory, author))]
    test_authors = [auth for auth in authors if auth in [
        'Pla', 'Disputed', 'PsPla']]
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


def combine_plato_profiles(work_list, input_input_directory, new_file_name, output_file_name=None):
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

    # Create a dictionary to store files based on their starting trigram
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


if __name__ == "__main__":
    input_corpus_directory = "data/Rcorpus"
    output_corpus_directory = "data/MLcorpus"

    os.makedirs(output_corpus_directory, exist_ok=True)
    # group_and_chunk(input_corpus_directory, output_corpus_directory)
    # split_train_test(output_corpus_directory)

    early = ["1Eu", "2Eu", "Cha", "Cra", "Cri", "Pro"
             "Gor", "Him", "Ion", "Lac", "Lys"]
    mid = ["1Ph", "2Ph", "Sym", "Men"]
    late = ["Crt", "Phi", "Sop", "Sta", "Tht", "Tim"]
    # combine_plato_profiles(work_list= mid, input_input_directory=input_directory_path, new_file_name="Plato_Mid",output_file_name=None)

    # Call the function to combine files by trigram
    combine_authors_profiles(input_directory_path, output_directory)
