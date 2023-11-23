import os
import re


def chunk_text(input_text, chunk_size):
    words = re.findall(r'\S+', input_text)
    chunks = [words[i:i + chunk_size]
              for i in range(0, len(words), chunk_size)]
    # Remove chunks that are smaller than chunk_size
    chunks = [chunk for chunk in chunks if len(chunk) == chunk_size]

    return chunks


def distribute_and_chunk(input_directory, output_directory, chunk_size=1000):
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
            words_dropped = len(content.split()) - (len(chunks)*chunk_size)
            with open(os.path.join("words_dropped_log.txt"), "a", encoding="utf-8") as log_file:
                log_file.write(
                    f"{filename}_{i + 1}: {words_dropped} words dropped\n")


def main(input_corpus_directory):
    # Ensure output directory exists or create it
    os.makedirs(output_corpus_directory, exist_ok=True)
    distribute_and_chunk(input_corpus_directory, output_corpus_directory)

    print("\ndone parsing directories")
    return


if __name__ == "__main__":
    input_corpus_directory = "Rcorpus"
    output_corpus_directory = "MLcorpus"

    main(input_corpus_directory)
