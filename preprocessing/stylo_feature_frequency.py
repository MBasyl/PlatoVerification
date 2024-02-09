import pandas as pd
import re
import os
import numpy as np


def load_corpus(files="all", corpus_dir="", encoding="UTF-8"):
    """
    Function for loading text files.

    :param files: Vector containing file names or "all" to use all files in the corpus directory
    :param corpus_dir: Directory containing the corpus files
    :param encoding: Encoding of the text files
    :return: Loaded corpus as a dictionary
    """
    # First of all, retrieve the current path name
    original_path = os.getcwd()

    # Checking if the specified directory exists
    if isinstance(corpus_dir, str) and len(corpus_dir) > 0:
        # Checking if the desired directory exists and if it is a directory
        if os.path.exists(corpus_dir) and os.path.isdir(corpus_dir):
            # If yes, then set the new working directory
            os.chdir(corpus_dir)
        else:
            # Otherwise, stop the script
            raise FileNotFoundError(
                f"There is no directory {os.getcwd()}/{corpus_dir}")

    else:
        # If the argument was empty, then relax
        print("Using the current directory...")

    # Now, checking which files were requested; usually, the user is expected to specify a vector with samples' names
    if files == "all":
        files = [f for f in os.listdir() if os.path.isfile(f)]

    # Variable initialization
    loaded_corpus = {}

    # Uploading all files listed in the vector "files"
    for file in files:
        if not os.path.exists(file):
            print("!")
            print(f"\"{file}\"? No such file -- check your directory!")
        else:
            # Loading the next file from the list "files";
            # If an error occurred, ignore it and send a message on the screen
            try:
                with open(file, 'r', encoding=encoding) as f:
                    current_file = f.readlines()
            except Exception as e:
                current_file = None
                print("!")
                print(
                    f"The file {file} could not be loaded for an unknown reason: {e}")

            # If successful, append the scanned file into the corpus; otherwise, send a message
            if current_file:
                loaded_corpus[file] = current_file

    # Returning to the original path
    os.chdir(original_path)

    # Returning the value
    return loaded_corpus


def load_corpus_and_parse(corpus_dir, features, ngram_size, files="all", encoding="UTF-8"):
    """
    High-level function that ties a number of other functions responsible
    for uploading texts, deleting markup, sampling, splitting into n-grams, etc.

    :param files: List of sample names or "all" to use all files in the corpus directory
    :param corpus_dir: Directory containing the corpus files
    :param sample_size: Size of the sample to load
    :param encoding: Encoding of the text files
    :return: Loaded and processed corpus as a dictionary
    """
    # Checking which files were requested; usually, the user is expected to specify a vector with samples' names
    if files == "all":
        files = [f for f in os.listdir(corpus_dir) if os.path.isfile(
            os.path.join(corpus_dir, f))]

    loaded_corpus = load_corpus(
        files=files, corpus_dir=corpus_dir, encoding=encoding)

    # Dropping file extensions from sample names
    loaded_corpus = {os.path.splitext(
        name)[0]: text for name, text in loaded_corpus.items()}
    print()

    # Deleting punctuation, splitting into words
    print("Slicing input text into tokens...")
    loaded_corpus = {name: txt_to_words(
        text, features, ngram_size) for name, text in loaded_corpus.items()}
    # Assigning a class
    # loaded_corpus["class"] = "stylo.corpus"

    # Returning the value
    return loaded_corpus


def txt_to_words(text, features, ngram_size, splitting_rule=None, preserve_case=True):
    """
    Function for splitting a given input text into single words.

    :param input_text: Text to be split (string)
    :param splitting_rule: Custom splitting rule (regular expression)
    :param preserve_case: If True, preserves the case; if False, converts to lowercase
    :return: List of words
    """
    # Converting characters to lowercase if necessary
    if not preserve_case:
        try:
            text = text.lower()
        except AttributeError as e:
            print(f"Turning into lowercase failed: {e}")
            text = "empty"

    # extract text from list
    text = "".join(text)
    if features == 'c':
        text = re.sub('PROPN', "*", text)
    # If no custom splitting rule was detected...
    if splitting_rule is None or len(splitting_rule) == 0:
        # Splitting into units specified by the regular expression
        # All sequences between non-letter characters are assumed to be words
        splitting_rule = "[^A-Za-z" \
            "*" \
            "\u00C0-\u00FF" \
            "\u0100-\u01BF" \
            "\u01C4-\u02AF" \
            "\u0386\u0388-\u03FF" \
            "\u0400-\u0481\u048A-\u0527" \
            "\u05D0-\u05EA\u05F0-\u05F4" \
            "\u0620-\u065F\u066E-\u06D3\u06D5\u06DC" \
            "\u1E00-\u1EFF" \
            "\u1F00-\u1FBC\u1FC2-\u1FCC\u1FD0-\u1FDB\u1FE0-\u1FEC\u1FF2-\u1FFC" \
            "\u03E2-\u03EF\u2C80-\u2CF3" \
            "\u10A0-\u10FF" \
            "\u3040-\u309F" \
            "\u30A0-\u30FF" \
            "\u3005\u3031-\u3035" \
            "\u4E00-\u9FFF" \
            "\u3400-\u4DBF" \
            "\uAC00-\uD7AF" \
            "]+"
        tokenized_text = re.split(splitting_rule, text)
    else:
        raise ValueError("Can't split.")

    # print(tokenized_text[:10])

    # Getting rid of empty strings
    tokenized_text = [word for word in tokenized_text if len(word) > 0]
    tokenized_text = " ".join(tokenized_text)
    featurized_text = txt_to_features(tokenized_text, features, ngram_size)
    return featurized_text


def txt_to_features(text, feature="w", n=1):
    """
    Function that carries out the necessary modifications for feature selection.
    Converts an input text into the type of sequence needed (n-grams, etc.)
    and returns the new list of items.

    :param tokenized_text: Vector of words (or chars)
    :param features: Type of features ("w" for words, "c" for characters)
    :param ngram_size: Size of n-grams (default is 1)
    :return: New list of items
    """
    text = ''.join(text)
    if feature == 'w':
        tokens = text.split()
        sample = [' '.join(tokens[i:i + n])
                  for i in range(len(tokens) - n + 1)]
    elif feature == 'c':
        tokens = list(text)
        sample = [''.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    else:
        raise ValueError("Invalid feature. Use 'w' or 'c'.")

    return sample


def make_frequency_list(data: dict, value=False, head=None, relative=True):
    """
    Function for generating a frequency list of words or other linguistic features.
    It counts the elements of a vector and returns a vector of these elements in descending order of frequency.

    :param data: Input data, either a stylo.corpus or a list/vector
    :param value: If True, return frequencies along with values; if False, return only feature names
    :param head: Limit the number of most frequent features
    :param relative: If True, return relative frequencies; if False, return raw frequencies
    :return: Frequency list of features
    """

    # Sanitize the input dataset
    if isinstance(data, dict):
        data = list(data.values())
    elif not isinstance(data, (list, pd.Series, np.ndarray)):
        raise ValueError("Unable to make a list of frequencies")

    # Check if the dataset has at least two elements
    if len(data) < 3:
        raise ValueError(
            "You are trying to measure frequencies of an empty vector!")

    # The dataset sanitized, let counting the features begin!
    frequent_features = pd.Series(np.concatenate(
        data)).value_counts(sort=True, ascending=False)

    # If relative frequencies were requested, they are normalized
    if relative:
        frequent_features = frequent_features / len(np.concatenate(data)) * 100
        print(frequent_features)
        print(len(np.concatenate(data)))
        exit(0)

    # Additionally, one might limit the number of the most frequent features
    if isinstance(head, (int, float)):
        head = abs(round(head))
        if head == 0:
            head = 1
        frequent_features = frequent_features.head(head)

    return frequent_features


def make_frequency_table(data: dict, frequencies):
    # Create a DataFrame with raw frequencies
    frequency_df = pd.DataFrame(index=data.keys(), columns=frequencies.index)

    # Fill in the frequency dataframe with normalized frequency values by document length
    for doc, tokens in data.items():
        doc_length = len(tokens)
        normalized_frequencies = frequencies / doc_length
        frequency_df.loc[doc] = normalized_frequencies

    return frequency_df


def main(head):
    tokenized_corpus = load_corpus_and_parse(
        files="all", corpus_dir=input_dir, features=feat, ngram_size=n, encoding="UTF-8")

    freq_list = make_frequency_list(tokenized_corpus, head=head)
    df = make_frequency_table(tokenized_corpus, freq_list)
    ##
    # Split index into 'author' and 'text' columns
    split_index = df.index.to_series().str.split('_', expand=True)
    df.insert(0, 'text', split_index[1])
    df.insert(0, 'author', split_index[0])

    # df[['author', 'text']] = df.index.str.split('_', 1, expand=True)

    # Drop the original index column
    df = df.reset_index(drop=True)

    df.to_csv(f"adhoc5cPLAINPlato.csv", index=False)
    # index=list(df.index))
    print(df)
    print("done!", feat, n)
    return df


if __name__ == '__main__':
    # Assuming you have a DataFrame named 'token_freq_df' with authors as index and tokens as columns.
    # Each cell contains the frequency of the corresponding token for the respective author.

    input_dir = 'data/secondadhoc/PLAIN'
    feat = 'c'
    n = 5
    main(head=3000)

    exit(0)
    input_dir = 'platoCorpus/PLAIN'
    feat = 'c'
    n = 3
    main(head=200)
    exit(0)
    # PARSED CORPUS
    input_dir = 'platoCorpus/PARSEDcomplex'
    feat = 'w'
    for n in range(2, 10):
        print(feat, n)
        main()

    # PLAIN corpus
    input_dir = 'platoCorpus/PLAIN'
    f = ['w', 'c']
    n = 1
    for feat in f:
        for n in range(2, 10):
            print(f, n)
            main()

    #  df = pd.read_csv(f"frequency{feat,n}Plato.csv", index_col=0)
