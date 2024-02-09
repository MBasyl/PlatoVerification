import json
import os
import json
import random
import pandas as pd
from utils import word_count
import glob
# iterate word_count() on all files in folder and make dataframe


def make_df(folder):
    stats_list = []
    for file in glob.glob(folder):
        stats_list.append(word_count(file))
    df = pd.DataFrame(stats_list)
    return df


def get_txt_stats(folder):
    # stack df from multple folders
    df_list = []

    df_list.append(make_df(folder + "/*.txt"))

    df = pd.concat(df_list)
    df_complete = df.reset_index(drop=True)
    print('\n\nOverview\n')
    print(df_complete.describe())
    print(df_complete.info(verbose=False))

    df_complete.to_csv(f"Overview{folder}.csv", index=False)

    print(f"\n\nfinished! see Overview{folder}.csv")

    return df_complete


def create_dataframe(input_directory):
    df = pd.DataFrame()
    f = pd.read_csv("outputs/processing_lists/authorList.txt", header=None)
    authorList = set(f.iloc[:, 0].tolist())  # 13
    titleList = set(f.iloc[:, 1].tolist())
    authors = []
    titles = []
    word_count = []
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            # get 'Author' from file name
            author, title = filename.split("_")[:2]
            authors.append(author)
            title = title.replace(".txt", "")
            titles.append(title)

            input_file_path = os.path.join(input_directory, filename)
            file = open(input_file_path, "r", encoding="latin-1")
            content = file.read()
            word_count.append(len(content.split()))

    df['Author'] = authors
    # df['Authors'] = match_trigrams_to_full_names(authors, authorList)
    df['Text'] = titles
    # df['Text'] = match_trigrams_to_full_names(titles, titleList)
    df["Word count"] = word_count
    df.to_csv(f"outputs/processing_lists/Overview_Dataset.csv", index=False)
    print(df.head())
    return df


def format_directory_to_json(directory_path):
    data = {}

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename == '.DS_Store':
            continue

        # Check if the path is a file and not a directory
        if os.path.isfile(file_path):
            # Extract author and title from the filename
            author = filename.split("_")[0]
            title = filename.split("_")[1].split(".")[0]
            if title == '.DS Store':
                continue
            # Read the content of the file
            print(author, title)

            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                book_text = file.read()

            # Create or update the dictionary
            if author not in data:
                data[author] = {title: book_text}
            else:
                data[author][title] = book_text

    return data


def split_dataset(dataset):
    # Initialize train and test datasets
    train_dataset = {}
    test_dataset = {}

    for key, subdict in dataset.items():
        # Extract keys and shuffle them
        keys = list(subdict.keys())
        random.shuffle(keys)

        # Split keys into train and test (with train having the extra one if odd)
        split_point = len(keys) // 2 + len(keys) % 2
        train_keys = keys[:split_point]
        test_keys = keys[split_point:]

        # Populate train dataset
        train_dataset[key] = {k: subdict[k] for k in train_keys}

        # Populate test dataset
        test_dataset[key] = {k: subdict[k] for k in test_keys}

    return train_dataset, test_dataset


if __name__ == "__main__":
    df = create_dataframe("data/PLAIN")

    folder = "platoCorpus/PLAIN"
    formatted_data = format_directory_to_json(folder)

    # train, test = split_dataset(formatted_data)
    # Save the formatted data as a JSON file
    # with open("PlatoCorpus/PLAINaug/ParsedPlatoSPLITtrain.json", 'w', encoding='utf-8') as json_file:
    #    json.dump(train, json_file, indent=4)
    with open("PlatoCorpus/PLAINaug/ProfilePlato.json", 'w', encoding='utf-8') as json_file:
        json.dump(formatted_data, json_file, indent=4)

    print(f"Formatted data saved to: {json_file}")
