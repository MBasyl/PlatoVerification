import json
import os
import json
import random


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

    folder = "platoCorpus/PLAIN"
    formatted_data = format_directory_to_json(folder)

    # train, test = split_dataset(formatted_data)
    # Save the formatted data as a JSON file
    # with open("PlatoCorpus/PLAINaug/ParsedPlatoSPLITtrain.json", 'w', encoding='utf-8') as json_file:
    #    json.dump(train, json_file, indent=4)
    with open("PlatoCorpus/PLAINaug/ProfilePlato.json", 'w', encoding='utf-8') as json_file:
        json.dump(formatted_data, json_file, indent=4)

    print(f"Formatted data saved to: {json_file}")
