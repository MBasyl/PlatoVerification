import json
import os


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


if __name__ == "__main__":

    folder = "data/Rcorpus"
    formatted_data = format_directory_to_json(folder)

    # Save the formatted data as a JSON file
    output_path = os.path.join("data", "MLcorpus", "SVM_data.json")
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(formatted_data, json_file, indent=4)

    print(f"Formatted data saved to: {output_path}")
