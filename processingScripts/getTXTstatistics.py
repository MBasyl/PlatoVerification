# function for dataframe with word and character count from txt files already chunked
import pandas as pd
import glob
import os
import numpy as np


def combine_files_by_author(input_folder, output_folder):
    # Get a list of all text files in the input folder
    text_files = [file for file in os.listdir(
        input_folder) if file.endswith(".txt")]

    # Create a dictionary to store content for each author
    author_contents = {}

    # Read the content of each file and group by author
    for text_file in text_files:
        name, _ = os.path.splitext(text_file)
        author = name.split("_")[0]
        author_contents.setdefault(author, []).append(text_file)

    # Combine files for each author
    for author, files in author_contents.items():
        combined_content = ""
        for file in files:
            file_path = os.path.join(input_folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                combined_content += f.read()

        # Write the combined content to a new file
        output_file_path = os.path.join(
            output_folder, f"{author}_profile.txt")
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(combined_content)
    print("Combined all authors for profile-approach!")
    return


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


# iterate word_count() on all files in folder and make dataframe
def make_df(folder):
    stats_list = []
    for file in glob.glob(folder):
        stats_list.append(word_count(file))
    df = pd.DataFrame(stats_list)
    return df


def main(folder):
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


if __name__ == "__main__":

    df = main('Rcorpus')
