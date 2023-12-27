import pandas as pd
import numpy as np
import os


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


df = create_dataframe("data/PLAIN")
# Bit of statistics
# Extract the median word count for each author
median_word_count_by_author = df.groupby(
    'Author')['Word count'].median().reset_index()

# create subset df with 'Plato' only made of the later works plus Parmenides and 2Phaedrus
late = ["Crt", "Phi", "Sop", "Sta", "Tht", "Tim", "2Ph", "Par", "Law"]  # "Rep"

index_names = df[(df['Author'] == 'Pla') & (~df['Text'].isin(late))].index
df.drop(index_names, inplace=True)  # 75 texts, 13 authors

df_small = df[~df['Text'].isin(['Law'])]
median_word_count_by_author = df_small.groupby(
    'Author')['Word count'].median().reset_index()

# make new overview dataset per profile rounding up median
data = {
    "Alcidamas": {"Mean": 2400, "Number of texts": 1},
    "Aristoteles": {"Mean": 43400, "Number of texts": 10},
    "Demosthenes": {"Mean": 2900, "Number of texts": 10},
    "Epicurus": {"Mean": 4200, "Number of texts": 2},
    "Gorgias": {"Mean": 2700, "Number of texts": 1},
    "Hyperides": {"Mean": 2500, "Number of texts": 4},
    "Isocrates": {"Mean": 4100, "Number of texts": 9},
    "Lysias": {"Mean": 2700, "Number of texts": 4},
    "Plato": {"Mean": 18200, "Number of texts": 9},
    "Plato-Laws": {"Mean": 110000, "Number of texts": 1},
    "PseudoPlato": {"Mean": 4100, "Number of texts": 8},
    "Speusippus": {"Mean": 5700, "Number of texts": 1},
    "Xenocrates": {"Mean": 5000, "Number of texts": 1},
    "Xenophon": {"Mean": 10.000, "Number of texts": 14}
}

df = pd.DataFrame(data)
df.to_csv("Overview_Profile.csv", index=False)
