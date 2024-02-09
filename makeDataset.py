from unittest import skip
import pandas as pd
import re
import glob
from preprocessing.utils import chunk_text


def create_dataframe(filename, size):

    # get metadata
    author, title = filename.split("/")[-1].split("_")[:2]
    # skip 0.25 obfuscation:
    if re.match(r'PsPla#0.25', author):
        skip
    else:
        # print(author, title)
        title = title.replace(".txt", "")
        # get binary label based on author
        if author == 'Pla':
            label = 1
        else:
            label = 0

        file = open(filename, "r", encoding="utf-8")
        content = file.read()
        chunks = chunk_text(content, size, exact=True)

        make_dict = {'label': label, 'author': author, 'title': title,
                     'text': chunks}
        df = pd.DataFrame(make_dict)

        return df


def main(folder):
    # iterate function over all files
    df_list = []
    for file in glob.glob(folder + '/*.txt'):
        print(f"creating dataframe from {file}...")
        df = create_dataframe(file, size=3000)
        df_list.append(df)

    final_df = pd.concat(df_list)
    print(final_df.value_counts(['label']))

    #  save to csv
    final_df.to_csv('PARSEDlabelDataset.csv', index=False)
    print("Finished!")


folder_path = "data/PARSED/svmdata2.0"
main(folder_path)
