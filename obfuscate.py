# Transform LAWS to our Disputed:
import random
import re
import glob
import os
import utils
import pandas as pd


def make_dataset(data_type: str) -> pd.DataFrame:
    """"PARSED or PLAIN"""
    authors, text_name, contents = utils.make_dataset_dictionary(
        folder=f'{data_type}Profiles', max_length=4300)
    if 'Law' in text_name:
        print("WARNING: did not remove Laws and Lovers from data")
    dataset = {
        'author': authors,
        'title': text_name,
        'text': contents}
    df = pd.DataFrame(dataset)
    # INCLUDE PseudoPlato shorter than 4k which had been excluded + VII
    df = insert_data(df, f"{data_type}Profiles/PsPla_The.txt")
    df = insert_data(df, f"{data_type}Profiles/PsPla_Min.txt")
    df = insert_data(df, f"{data_type}Profiles/PsPla_Hip.txt")
    df = insert_data(df, f"{data_type}Profiles/Disputed_VII.txt")
    df.to_csv(f"{data_type}dataset_unobfusc.csv", index=False)
    return


def insert_data(df, new_file):
    author, title = new_file.split("/")[-1].split("_")
    content = open(new_file, "r").read()
    new_row = {'author': author, 'title': title.split(".")[0], 'text': content}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print(author, title.split(".")[0])
    return df


def split_Laws(laws_file, data_type, chunk_size):
    words = re.findall(r'\S+', laws_file)
    chunks = utils.chunk_text(words, chunk_size=chunk_size, exact=True)
    # Write chunks to separate files
    for i, chunk in enumerate(chunks):
        output_file_path = os.path.join(
            f"configure{data_type}", f"Pla_Law{i + 1}.txt")
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write(" ".join(chunk))
    print("Now MANUALLY clip off hanging sentences! ")


def obfuscate(folder, p):
    for filename in os.listdir(folder):
        author = filename.split("_")[0]
        title = filename.split("_")[1].split(".")[0]
        if author == 'Pla' and title in ['Law3', 'Law7', 'Law10', 'Law15', 'Law20']:
            file = open(os.path.join(folder, filename),
                        "r", encoding="utf-8")
            sentences = file.read().split(" _-_ ")
            print(filename, "number of sentences:", len(sentences))
            for i in range(round(p * len(sentences))):
                number_sentences = random.randint(0, len(sentences)-1)
                sentences.pop(number_sentences)

                #  inject random sentences
                new_filename = f'PsPla_{title}#{p}.txt'
                lovers = open("configureAlt/PsPla_Lov.txt",
                              "r", encoding="utf-8")
                subs = lovers.read().split(" _-_ ")
                try:
                    for el in random.sample(subs, i):
                        sentences.append(el)
                except ValueError:
                    print(
                        f"Sentences required: {i}, \nsentences in Pseudo total: {len(subs)}, Skipping...\n")
                    continue

            random.shuffle(sentences)

            print(len(sentences))
            with open(os.path.join(folder, new_filename), "w", encoding="utf-8") as output_file:
                output_file.write(" _-_ ".join(sentences))
                output_file.close()


if __name__ == "__main__":

    # FIRST: REMOVE Law and Lov from folder
    data_type = 'Alt'
    make_dataset(data_type)

    # Split Laws into 21 chunks
    laws_file = open(
        f"configure{data_type}/Pla_Law.txt", "r", encoding='utf-8').read()
    split_Laws(laws_file, data_type, chunk_size=5000)
    exit(0)

    # Obfuscate 1 Law with different percentage of a Pseudo-Plato
    # PsPla_Lov chosen has consistently most distant from Laws in previous tests
    percentage = [0.9, 0.8, 0.7, 0.5]
    for p in percentage:
        obfuscate(folder=f'configure{data_type}', p=p)
    exit(0)

    # DELETE PsPla_Lov and Laws used in obfuscation
    df = pd.read_csv(f"{data_type}dataset_unobfusc.csv")
    for file in glob.glob(f"configure{data_type}/*.txt"):
        df = insert_data(df, file)
    df['label'] = df['title'].apply(
        lambda x: 1 if x == 'Late' or x.startswith('Law') else 0)

    df.to_csv(f"{data_type}dataset_obfuscate.csv", index=False)
