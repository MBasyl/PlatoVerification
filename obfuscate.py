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


def make_validation_set(folder: str) -> pd.DataFrame:
    """"PARSED or PLAIN"""
    authors, text_name, contents = utils.make_dataset_dictionary(
        folder=folder, max_length=4300)
    if 'Law' in text_name:
        print("WARNING: did not remove full Laws from data")
    dataset = {
        'author': authors,
        'title': text_name,
        'text': contents}
    df = pd.DataFrame(dataset)
    # df = insert_data(df, f"configure{data_type}/Disputed_VII.txt")

    return df


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
        if author == 'Pla' and title in ['Law3#val', 'Law7#val', 'Law10#val', 'Law15#val', 'Law20#val']:
            file = open(os.path.join(folder, filename),
                        "r", encoding="utf-8")
            sentences = file.read().split(". ")
            print(filename, "number of sentences:", len(sentences))
            num_to_remove = round(p * len(sentences))

            # Remove random sentences
            for _ in range(num_to_remove):
                sentences.pop(random.randint(0, len(sentences)-1))

            # Inject random sentences
            curr_percentage = str(p).split(".")[-1]
            new_filename = f'PsPla_{title}#{curr_percentage}.txt'
            with open(f"configure{data_type}/PsPla_2Al.txt", "r", encoding="utf-8") as lovers_file:
                lovers_sentences = lovers_file.read().split(". ")
                try:
                    for _ in range(num_to_remove):
                        random_sentence = random.choice(lovers_sentences)
                        sentences.append(random_sentence)
                except ValueError:
                    print(
                        f"Sentences required: {num_to_remove}, sentences in 'Lovers' document: {len(lovers_sentences)}, Skipping...")
                    continue

            random.shuffle(sentences)

            with open(os.path.join(folder, new_filename), "w", encoding="utf-8") as output_file:
                output_file.write(". ".join(sentences))
                output_file.close()


if __name__ == "__main__":

    # FIRST: REMOVE Law and Lov from folder
    data_type = 'PARSED'
    # make_dataset(data_type)

    # Split Laws into 21 chunks
    # laws_file = open(
    #     f"configure{data_type}/Pla_Law.txt", "r", encoding='utf-8').read()
    # split_Laws(laws_file, data_type, chunk_size=4500)
    # exit(0)

    # Obfuscate 1 Law with different percentage of a Pseudo-Plato
    # PsPla_2Al chosen has consistently most distant from Laws in previous tests
    # percentage = [0.9, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2]
    # for p in percentage:
    #     obfuscate(folder=f'configure{data_type}', p=p)

    # PLACE PsPla_Lov and DELETE full Laws
    df = make_validation_set(data_type)
    for file in glob.glob(f"configure{data_type}/*.txt"):
        df = insert_data(df, file)
    df['label'] = df['author'].apply(
        lambda x: 1 if x in ['Disputed', 'Pla'] else 0)

    df.to_csv(f"{data_type}_validation.csv", index=False)
