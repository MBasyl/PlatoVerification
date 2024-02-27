import pandas as pd
import glob
import re
import os
import utils
import shutil


def combine_new_profiles(input_dir):
    """Combine files according to chronological/genre order for profile processing."""
    output_dir = 'Profiles'
    os.makedirs(output_dir, exist_ok=True)

    new_late = ["Tht", "Sop", "Sta", "Phi", "2Ph", 'Rep']
    new_early = ["Men", "Sym", "1Ph", "Apo", "Cri", "Ion", "Lac", "Lys", "Cha",
                 "1Eu", "2Eu", "Him", "Cra", "Gor", "Pro"]
    dialogues = ['Apo', 'Eco', 'Hie', 'Mem', 'Sym']
    histories = ['Ana', 'Cyr', 'Age', 'Hel']

    for file in glob.glob(input_dir + '/*.txt'):
        filename = os.path.basename(file)
        author = filename.split("_")[0]
        title = filename.split("_")[1].split(".")[0]

        if author == 'Pla':
            if title in new_late:
                output_file_path = os.path.join(output_dir, 'Pla_Late.txt')
                with open(output_file_path, 'a') as output_file, open(file, 'r') as input_file:
                    content = input_file.read()
                    output_file.write(content)
            elif title in new_early:
                output_file_path = os.path.join(output_dir, 'Pla_Early.txt')
                with open(output_file_path, 'a') as output_file, open(file, 'r') as input_file:
                    content = input_file.read()
                    output_file.write(content)
            else:
                print("This title has not been included in any Pla Profile:", title)
                move_file = os.path.join(output_dir, filename)
                shutil.move(file, move_file)
        elif author == 'Xen':
            if title in dialogues:
                output_file_path = os.path.join(
                    output_dir, 'Xen_Dialogues.txt')
                with open(output_file_path, 'a') as output_file, open(file, 'r') as input_file:
                    content = input_file.read()
                    output_file.write(content)
            elif title in histories:
                output_file_path = os.path.join(
                    output_dir, 'Xen_Histories.txt')
                with open(output_file_path, 'a') as output_file, open(file, 'r') as input_file:
                    content = input_file.read()
                    output_file.write(content)
            else:
                output_file_path = os.path.join(output_dir, 'Xen_Treatise.txt')
                with open(output_file_path, 'a') as output_file, open(file, 'r') as input_file:
                    content = input_file.read()
                    output_file.write(content)
        else:
            print("This title has not been included in any Profile:", title)
            move_file = os.path.join(output_dir, filename)
            shutil.move(file, move_file)


def combine_profiles(input_dir):
    """combine files according to chronological/genre order for profile processing."""
    output_dir = 'Profiles'
    os.makedirs(output_dir, exist_ok=True)
    # Plato FIRST Chronological Profiles
    early = ["Apo", "Cri", "Ion", "Lac", "Lys", "Cha",
             "1Eu", "2Eu", "Him", "Cra", "Gor", "Pro"]
    mature = ["Men", "Sym", "Rep", "1Ph", "2Ph"]
    late = ["Par", "Tht", "Sop", "Sta", "Phi", "Tim", "Crt", "Law"]
    # Xenophon GENRE Profiles
    dialogues = ['Apo', 'Eco', 'Hie', 'Mem', 'Sym']
    histories = ['Ana', 'Cyr', 'Age', 'Hel']

    for file in glob.glob(input_dir + '/*.txt'):
        filename = file.split("/")[-1]
        author = filename.split("_")[0]
        title = filename.split("_")[1].split(".")[0]
        if author == 'Pla':
            if title in early:
                output_file_path = os.path.join(output_dir, "Pla_Early.txt")
                # Open the new file in append AND the corresponding input file with the current title
                with open(output_file_path, 'a') as output_file, open(file, 'r') as input_file:
                    content = input_file.read()
                    output_file.write(content)
            elif title in mature:
                output_file_path = os.path.join(output_dir, "Pla_Mature.txt")
                with open(output_file_path, 'a') as output_file, open(file, 'r') as input_file:
                    content = input_file.read()
                    output_file.write(content)
            elif title in late:
                output_file_path = os.path.join(output_dir, "Pla_Late.txt")
                with open(output_file_path, 'a') as output_file, open(file, 'r') as input_file:
                    content = input_file.read()
                    output_file.write(content)
            else:
                print("This title has not been included in any Pla Profile:", title)
                move_file = os.path.join(output_dir, filename)
                shutil.move(file, move_file)
        elif author == 'Xen':
            if title in dialogues:
                output_file_path = os.path.join(
                    output_dir, 'Xen_Dialogues.txt')
                with open(output_file_path, 'a') as output_file, open(file, 'r') as input_file:
                    content = input_file.read()
                    output_file.write(content)
            elif title in histories:
                output_file_path = os.path.join(
                    output_dir, 'Xen_Histories.txt')
                with open(output_file_path, 'a') as output_file, open(file, 'r') as input_file:
                    content = input_file.read()
                    output_file.write(content)
            else:
                output_file_path = os.path.join(output_dir, 'Xen_Treatise.txt')
                with open(output_file_path, 'a') as output_file, open(file, 'r') as input_file:
                    content = input_file.read()
                    output_file.write(content)
        else:
            print("This title has not been included in any Profile:", title)
            move_file = os.path.join(output_dir, filename)
            shutil.move(file, move_file)


def make_profile_dictionary(folder, max_length=4300):
    """Make dictionary of AUTHOR-PROFILES"""
    authors = []
    text_name = []
    contents = []
    for file in glob.glob(folder + "/*.txt"):
        content = open(file, "r", encoding="utf-8").read()
        words = re.findall(r'\S+', content)
        author, text = file.split("/")[-1].split(".")[0].split("_")
        chunks = utils.chunk_text(words, max_length, exact=False)
        for c in chunks:
            contents.append(' '.join(c))
            text_name.append(text)
            authors.append(author)
    print(text_name)
    print(len(text_name))
    return authors, text_name, contents


def make_profile_dataset(dataset, output_path):
    df = pd.DataFrame(dataset)
    print(len(df.title.tolist()))
    # Apply the function to count words and filter rows
    df = df[df['text'].apply(utils.count_words) >= 2000]
    # Resetting index after dropping rows
    df.reset_index(drop=True, inplace=True)
    print(len(df.title.tolist()))
    print("\n\nAssigning positive label to Law and Late profiles")
    df['binary_label'] = df['title'].apply(
        lambda x: 1 if x in ['Law', 'Late', 'Law#test', 'Late#test'] else -1)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    # !! Remember: Change filename to VII letter and Remove Menexenus

    # Plato FIRST Chronological Profiles
    # combine_profiles(input_dir)

    # Plato SECOND Chronological Profiles
    folder = 'alternativeParsed'  # [PARSED/PLAIN]
    # combine_new_profiles(folder)
    exit(0)
    authors, text_name, contents = make_profile_dictionary(
        folder='Profiles', max_length=4300)
    # Manually assign #test to 20% of titles (ca 30)
    exit(0)
    output_path = 'Plain4kNewprofile.csv'
    dataset = {
        'author': authors,
        'title': ['2Al', '2Al#test', 'Dialogues', 'Dialogues#test', 'Dialogues', 'Dialogues', 'Dialogues', 'Dialogues',
                  'Dialogues', 'Dialogues#test', 'Dialogues', 'Dialogues', 'Dialogues', 'Dialogues#test', 'Dialogues',
                  'Dialogues', 'Dialogues', 'Dialogues#test', 'Dialogues', 'Dialogues', 'Crt', 'Crt#test', 'Hip', '1Al',
                  '1Al#test', '1Al', 'The', 'Epi#test', 'Epi', 'Treatise', 'Treatise#test', 'Treatise', 'Treatise', 'Treatise',
                  'Treatise#test', 'Treatise', 'Treatise', 'Lov', 'Late#test', 'Late', 'Late', 'Late', 'Late#test', 'Late', 'Late',
                  'Late', 'Late#test', 'Late', 'Late', 'Late', 'Late', 'Late', 'Late', 'Late', 'Late#test', 'Late', 'Late',
                  'Late', 'Late', 'Late', 'Late#test', 'Late', 'Late', 'Late', 'Late', 'Late', 'Late', 'Late', 'Late',
                  'Late', 'Late', 'Late', 'Late', 'Late', 'Late', 'Late', 'Late#test', 'Late', 'Late', 'Late', 'Late',
                  'Late', 'Late', 'Par', 'Par#test', 'Par', 'Par', 'Min', 'VII', 'VII', 'VII', 'Law#test', 'Law', 'Law', 'Law',
                  'Law', 'Law', 'Law#test', 'Law', 'Law', 'Law', 'Law', 'Law#test', 'Law', 'Law', 'Law', 'Law', 'Law', 'Law',
                  'Law', 'Law', 'Law', 'Law', 'Law', 'Law#test', 'Law', 'Law', 'Histories', 'Histories#test', 'Histories',
                  'Histories', 'Histories', 'Histories', 'Histories', 'Histories', 'Histories', 'Histories',
                  'Histories', 'Histories#test', 'Histories', 'Histories', 'Histories', 'Histories', 'Histories',
                  'Histories', 'Histories', 'Histories', 'Histories', 'Histories#test', 'Histories', 'Histories',
                  'Histories', 'Histories#test', 'Histories', 'Histories', 'Histories', 'Histories', 'Histories',
                  'Histories', 'Histories', 'Histories', 'Histories#test', 'Histories', 'Histories', 'Histories',
                  'Histories', 'Histories#test', 'Histories', 'Histories', 'Histories', 'Histories', 'Histories',
                  'Histories', 'Histories', 'Histories', 'Histories', 'Histories#test', 'Histories', 'Histories',
                  'HiM', 'HiM#test', 'HiM', 'Early', 'Early', 'Early#test', 'Early', 'Early', 'Early', 'Early', 'Early',
                  'Early', 'Early', 'Early', 'Early', 'Early', 'Early#test', 'Early', 'Early', 'Early', 'Early', 'Early',
                  'Early', 'Early#test', 'Early', 'Early', 'Early', 'Early', 'Early', 'Early#test', 'Early', 'Early', 'Early',
                  'Early', 'Early#test', 'Early', 'Early', 'Early', 'Early', 'Early', 'Early', 'Early', 'Early#test', 'Early',
                  'Early', 'Early', 'Tim', 'Tim#test', 'Tim', 'Tim', 'Tim', 'Tim'],
        'text': contents}
    make_profile_dataset(dataset, output_path)
