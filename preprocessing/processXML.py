import glob
from bs4 import BeautifulSoup
import re
import os
# spirito = ["ἀἁαἐἑεἠἡηἰἱιὀοὁὐὑυὠὡω"]
# accento = ["άὰᾶέὲήὴῆίὶῖόὸύὺῦώὼῶ"]
# combinazione = ["ἄἅἂἃἆἇἔἕἒἓἕἤἥἢἣἦἧἴἵἲἳἶἷὄοὅὂὃὔὕὒὓὖὗὤὥὢὣὦὧ"]


def subset_Dataset(directory_path, threshold=2000, subdirectory_name='short_files'):
    # Create a subdirectory if it doesn't exist
    subdirectory_path = os.path.join(directory_path, subdirectory_name)
    os.makedirs(subdirectory_path, exist_ok=True)

    # Iterate through files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Check if it's a file and not the subdirectory itself
        if os.path.isfile(file_path) and filename != subdirectory_name and filename.endswith('.txt'):
            # Read the file and count words
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                word_count = len(content.split())

            # Move the file if it has less than the threshold words
            if word_count < threshold:
                new_path = os.path.join(subdirectory_path, filename)
                os.rename(file_path, new_path)
                print(f"Moved '{filename}' to '{subdirectory_name}'.")


def clean(content):
    # remove all non-greek characters
    clean_content = re.sub(r"[a-zA-Z\d]", " ", content)
    # remove commas and symbols (not punkt)
    # remove Socrate's (ΣΩ) dialogue tag
    content = re.sub(r"[><\[\]%*:,()‹›᾿〈⟩⟨〉\-—]", " ", clean_content)
    content = re.sub(r'(\.\s..\.)|(\.\s...\.)', ' ', content)
    content = re.sub(r'(\(ς\))', ' ', content)
    content = re.sub(r'·', '.', content)
    # remove multiple spaces
    content = re.sub(r'\s+', ' ', content)
    content = re.sub("[\t\n]", " ", content)
    text = re.sub(r'\s\.\s', '. ', content)
    final_content = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    return final_content


def processLetters(data, new_folder):
    """pre-process Letters file and separate in multiple files"""

    # read tei file
    with open(data, 'r') as tei:
        soup = BeautifulSoup(tei, features="xml")
        # div type="textpart" ="letter" n="1"
        for letter_num in range(1, 14):
            # letter_selector = f'div[type="textpart"][subtype="letter"][n="{letter_num}"]'
            letter = soup.find('div', {'n': f'{letter_num}'})

            if letter is not None:
                for quote in letter.find_all('quote'):
                    quote.decompose()
                for note in letter.find_all('note'):
                    note.decompose()
                content = ' '.join([p.text.strip()
                                    for p in letter.find_all('p')])

                final_content = clean(content)

                output_file_path = f"{new_folder}/PsPla_{letter_num}.txt"
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(final_content)
                    output_file.close()

    return


def parse_tei(data, new_folder):
    """pre-process files"""
    # read tei file
    with open(data, 'r') as tei:
        soup = BeautifulSoup(tei, features="xml")
        title = soup.title.getText()
        author = soup.author.getText()
        if author == "PsPlato":
            new_title = 'PsPla_' + str(title)[:3]
        else:
            # change file name to first 3 characters of author and title
            new_title = str(author)[:3] + '_' + str(title)[:3]

        # strip=True ensures no newlines from the original XML document
        # content = soup.body.getText(separator=' ', strip=True)
        body = soup.find('body')
        for lab in body.find_all('label'):
            lab.decompose()
        for quote in body.find_all('quote'):
            quote.decompose()
        for note in body.find_all('note'):
            note.decompose()
        content = ' '.join([p.text.strip()
                            for p in body.find_all('p')])
        final_content = clean(content)

        os.makedirs(new_folder, exist_ok=True)
        # write txt file with title, author, and content
        with open(f"{new_folder}/{new_title}.txt", "w") as f:
            f.write(final_content)
            f.close()
    return [author, title]


def process_files(folder, new_folder):
    """iterate pre-processing tei over all files in rawCorpus. 
    Make ListNER"""
    print(f"parsing_tei...")
    authorList = []
    for file in glob.glob(folder + '/*.xml'):
        if file == "rawCorpus/tlg0059.tlg036.perseus-grc2.xml":
            continue

        authorList.append(parse_tei(file, new_folder))

    return authorList


if __name__ == "__main__":

    new_folder = 'data/processedXML'
    authorList = process_files("data/rawCorpus", new_folder)
    processLetters(
        "data/rawCorpus/tlg0059.tlg036.perseus-grc2.xml", new_folder)
    subset_Dataset(new_folder)
