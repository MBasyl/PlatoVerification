import glob
from bs4 import BeautifulSoup
import re

# spirito = ["ἀἁαἐἑεἠἡηἰἱιὀοὁὐὑυὠὡω"]
# accento = ["άὰᾶέὲήὴῆίὶῖόὸύὺῦώὼῶ"]
# combinazione = ["ἄἅἂἃἆἇἔἕἒἓἕἤἥἢἣἦἧἴἵἲἳἶἷὄοὅὂὃὔὕὒὓὖὗὤὥὢὣὦὧ"]


def clean(content):
    # remove all non-greek characters
    clean_content = re.sub(r"[a-zA-Z\d]", "", content)
    # remove commas and symbols (not punkt)
    # remove Socrate's (ΣΩ) dialogue tag
    cont = re.sub(r"[><\[\]%*:,()‹›〈⟩⟨〉\-—]", "", clean_content)
    # remove multiple spaces
    final_content = re.sub("[\t\n]", " ", cont)
    return final_content


def processLetters(data):
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
                content = ' '.join([p.text.strip()
                                    for p in letter.find_all('p')])

                final_content = clean(content)

                output_file_path = f"rawCorpus/PsPla_let{letter_num}.txt"
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(f"{final_content}")
                    output_file.close()

                # print(
                #    f"Letter {letter_num} processed. Output written to {output_file_path}")

    return


def parse_tei(data):
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
        content = ' '.join([p.text.strip()
                            for p in body.find_all('p')])
        final_content = clean(content)

        # write txt file with title, author, and content
        with open(f"rawCorpus/{new_title}.txt", "w") as f:
            f.write(f"{final_content}")
            f.close()
    return [author, title]


def process_files(folder):
    """iterate pre-processing tei over all files in rawCorpus. 
    Make ListNER"""
    print(f"parsing_tei...")
    authorList = []
    for file in glob.glob(folder + '/*.xml'):
        if file == "rawCorpus/tlg0059.tlg036.perseus-grc2.xml":
            continue

        authorList.append(parse_tei(file))

    return authorList


if __name__ == "__main__":

    authorList = process_files("rawCorpus")
    processLetters("rawCorpus/tlg0059.tlg036.perseus-grc2.xml")
