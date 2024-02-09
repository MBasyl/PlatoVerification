# write a class to pre-process dataset
from cltk.tag import ner
from cltk import NLP
from bs4 import BeautifulSoup
import utils
import time as t
import glob
import re
import os


class DataPreprocessor:
    """Takes a directory of XML/TXT files and performs NLP as needed. """

    def __init__(self):
        pass

    @staticmethod
    def process_syntax(triples):
        result = []
        for lemma, upos, dep in triples:
            if upos in ['PROPN', 'NOUN', 'VERB', 'PRON']:
                result.append("%s-%s" % (upos, dep))
            else:
                result.append("%s-%s" % (lemma, dep))
        return ' '.join(result)

    @staticmethod
    def process_morpho(tuples):
        result = []
        for lemma, upos in tuples:
            if upos in ['PROPN', 'NOUN', 'AUX', 'VERB', 'PRON']:
                result.append(upos)
            else:
                result.append(lemma)
        return ' '.join(result)

    @staticmethod
    def annotate_pos(txt, filename, new_folder, morpho=True):
        start_time = t.time()
        nlp = NLP(language="grc", suppress_banner=True)
        print("NLPing...", filename)
        doc = nlp.analyze(text=txt)

        if morpho:
            # Extract basic NOUN/VERB-morphology sequences
            morpho_list = []
            for word in doc.words:
                punkt = re.search('\.|;|·', word.string)
                morpho_list.append((word.lemma, word.upos))
                if punkt:
                    morpho_list.append(("_", "_"))
            masked_string = DataPreprocessor.process_morpho(morpho_list)
        else:
            # Extract complex syntactic sequences
            syntax_list = []
            for word in doc.words:
                punkt = re.search('\.|;|·', word.string)
                syntax_list.append((word.lemma, word.upos,
                                    word.dependency_relation))
                if punkt:
                    syntax_list.append(("_", "_", "_"))
            masked_string = DataPreprocessor.process_syntax(syntax_list)

        masked_string = re.sub("--", "-", masked_string)

        end_time = t.time()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time} seconds")

        new_name = filename.split('/')[-1]

        with open(f"{new_folder}/{new_name}", "w", encoding='utf-8') as f:
            f.write(str(masked_string))
            print(f"Parsed {new_name}")

    @classmethod
    def make_parsed(self, folder, morpho: bool, outpath='data/PARSED'):
        os.makedirs(outpath, exist_ok=True)
        for file in glob.glob(folder + '/*.txt'):
            content = open(file, 'r').read()
            self.annotate_pos(content, file, outpath, morpho)
            print("\n\nDone!")

    @classmethod
    def make_plain(self, folder, outpath='data/PLAIN'):
        os.makedirs(outpath, exist_ok=True)
        for file in glob.glob(folder + '/*.txt'):
            # sub NER
            text = open(file, 'r').read()
            clean_text = utils.replace_named_entities(text)
            # clean_text = re.sub("·", ".", clean_text)
            # clean_text = re.sub("ʼ", "", clean_text)
            filename = file.split("/")[-1]
            input_file_path = os.path.join(outpath, filename)
            with open(input_file_path, 'w') as f:
                print("number of words in", filename, len(clean_text.split()))
                f.write(clean_text)

    @staticmethod
    def create_ner(text):
        # make list of NER for each row
        ner_list = ner.tag_ner("grc", text)
        # Filter elements with 'Entity'
        entity_list = [element[0] for element in ner_list if len(
            element) == 2 and element[1] == 'Entity']
        return entity_list

    @staticmethod
    def write_ALLner(flat_list, new_folder):
        # sort alphabetically
        word_count_ordered = sorted(set(flat_list))
        with open(os.path.join(new_folder, 'NER_tokens.txt'), 'w', newline='') as file:
            file.write("Number of NER: " + str(len(flat_list)) + "\n")
            for el in word_count_ordered:
                file.write(el + '\n')

    @classmethod
    def NER_list(self, folder):
        start_time = t.time()
        ListNER = []
        for file in glob.glob(folder + '/*.txt'):
            print(f'making list of NER from {file}...')
            content = open(file, 'r').read()
            ListNER.append(self.create_ner(content))

        end_time = t.time()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time} seconds")
        # exec time: 33 min ca
        flat_list = [item for sublist in ListNER for item in sublist]
        # create CSV doc
        # print("length: ", len(flat_list))
        output_folder = "outputs"
        os.makedirs(output_folder, exist_ok=True)
        # call write_ALLner with correct output folder
        self.write_ALLner(flat_list, output_folder)

    @classmethod
    def subset_dataset(self, threshold, directory_path, subdirectory_name='short_files'):
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
                    print(
                        f"Moved '{filename}' to '{subdirectory_name}' with ", word_count)

    @staticmethod
    def clean_text(content):
        """Clean text by removing non-Greek characters and symbols."""
        clean_content = re.sub(r"[a-zA-Z\d]", " ", content)
        clean_content = re.sub(r"[><\[\]%*:,()‹›᾿〈⟩⟨〉\-—]", " ", clean_content)
        clean_content = re.sub(r'(\.\s..\.)|(\.\s...\.)', ' ', clean_content)
        clean_content = re.sub(r'(\(ς\))', ' ', clean_content)
        clean_content = re.sub(r'·', '.', clean_content)
        clean_content = re.sub(r'\s+', ' ', clean_content)
        clean_content = re.sub("[\t\n]", " ", clean_content)
        clean_content = re.sub(r'\s\.\s', '. ', clean_content)
        clean_content = re.sub(r'\s+', ' ', clean_content)
        return clean_content

    @staticmethod
    def process_letters(data, new_folder):
        """Pre-process Letters file and separate into multiple files."""
        with open(data, 'r') as tei:
            soup = BeautifulSoup(tei, features="xml")
            # div type="textpart" ="letter" n="1"
            for letter_num in range(1, 14):
                letter = soup.find('div', {'n': f'{letter_num}'})
                if letter is not None:
                    for quote in letter.find_all('quote'):
                        quote.decompose()
                    for note in letter.find_all('note'):
                        note.decompose()
                    content = ' '.join([p.text.strip()
                                       for p in letter.find_all('p')])
                    final_content = DataPreprocessor.clean_text(content)
                    output_file_path = f"{new_folder}/PsPla_{letter_num}.txt"
                    with open(output_file_path, 'w', encoding='utf-8') as output_file:
                        output_file.write(final_content)

    @staticmethod
    def parse_tei(data, new_folder):
        """Pre-process files."""
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
            for tag in ['label', 'quote', 'note']:
                [elem.decompose() for elem in body.find_all(tag)]
            content = ' '.join([p.text.strip() for p in body.find_all('p')])
            final_content = DataPreprocessor.clean_text(content)
            # write txt file with title, author, and content
            with open(f"{new_folder}/{new_title}.txt", "w") as f:
                f.write(final_content)
        return [author, title]

    @staticmethod
    def process_tei(folder, new_folder):
        """Iterate pre-processing tei over all files in rawCorpus."""
        author_list = []
        for file in glob.glob(folder + '/*.xml'):
            if file == "rawCorpus/tlg0059.tlg036.perseus-grc2.xml":
                continue
            author_list.append(DataPreprocessor.parse_tei(file, new_folder))
        return author_list

    @classmethod
    def xml_files(self, folder):
        """Process XML files."""
        print("Creating processedXML folder if not exists")
        os.makedirs("processedXML", exist_ok=True)
        new_folder = "processedXML"
        author_list = self.process_tei(folder, new_folder)
        print("\n\nTotal: ", len(author_list), "\nAuthor List:")
        for author in author_list:
            print(author)
        self.process_letters(
            f"{folder}/tlg0059.tlg036.perseus-grc2.xml", new_folder)
        print(
            "\nTake a moment to change FILE NAMES for Disputed (and cardinals for Letters)")
