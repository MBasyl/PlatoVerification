# Transform LAWS to our Disputed:
import random
import re
import os
from cltk import NLP
import processPOS as processpos
import processXML as processXML
import NERtokens as processner
from processPOS import process_tuples
from NERtokens import replace_named_entities


def annotate_pos(txt):
    nlp = NLP(language="grc", suppress_banner=True)
    print("NLPing...")
    doc = nlp.analyze(text=txt)
    # To extract basic NOUN/VERB-stopword sequences
    tuple_tags = []
    for word in doc.words:
        punkt = re.search('\.|;|·', word.string)
        tuple_tags.append((word.lemma, word.upos))
        if punkt:
            tuple_tags.append(("_", "_"))

    masked_string = process_tuples(tuple_tags)

    return masked_string


def chunk_text(input_text, chunk_size):
    words = re.findall(r'\S+', input_text)
    chunks = [words[i:i + chunk_size]
              for i in range(0, len(words), chunk_size)]

    chunks = [chunk for chunk in chunks if len(chunk) <= chunk_size]

    return chunks


def group_and_chunk(input_directory, output_dir, chunk_size=15000):
    """group single docs in author subdirectories"""
    # Process each text file
    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            title = filename.split(".")[0]
            input_file_path = os.path.join(input_directory, filename)
            file = open(input_file_path, "r", encoding="utf-8")
            content = file.read()
            chunks = chunk_text(content, chunk_size)

            # Write chunks to separate files
            for i, chunk in enumerate(chunks):
                output_file_path = os.path.join(
                    output_dir, f"{title}_{i + 1}.txt")
                with open(output_file_path, "w", encoding="utf-8") as output_file:
                    output_file.write(" ".join(chunk))


def extract_random_chunk(file_path, new_file, chunk_size=8000):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    print(len(content.split()))
    # Find all sentence-ending positions
    sentence_end_positions = [match.end()
                              for match in re.finditer(r'[.;]', content)]

    if not sentence_end_positions:
        raise ValueError(
            "No sentence-ending periods or question marks found in the document.")
    # max_position = max(sentence_end_positions) - chunk_size
    target = 557830  # sentence_end_positions.index(max_position)
    subset_random = sentence_end_positions[:target]
    # Randomly select a sentence-ending position to start the chunk
    start_position = random.choice(subset_random)

    # Find the nearest word boundary after the selected position
    start_boundary = content.rfind(' ', 0, start_position) + 1

    # Extract words from the selected position to meet the specified chunk size
    words = re.findall(r'\S+', content[start_boundary:])
    selected_chunk = ' '.join(words[:chunk_size])
    print(len(selected_chunk.split()))
    with open(new_file, "w", encoding="utf-8") as save_file:
        save_file.write(selected_chunk)


def obfuscate(input_dir, output_plain, output_parsed, p=0.10, pseudo=True):
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            try:
                n_chunk = filename.split("_")[2]
            except IndexError:
                continue
            file = open(os.path.join(input_dir, filename),
                        "r", encoding="utf-8")
            sentences = file.read().split(". ")
            print(filename, "number of words:", len(sentences))
            for i in range(round(p * len(sentences))):
                number_sentences = random.randint(0, len(sentences)-1)
                sentences.pop(number_sentences)

            #  inject random sentences
            if pseudo:
                new_filename = f'PsPla#{p}_Law-1Al_{n_chunk}'
                alcibiades = open("data/PLAIN/PsPla_1Al.txt",
                                  "r", encoding="utf-8")
                subs = alcibiades.read().split(". ")
                for el in random.sample(subs, i):
                    sentences.append(el)

            else:
                new_filename = f'PsPla#{p}_Law-Pol_{n_chunk}'
                aristotle = open("data/PLAIN/Ari_Pol.txt",
                                 "r", encoding="utf-8")
                subs = aristotle.read().split(". ")
                for el in random.sample(subs, i):
                    sentences.append(el)

            random.shuffle(sentences)

            # make file PLAIN
            PLAIN_text = replace_named_entities(". ".join(sentences))
            print(len(sentences))
            with open(os.path.join(output_plain, new_filename), "w", encoding="utf-8") as output_file:
                output_file.write(PLAIN_text)
                output_file.close()
            # make file PARSED
            PARSED_text = annotate_pos(". ".join(sentences))
            with open(os.path.join(output_parsed, new_filename), "w", encoding="utf-8") as output_file:
                output_file.write(PARSED_text)
                output_file.close()

    return


# process singular file: LAWS
input_dir = 'data/PlainLaws'
output_plain_dir = 'data/PlainLaws/PLAIN'
output_parsed_dir = 'data/PlainLaws/PARSED'

os.makedirs(output_plain_dir, exist_ok=True)
os.makedirs(output_parsed_dir, exist_ok=True)

# obfuscate(input_dir, output_plain_dir, output_parsed_dir, p=0.25, pseudo=True)
percentage = [0.1, 0.25, 0.5, 0.7, 0.85, 0.91]
ari = [True, False]
for per in percentage:
    for a in ari:
        obfuscate(input_dir, output_plain_dir,
                  output_parsed_dir, p=per, pseudo=a)

exit(0)
os.makedirs("data/PlainLaws", exist_ok=True)
# extract random chunk to add to Plato profile (place it in PLAIN)
extract_random_chunk(file_path, new_file, chunk_size=15000)
print("Now MANUALLY clip off hanging sentences")
exit(0)
content = open(new_file, 'r').read()
content = processpos.no_NERaccents(content)
processpos.annotate_pos(content, "Pla_Law.txt", simple=True)

exit(0)
outpath = 'data/PLAIN'
# apply only replace_named_entities() on a dir
processner.main("data/PlainLaws", outpath)
# cut up original Laws
output_dir = "data/PlainLaws"
group_and_chunk(file_path, output_dir, chunk_size=15000)
print("Now MANUALLY clip off hanging sentences")

exit(0)
input_dir = 'data/PlainLaws'
output_plain_dir = 'data/PlainLaws/PLAIN'
output_parsed_dir = 'data/PlainLaws/PARSED'
# obfuscate(input_dir, output_plain_dir, output_parsed_dir, p=0.25, pseudo=True)
percentage = [0.1, 0.25, 0.5, 0.7, 0.85, 0.91]
ari = [True, False]
for per in percentage:
    for a in ari:
        obfuscate(input_dir, output_plain_dir,
                  output_parsed_dir, p=per, pseudo=a)
