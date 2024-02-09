from cltk.tag import ner
import time as t
import glob
import re
import os


def NER_list(folder):

    print('making list of NER...')
    start_time = t.time()
    ListNER = []
    for file in glob.glob(folder + '/*.txt'):
        f = open(file, 'r').read()
        ListNER.append(create_ner(f))

    end_time = t.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")
    # exec time: 33 min ca
    flat_list = [item for sublist in ListNER for item in sublist]
    print("length: ", len(flat_list))
    print("lenght SET: ", len(set(flat_list)))
    # create CSV doc
    write_ALLner(flat_list, folder)

    def create_ner(text):
        # make list of NER for each row
        ner_list = ner.tag_ner("grc", text)
        # Filter elements with 'Entity'
        entity_list = [element[0] for element in ner_list if len(
            element) == 2 and element[1] == 'Entity']
        return entity_list

    def write_ALLner(flat_list, folder):
        # sort alphabetically
        word_count_ordered = sorted(set(flat_list))

        with open(f'outputs/processing_lists/newNER_tokens.txt', 'w', newline='') as file:
            for el in word_count_ordered:
                file.write(el + '\n')

    return


def replace_named_entities(string1):
    """reads lists of named entities and substitutes them in text with '*'
    """
    named_entities = open(
        'outputs/processing_lists/NER_tokens.txt', 'r').read().splitlines()

    pattern = re.compile(r'\w+')
    cleanstring = ['*' if any(m.group() in named_entities for m in pattern.finditer(
        s)) else s for s in string1.split(" ")]

    return ' '.join(cleanstring)


def main(f, out):
    for file in glob.glob(f + '/*.txt'):
        # sub NER
        text = open(file, 'r').read()
        clean_text = replace_named_entities(text)
        # clean_text = re.sub("·", ".", clean_text)
        # clean_text = re.sub("ʼ", "", clean_text)
        filename = file.split("/")[-1]
        input_file_path = os.path.join(out, filename)
        with open(input_file_path, 'w') as f:
            print("number of words in", filename, len(clean_text.split()))
            f.write(clean_text)
            f.close()


if __name__ == "__main__":
    f = 'data/processedXML'
    # NER_list(f)
    outpath = 'data/PLAIN'
    # apply only replace_named_entities() on a dir
    main(f, outpath)
