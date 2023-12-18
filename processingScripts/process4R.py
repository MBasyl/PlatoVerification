import time as t
import glob
import re
import unicodedata
from cleanCSV import sentenceCapitalizer
# pre-process corpus for R stylo


def replace_named_entities(string1):
    """reads lists of named entities and substitutes them in text with '*'
    """
    named_entities = open(
        'outputs/processing_lists/NER_tokens.txt', 'r').read().splitlines()

    pattern = re.compile(r'\w+')
    cleanstring = ['*' if any(m.group() in named_entities for m in pattern.finditer(
        s)) else s for s in string1.split(" ")]

    return ' '.join(cleanstring)


def strip_accents(s):
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def no_NERaccents(text):
    # remove accents
    # combinazione = ["ἄἅἂἃἆἇἔἕἒἓἕἤἥἢἣἦἧἴἵἲἳἶἷὄοὅὂὃὔὕὒὓὖὗὤὥὢὣὦὧ"]
    no_NER = replace_named_entities(text)
    no_tags = re.sub(r'(\.\s..\.)|(\.\s...\.)', '', no_NER)
    # no_accents = strip_accents(no_NER)
    caps = sentenceCapitalizer(no_tags)
    final_clean = re.sub(r'\s+', ' ', caps)  # remove extra whitespace

    return final_clean


def clean_corpus(folder):
    print(f'cleaning text and subbing NER...')
    # place txt files in directory Rcorpus AFTER CLEANING
    for file in glob.glob(folder + '/*.txt'):

        # clean text
        text = open(file, 'r').read()
        text = re.sub(r'\s\.\s', ' ', text)  # remove extra whitespace

        # skip text if contains less than 2k
        if len(text.split(' ')) > 2000:
            clean_text = no_NERaccents(text)
        # lemmatize and save in Rcorpus/lemmata
            # lemmatizing(clean_text)  # POS??
        # save clean file in Rcorpus/plain
            with open(f'platoCorpus/PLAIN/{file.split("/")[-1]}', 'w') as f:
                f.write(clean_text)
                f.close()
    print("Files done in platoinstances!\n\n")
    return


if __name__ == '__main__':
    folder = 'data/processedXML/plato'
    clean_corpus(folder)
    # folder = 'PsPlaProcess'
