import time as t
from cltk import NLP
import time as t
import glob
import re
import os


def process_triples(tuples):
    result = []
    for string, pos, upos in tuples:
        if upos in ['PROPN', 'NOUN', 'AUX', 'VERB', 'PRON']:
            result.append(upos)
        else:
            result.append(string)

    return ' '.join(result)


def process_catruplets(tuples):
    result = []
    for string, pos, upos, dep, gov in tuples:
        if upos in ['PROPN', 'NOUN', 'VERB', 'PRON']:
            result.append("%s-%s-%s" % (upos, dep, gov))
        else:
            result.append("%s-%s-%s" % (string, dep, gov))

    return ' '.join(result)


def process_tuples(tuples):
    result = []
    for lemma, upos in tuples:
        if upos in ['PROPN', 'NOUN', 'AUX', 'VERB', 'PRON']:
            result.append(upos)
        else:
            result.append(lemma)

    return ' '.join(result)


def lemmatize(txt, filename, simple=True):
    start_time = t.time()
    nlp = NLP(language="grc", suppress_banner=True)
    print("NLPing...", filename)
    doc = nlp.analyze(text=txt)

    # To extract complex morpho-syntactic sequences
    morpho_syn = []

    # for word in doc.words:
    # morpho_syn.append((word.string, word.pos, word.upos,
    #                  word.dependency_relation, word.governor))

    # To extract basic NOUN/VERB-stopword sequences
    tuple_tags = []
    for word in doc.words:
        punkt = re.search('\.|;|·', word.string)
        tuple_tags.append((word.lemma, word.upos))
        if punkt:
            tuple_tags.append(("_", "_"))

    # CHOOSE IF COMPLEX OR SIMPLE VERSION
    # if simple:
    masked_string = process_tuples(tuple_tags)

    # masked_string = process_catruplets(morpho_syn)
    # masked_string = re.sub("--", "-", masked_string)

    end_time = t.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")

    # save morpho_syn to text file
    new_name = filename.split('/')[-1]

    with open(f"platoCorpus/PARSED/{new_name}", "w", encoding='utf-8') as f:
        f.write(str(masked_string))
        print(f"written {new_name} to MorphoSynCorpus")

    return


def no_NERaccents(text):

    no_tags = re.sub(r'(\.\s..\.)|(\.\s...\.)', '', text)
    text = re.sub(r'\s\.\s', '. ', no_tags)
    final_clean = re.sub(r'\s+', ' ', text)  # remove extra whitespace

    return final_clean


if __name__ == "__main__":
    f = 'data/processedXML/plato'
    # os.makedirs("allinstancesPASRED", exist_ok=True)
    for file in glob.glob(f + '/*.txt'):
        content = open(file, 'r').read()
        content = no_NERaccents(content)
        lemmatize(content, file, simple=True)
        # masked_content would have a splitting rule such as:
        # "[^[A-Z]\s]+"

        print("\n\nDone!")
