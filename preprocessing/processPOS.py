import time as t
from cltk import NLP
import time as t
import glob
import re


def annotate_pos(txt, filename, new_folder, morpho=True):
    start_time = t.time()
    nlp = NLP(language="grc", suppress_banner=True)
    print("NLPing...", filename)
    doc = nlp.analyze(text=txt)

    if morpho:
        # Extract basic NOUN/VERB-morphology sequences
        morpho = []
        for word in doc.words:
            punkt = re.search('\.|;|·', word.string)
            morpho.append((word.lemma, word.upos))
            if punkt:
                morpho.append(("_", "_"))
        masked_string = process_morpho(morpho)
    else:
        # Extract complex syntactic sequences
        syntax = []
        for word in doc.words:
            punkt = re.search('\.|;|·', word.string)
            syntax.append((word.lemma, word.upos,
                           word.dependency_relation))
            if punkt:
                syntax.append(("_", "_"))
        masked_string = process_syntax(syntax)

    masked_string = re.sub("--", "-", masked_string)

    end_time = t.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")

    # save syntax to text file
    new_name = filename.split('/')[-1]

    with open(f"{new_folder}/{new_name}", "w", encoding='utf-8') as f:
        f.write(str(masked_string))
        print(f"Parsed {new_name}")

    def process_syntax(truples):
        result = []
        for lemma, upos, dep in truples:
            if upos in ['PROPN', 'NOUN', 'VERB', 'PRON']:
                result.append("%s-%s" % (upos, dep))
            else:
                result.append("%s-%s" % (lemma, dep))

        return ' '.join(result)

    def process_morpho(tuples):
        result = []
        for lemma, upos in tuples:
            if upos in ['PROPN', 'NOUN', 'AUX', 'VERB', 'PRON']:
                result.append(upos)
            else:
                result.append(lemma)

        return ' '.join(result)


if __name__ == "__main__":
    f = 'data/complexPARSED/to_parse'
    # os.makedirs("allinstancesPASRED", exist_ok=True)
    for file in glob.glob(f + '/*.txt'):
        content = open(file, 'r').read()
        annotate_pos(content, file, simple=True)
        print("\n\nDone!")
