import re
import pandas as pd
import unicodedata

# spirito = ["ἀἁαἐἑεἠἡηἰἱιὀοὁὐὑυὠὡω"]
# accento = ["άὰᾶέὲήὴῆίὶῖόὸύὺῦώὼῶ"]
# combinazione = ["ἄἅἂἃἆἇἔἕἒἓἕἤἥἢἣἦἧἴἵἲἳἶἷὄοὅὂὃὔὕὒὓὖὗὤὥὢὣὦὧ"]


def replace_named_entities(string1):
    """reads lists of named entities and substitutes them in text with 'x'
    """
    df = pd.read_csv('ListNER_count.csv')
    named_entities = df['Entity'].to_list()
    # lexicon = set(named_entities)
    pattern = re.compile(r'\w+')
    cleanstring = ['x' if any(m.group() in named_entities for m in pattern.finditer(
        s)) else s for s in string1.split(" ")]

    return cleanstring


def sentenceCapitalizer(text):
    # Split the text into sentences using regular expressions
    punkt = r'(?<=[.;·])\s+'
    sentences = re.split(punkt, text)
    # Capitalize the first letter of each sentence
    sentences = [sentence.capitalize(
    ) if sentence and sentence not in punkt else sentence for sentence in sentences]
    # Join the sentences back together
    string12 = ' '.join(sentences)

    return string12


def strip_accents(s):
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def caps_and_punkt(text):
    # Capitalize character after punctuation
    try:
        cap_text = sentenceCapitalizer(text)
    except IndexError:
        print("An IndexError occurred, and the text processing is stopped.")

    punkt = '[.;·]'
    # no_punkt = re.sub(punkt, '', cap_text)  # remove all punctuation
    # remove accents
    # no_punkt_accents = strip_accents(no_punkt) # combinazione = ["ἄἅἂἃἆἇἔἕἒἓἕἤἥἢἣἦἧἴἵἲἳἶἷὄοὅὂὃὔὕὒὓὖὗὤὥὢὣὦὧ"]
    final_clean = re.sub(r'\s+', ' ', cap_text)  # remove extra whitespace

    return final_clean


def main(file, column_name):
    print("\ncleaning LEMMATA content in Dataframe...\n\n")
    df = pd.read_csv(file)
    # Capitalize, remove '[.;·]' & accents
    df[column_name] = df[column_name].apply(caps_and_punkt)
    print("!!Attention! Have not removed punkt or accents yet")
    # replace NER lemmas with 'x'
    df[column_name] = df[column_name].apply(replace_named_entities)

    df.to_csv(f'{file}_clean.csv', index=False)

    return


if __name__ == "__main__":

    file = 'smallLemmaDataset.csv'
    df = main(file, 'lemmata')
