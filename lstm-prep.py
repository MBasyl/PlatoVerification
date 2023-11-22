import pandas as pd
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from cltk.tokenizers.word import WordTokenizer
from cltk.lemmatize.grc import GreekBackoffLemmatizer


def chunk_by_words(string, word_count):
    words = string.split()  # Split the input string into a list of words
    chunks = [" ".join(words[i:i+word_count])
              for i in range(0, len(words), word_count)]
    #  remove empty chunks
    return [chunk for chunk in chunks if len(chunk.split()) == word_count]


def create_dataframe(file, chunk_length=500):
    with open(file, 'r') as f:
        data = f.read()
        f_name = f.name.split('/')[-1].split('.')[0]
        author = f_name.split('_')[0]
        title = f_name.split('_')[1]
        # get binary label based on author
        if author == 'Pla':
            label = 1
        else:
            label = 0
        text = chunk_by_words(data, chunk_length)
        make_dict = {'author': author, 'title': title,
                     'text': text, 'label': label}
        # create dataframe
        df = pd.DataFrame(make_dict)
    return df


def formatting():
    # iterate function over all files
    df_list = []
    for file in glob.glob('smallCorpus/*.txt'):
        df = create_dataframe(file, chunk_length=100)
        df_list.append(df)

    final_df = pd.concat(df_list)
    # drop author Din from df to have 7
    # final_df = final_df[final_df.author != 'Din']
    print("Finished formatting")
    #  save to csv
    final_df.to_csv('smallDataset.csv', index=False)


def lemmatise_text(text):
    # lemmatise text with cltk vs Byte Pair Encoding
    word_tokenizer = WordTokenizer('grc')
    tokens = word_tokenizer.tokenize(text)  # athenaeus_incipit.lower()
    lemmatizer = GreekBackoffLemmatizer()
    lemmata = lemmatizer.lemmatize(tokens)
    return lemmata


def tfidf(text):
    lemmatized_text = lemmatise_text(text)
    # apply tf-idf on text
    tfidf = TfidfVectorizer()
    tfidf.fit(lemmatized_text)
    # transform text to tf-idf
    tfidf_matrix = tfidf.transform(df.text)
    return


if __name__ == '__main__':
    formatting()
    df = pd.read_csv('smallDataset.csv')
    # get information on df
    print(df.author.value_counts())
    print(df.label.value_counts())
