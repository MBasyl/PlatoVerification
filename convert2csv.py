import pandas as pd
import glob
from cltk import NLP
from sklearn.preprocessing import LabelEncoder
import time as t


def chunk_by_words(string, word_count):
    words = string.split()  # Split the input string into a list of words
    chunks = [" ".join(words[i:i+word_count])
              for i in range(0, len(words), word_count)]
    #  remove empty chunks
    return [chunk for chunk in chunks if len(chunk.split()) == word_count]


def create_dataframe(file, chunk_length=500):
    #  read txt file
    with open(file, 'r') as f:
        data = f.read()
        # get 'Author' from file name
        f_name = f.name.split('/')[-1].split('.')[0]
        author = f_name.split('_')[0]
        title = f_name.split('_')[1]
        # get binary label based on author
        if author == 'Pla':
            label = 1
        else:
            label = 0
        text = chunk_by_words(data, chunk_length)
        # create dictionary
        make_dict = {'author': author, 'title': title,
                     'text': text, 'binary_label': label}
        # create dataframe
        df = pd.DataFrame(make_dict)

    # print(df.head())
    return df


def lemmatize(df):
    # Exec time: 50' on 2015 Macbook Pro
    start_time = t.time()
    nlp = NLP(language="grc", suppress_banner=True)
    print("nlping df...")  # 20 min ca
    lemmata = []
    pos_tags = []
    for txt in df.text:
        doc = nlp.analyze(text=txt)
        lemmata.append(' '.join(doc.lemmata))
        pos_tags.append(' '.join(doc.pos))
    end_time = t.time()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time} seconds")
    df['lemmata'] = lemmata
    df['pos'] = pos_tags
    # could also get other morphosyntac feat

    return df


def main(folder):
    # iterate function over all files
    df_list = []
    for file in glob.glob(folder + '/*.txt'):
        print(f"creating dataframe from {file}...")
        df = create_dataframe(file)
        df_list.append(df)

    final_df = pd.concat(df_list)
    # add lemmas and pos columns
    final_df = lemmatize(final_df)
    # encode authors as array
    le = LabelEncoder()
    final_df['category_label'] = le.fit_transform(final_df['author'])
    # get what category_label corresponds to author
    for i in final_df['category_label'].unique():
        author_label = le.inverse_transform([i])
        # Print the author label
        print(f"Author label: {i} for ", author_label[0])

    #  save to csv
    final_df.to_csv('labelDataset.csv', index=False)
    print("Finished!")


if __name__ == "__main__":
    main('rawCorpus')
