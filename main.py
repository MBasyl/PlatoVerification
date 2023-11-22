#! source venv/bin/activate

import process4R as R
import processXML as process
# import named_entitiesLEMMATA as ne
import NERtokens as ne
import oldNER4lemma as nl
import getTXTstatistics
import convert2csv
import cleanCSV as clean
import glob


def preprocessing(folder):

    # process xml to txt files
    authorList = process.process_files(folder)
    process.processLetters("rawCorpus/tlg0059.tlg036.perseus-grc2.xml")
    # get general overview of data
    getTXTstatistics.main(folder)
    getTXTstatistics.combine_files_by_author(folder, 'profiles')
    getTXTstatistics.main('profiles')

    with open('authorList.txt', 'w') as f:
        for item in authorList:
            f.write("%s\n" % item)

    return


def styloProcess(folder):
    print("___ will take around 1h....")
    # create ref list of NER from plain text
    # ne.NER_list(folder)
    # clean text and save in Rcorpus/text
    R.clean_corpus(folder)
    # metadata from fully processed txt files, confront with X,y for models
    # getTXTstatistics.combine_files_by_author('Rcorpus', 'profiles')
    print("\ncontinue processing in R...")
    return


def pythonProcess(folder):
    # create single CSV with author, title, chunked-text,
    # lemmata, pos, binary and categorical LABELS
    convert2csv.main(folder)
    print("Output is complete_dataset.csv")
    # create ref list of NER from lemma text
    nl.NER_list(folder)
    # clean lemmata/plain text before processing
    clean.main('labelDataset.csv', 'lemmata')  # or 'text'
    print("Output is labelDataset_cleaned.csv")

    return


if __name__ == '__main__':

    folder = "rawCorpus"
    # preprocessing(folder)
    # count number of txt files in folder rawCorpus folder
    # fileCounter = len(glob.glob1("rawCorpus/", "*.xml"))
    # print("Number of xml files in folder rawCorpus: ", fileCounter)
    # fileCounter = len(glob.glob1("rawCorpus/", "*.txt"))
    # print("Number of txt files in folder rawCorpus: ", fileCounter)

    styloProcess(folder)

    fileCounter = len(glob.glob1("RCorpus/", "*.txt"))
    print("Number of txt files in folder RCorpus: ", fileCounter)

    # pythonProcess(folder)
