#! source venv/bin/activate

import glob
import processingScripts.process4R as R
import processingScripts.processXML as process
import processingScripts.NERtokens as ne
import processingScripts.getTXTstatistics as statistics
import parse_dir
import convert2csv
import processingScripts.cleanCSV as clean


def preprocessing(folder):

    # process xml to txt files
    authorList = process.process_files(folder)
    process.processLetters("rawCorpus/tlg0059.tlg036.perseus-grc2.xml")
    # get general overview of data
    statistics.main(folder)
    statistics.combine_files_by_author(folder, 'profiles')
    statistics.main('profiles')

    with open('authorList.txt', 'w') as f:
        for item in authorList:
            f.write("%s\n" % item)

    return


def styloProcess(folder):
    print("___ will take around 1h....")
    # create ref list of NER from plain text
    ne.NER_list(folder)
    # clean text and save in Rcorpus/text
    R.clean_corpus(folder)
    # metadata from fully processed txt files, confront with X,y for models
    statistics.combine_files_by_author('Rcorpus', 'profiles')
    print("\ncontinue processing in R...")
    return


def pythonProcess(folder):

    parse_dir.main(folder)
    # create single CSV with author, title, chunked-text,
    # lemmata, pos, binary and categorical LABELS: convert2csv.main(folder)

    # clean lemmata/plain text before processing: clean.main('labelDataset.csv', 'lemmata')  # or 'text'
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

    folder = "RCorpus"
    pythonProcess(folder)
