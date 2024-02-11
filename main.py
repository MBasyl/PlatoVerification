from preprocessor import DataPreprocessor as dp
import glob
import re


def preprocess_pipeline(corpusfolder='processedXML'):
    # process XML files
    dp.xml_files(folder='rawCorpus')
    # remove works with less than 2k words from processed files
    dp.subset_dataset(threshold=2000, directory_path=corpusfolder)

    # Execute NER to manually inspect (takes ca 30 mins)
    # dp.NER_list(corpusfolder)

    # Create two datasets directories (use for Cluster Analysis)
    dp.make_parsed(corpusfolder, morpho=True)  # time: 20 min
    dp.make_plain(corpusfolder)
    # OPTIONAL: dp.make_parsed(corpusfolder, morpho=False, outpath='data/alternativeParsed')

    # for f in glob.glob("PARSED/*.txt"):
    #   content = file.read()
    #   modified_content = re.sub("\.", "", content)
    #   modified_content = re.sub(";", "_ ", modified_content)
    #   file.write(modified_content)
    return


# STEP 1 : PREPROCESSING
preprocess_pipeline()

# STEP 2 :
# Create PROFILE CONFIGURATION (makedfprofiles.py)

# STEP 3 : Perform SVDD
# (performSVDD.py)

# STEP 4: TO DO
# (obfuscate.py)

# STEP 5: TO DO
# on PLAIN: make . and ; as separate characters
# split train/test/val!!
# Perform SVM/RLP
# Inspect features. i.e. start of sentences influent?
