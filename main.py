#! venv/bin/activate
import glob
import os
import preprocessing.processPOS as processpos
import preprocessing.processXML as processXML
import preprocessing.NERtokens as processner
import parse_dir

###########

folder = 'data/processedXML'
# process XML to clean files
# authorList = processXML.process_files("data/rawCorpus", folder)
# processXML.processLetters(
#    "rawCorpus/tlg0059.tlg036.perseus-grc2.xml", folder)
# remember to change FILE NAMES for Disputed (and cardinals for Letters)

# select only 2k+ texts and get stats (makecsv.py)


# To get STATISTICS:
# processXML.subset_Dataset(folder)
# df = create_dataframe(folder) <- function from formatJSON-CSV


# make PARSED and PLAIN subdirs

os.makedirs("PARSED", exist_ok=True)
for file in glob.glob(folder + '/*.txt'):
    content = open(file, 'r').read()
    processpos.annotate_pos(content, file, simple=True)
    print("\n\nDone!")

os.makedirs("PLAIN", exist_ok=True)
outpath = 'data/PLAIN'
# apply only replace_named_entities() on a dir
processner.main(folder, outpath)

# create ML dir with Late_Plato profile + rest of dataset exluding PsPla_Mex and PsPla_VIII

# Chunk Laws to treat as test-documents:
#
# 1. configure-Laws.py > after chunking, manually clip off clipped sentences
# 2. OBFUSCATE part (increase: non-plato) --> pseudo_laws
# 3. process PSEUDOLAWS as PLAIN and PARSED
