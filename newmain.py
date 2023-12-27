
import glob
import processingScripts.processPOS as processpos
import processingScripts.processXML as processXML
import processingScripts.NERtokens as processner
import parse_dir
import processingScripts.makecsv as makecsv

folder = 'data/processedXML'
authorList = processXML.process_files("data/rawCorpus", folder)
processXML.processLetters(
    "data/rawCorpus/tlg0059.tlg036.perseus-grc2.xml", folder)
processXML.subset_Dataset(folder)

df = makecsv.create_dataframe(folder)

# os.makedirs("allinstancesPASRED", exist_ok=True)
for file in glob.glob(folder + '/*.txt'):
    content = open(file, 'r').read()
    content = processpos.no_NERaccents(content)
    processpos.annotate_pos(content, file, simple=True)
    print("\n\nDone!")

outpath = 'data/PLAIN'
# apply only replace_named_entities() on a dir
processner.main(folder, outpath)
