- process XML to clean files (processXML.py)
- change names to letters
- select only 2k+ texts and get stats (makecsv.py)
- make PARSED and PLAIN(ner mased) subdirs (processPOS.py + fun(replace_named_entities()))
- create personalized Platocorpus: Late+Par+2Ph ++ rest of authors (NO REP, Mex or VIII)
- get MEDIAN and MEAN of wrds excluding Laws >> CUT UP LAWS\* into smaller parts >> use them to OBFUSCATE (increase: non-plato)

- configure-Laws >after chunking, manually clip off clipped sentences

- get PSEUDOLAWS in plainlaws dir as PLAIN and PARSED, in their own little dir, so that they can then be OBFUSCATED and mixed with the others as "PSEUDO_LAWS"

- compare features across obfuscated plato + random externals >> strange difference between Law+Al e Law+Ari >>
  > > > new approach: mix laws with 1. other LATE plato 2. EARLY plato (=2Eu) 3. Disputed (+ keep Law+Al)
