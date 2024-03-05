# Unsupervised Analysis

##### PLATO: 
# SET UP

# set directory
setwd("~/Desktop/UNI_PV/TESI_PV/PlatoCode/PlatoR")
# load libraries
library(stylo)
library(tidyverse)
library(stringi)
library(dplyr)
library(ggplot2)
library(data.table)
library(wordcloud)
library(arules)
library(gplots)


#####
# --> INSTANCE: Plato+Pseudo
## PLAIN
# word 100 2-grams and 3-grams
# character 500 3-grams, 4-grams 5-grams

title = "Plato Instance overview"
# word 3-gram not enough for culling
PlatoPlain_profileCAchr5cull50 <- stylo(gui=F, corpus.dir = "platoinstances",
                                        corpus.format = "plain",
                                        corpus.lang = "Other",
                                        # Features
                                        analyzed.features = "c",
                                        ngram.size = 5,
                                        preserve.case = T, #
                                        mfw.min = 500,
                                        mfw.max = 500,
                                        culling.min = 50,
                                        # Analysis
                                        analysis.type = "CA",
                                        consensus.strength = 0.5,
                                        distance.measure = "minmax", # to have a stable measure, proved by Koppel2014 and Kestemont2016
                                        sampling = "no.sampling",
                                        # Result
                                        display.on.screen = TRUE,
                                        colors.on.graphs = "greyscale",
                                        #write.png.file = TRUE,
                                        plot.custom.height = 7,
                                        plot.custom.width = 7,
                                        plot.font.size = 11,
                                        titles.on.graphs = TRUE,
                                        custom.graph.title = title)



#####
## PARSED
# word 2-grams, 3-grams, 4-grams, 5-grams 6-grams
