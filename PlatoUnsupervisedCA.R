# Unsupervised Analysis using STYLO (see References)

### 
# SET UP

# set directory
setwd("~/my_path/PlatoR")
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

## PLAIN
# character 500 3-grams, 4-grams, 5-grams 8-grams
title = "Plain Text overview"
 
PlatoPlain_profileCAwrd2cull75 <- stylo(gui=F, corpus.dir = "profiles",
                      corpus.format = "plain",
                      corpus.lang = "Other",
                      # Features
                      analyzed.features = "c",
                      ngram.size = 3,
                      preserve.case = T, #
                      mfw.min = 1000,
                      mfw.max = 1000,
                      #culling.min = 75,
                      # Analysis
                      analysis.type = "CA",
                      consensus.strength = 0.5,
                      # stable measure, proved by Koppel2014 and Kestemont2016
                      distance.measure = "minmax", 
                      sampling = "no.sampling",
                      # Result
                      display.on.screen = TRUE,
                      #colors.on.graphs = "greyscale",
                      write.png.file = TRUE,
                      plot.custom.height = 7,
                      plot.custom.width = 7,
                      plot.font.size = 12,
                      titles.on.graphs = TRUE,
                      custom.graph.title = title)


#####
## PARSED
# word 2-grams, 3-grams, 4-grams, 5-grams, 6-gram

title = "Plato Parsed-corpus Instance overview"

PlatoParsed_instanceCAwrd6cull0 <- stylo(gui=F, corpus.dir = "",
                                        corpus.format = "plain",
                                        splitting.rule = "\\s",
                                        corpus.lang = "Other",
                                        # Features
                                        analyzed.features = "w",
                                        ngram.size = 6,
                                        preserve.case = T, #
                                        mfw.min = 500,
                                        mfw.max = 500,
                                        culling.min = 0,
                                        # Analysis
                                        analysis.type = "CA",
                                        consensus.strength = 0.5,
                                        # stable measure, proved by Koppel2014 and Kestemont2016
                                        distance.measure = "minmax", 
                                        sampling = "no.sampling",
                                        # Result
                                        display.on.screen = TRUE,
                                        colors.on.graphs = "greyscale",
                                        write.png.file = TRUE,
                                        plot.custom.height = 7,
                                        plot.custom.width = 7,
                                        plot.font.size = 12,
                                        titles.on.graphs = TRUE,
                                        custom.graph.title = title)

