#!/bin/bash

mosesdecoder=./data/mosesdecoder

decodes_file=$1
gold_targets=$2
lang=$3

# Replace unicode.
perl $mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl -l $3  < $decodes_file > $decodes_file.n
perl $mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl -l $3  < $gold_targets > $gold_targets.n

# Tokenize.
perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l $3 < $decodes_file.n > $decodes_file.tok
perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l $3 < $gold_targets.n > $gold_targets.tok

# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decodes_file.tok > $decodes_file.tok.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $gold_targets.tok > $gold_targets.atat

# Get BLEU.
perl $mosesdecoder/scripts/generic/multi-bleu.perl $gold_targets.atat < $decodes_file.tok.atat

