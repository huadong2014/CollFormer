#!/bin/bash

LANG=fr
PROBLEM=translate_en${LANG}_wmt32k

MODEL=transformer
#MODEL=collaboration

#HPARAMS_SET=transformer_base
HPARAMS_SET=transformer_big
#HPARAMS_SET=collaboration_base
#HPARAMS_SET=collaboration_big_enfr

DECODE_FILE=./data/dev/en${LANG}/newstest2014.en
REF_FILE=./data/dev/en${LANG}/newstest2014.${LANG}

POSTFIX=v1
HPARAMS=''
DATA_DIR=./t2t_data/$PROBLEM
TRAIN_DIR=./t2t_train/$PROBLEM/$MODEL-$HPARAMS_SET-$POSTFIX

NUM_CHECKPOINTS=300
LAST_CHECKPOINTS=1070000

BEAM_SIZE=4
ALPHA=0.6
DECODE_OUTPUT=$TRAIN_DIR/decode-translation.txt

CHECKPOINTS_PATH=$TRAIN_DIR/model.ckpt-$LAST_CHECKPOINTS
for((i=$[LAST_CHECKPOINTS-1000]; i>$[LAST_CHECKPOINTS-1000*NUM_CHECKPOINTS]; i-=1000)); do
    CHECKPOINTS_PATH=$TRAIN_DIR/model.ckpt-$i","$CHECKPOINTS_PATH
done

python ./avg_checkpoints.py --checkpoints $CHECKPOINTS_PATH --output_path $TRAIN_DIR/averaged.ckpt

if [ $? -eq 0 ]; then
    t2t-decoder \
        --data_dir=$DATA_DIR \
        --problem=$PROBLEM \
        --model=$MODEL \
        --hparams_set=$HPARAMS_SET \
        --hparams=$HPARAMS \
        --output_dir=$TRAIN_DIR \
        --t2t_usr_dir=$TRAIN_DIR/src \
        --checkpoint_path=$TRAIN_DIR/averaged.ckpt-0 \
        --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
        --decode_from_file=$DECODE_FILE \
        --decode_to_file=$DECODE_OUTPUT
fi

# Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
if [ $? -eq 0 ]; then
    t2t-bleu --translation=$DECODE_OUTPUT --reference=$REF_FILE
    sh ./get_bleu.sh $DECODE_OUTPUT $REF_FILE $LANG 2>/dev/null
fi

