#/bin/bash
#source /data/.bashrc
#conda activate py36
#cd /data/exp/attn/mt_final/
export CUDA_VISIBLE_DEVICES=$1 #i

PROBLEM=$5   #translate_envi_iwslt32k  translate_enfr_wmt32k

MODEL=collaboration

HPARAMS_SET=collaboration_tiny  #collaboration_base

POSTFIX=$2 # v6_0.8_5_2
HPARAMS=$3 #'max_length=128,num_hidden_layers=6,usedegray=0.8,reuse_n=5'

GPUs=$4


TRAIN_STEPS=$6

DATA_DIR=./t2t_data/$PROBLEM
TRAIN_DIR=./t2t_train/$PROBLEM/$MODEL-$HPARAMS_SET-$POSTFIX

#rm -rf $TRAIN_DIR
if [ ! -d $TRAIN_DIR ]; then
    mkdir -p $TRAIN_DIR
    cp -r src $0 $TRAIN_DIR/
fi

# Train
time t2t-trainer \
    --data_dir=$DATA_DIR \
    --problem=$PROBLEM \
    --model=$MODEL \
    --hparams_set=$HPARAMS_SET \
    --hparams=$HPARAMS \
    --output_dir=$TRAIN_DIR \
    --t2t_usr_dir=$TRAIN_DIR/src \
    --train_steps=$TRAIN_STEPS \
    --worker_gpu=$GPUs \
    --keep_checkpoint_max=10 \
    --eval_throttle_seconds=10 \
    2>&1 | tee -a $TRAIN_DIR/output.log

# decoding
LANG=vi
DECODE_FILE=./data/dev/en${LANG}/tst2013.en
REF_FILE=./data/dev/en${LANG}/tst2013.${LANG}


# if [$5 == translate_envi_iwslt32k ]; then
#     LANG=vi
#     DECODE_FILE=./data/dev/en${LANG}/tst2013.en
#     REF_FILE=./data/dev/en${LANG}/tst2013.${LANG}
# fi

# if [$5 == translate_ende_wmt32k ]; then
#     LANG=de
#     DECODE_FILE=./data/dev/en${LANG}/newstest2014.en
#     REF_FILE=./data/dev/en${LANG}/newstest2014.${LANG}
# fi

# # if [$5 == translate_enfr_wmt32k ]; then
# LANG=fr
# DECODE_FILE=./data/dev/en${LANG}/newstest2014.en
# REF_FILE=./data/dev/en${LANG}/newstest2014.${LANG}
# # fi


# HPARAMS='num_hidden_layers=6'
BEAM_SIZE=4
ALPHA=0.6

# decode 1
NUM_CHECKPOINTS=10
LAST_CHECKPOINTS=$6
DECODE_OUTPUT=$TRAIN_DIR/decode-translation-${LAST_CHECKPOINTS}.txt

CHECKPOINTS_PATH=$TRAIN_DIR/model.ckpt-$LAST_CHECKPOINTS
for((i=$[LAST_CHECKPOINTS-1000]; i>$[LAST_CHECKPOINTS-1000*NUM_CHECKPOINTS]; i-=1000)); do
    CHECKPOINTS_PATH=$TRAIN_DIR/model.ckpt-$i","$CHECKPOINTS_PATH
done

python ./avg_checkpoints.py --checkpoints $CHECKPOINTS_PATH --output_path $TRAIN_DIR/averaged.ckpt 2>&1 | tee $TRAIN_DIR/decode-log-${LAST_CHECKPOINTS}.log

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
        --decode_to_file=$DECODE_OUTPUT 2>&1 | tee -a $TRAIN_DIR/decode-log-${LAST_CHECKPOINTS}.log
fi

# decode 2
#NUM_CHECKPOINTS=5
#LAST_CHECKPOINTS=50000
#DECODE_OUTPUT=$TRAIN_DIR/decode-translation-${LAST_CHECKPOINTS}.txt
#
#CHECKPOINTS_PATH=$TRAIN_DIR/model.ckpt-$LAST_CHECKPOINTS
#for((i=$[LAST_CHECKPOINTS-1000]; i>$[LAST_CHECKPOINTS-1000*NUM_CHECKPOINTS]; i-=1000)); do
#    CHECKPOINTS_PATH=$TRAIN_DIR/model.ckpt-$i","$CHECKPOINTS_PATH
#done
#
#python ./avg_checkpoints.py --checkpoints $CHECKPOINTS_PATH --output_path $TRAIN_DIR/averaged.ckpt 2>&1 | tee $TRAIN_DIR/decode-log-${LAST_CHECKPOINTS}.log
#
#if [ $? -eq 0 ]; then
#    t2t-decoder \
#        --data_dir=$DATA_DIR \
#        --problem=$PROBLEM \
#        --model=$MODEL \
#        --hparams_set=$HPARAMS_SET \
#        --hparams=$HPARAMS \
#        --output_dir=$TRAIN_DIR \
#        --t2t_usr_dir=$TRAIN_DIR/src \
#        --checkpoint_path=$TRAIN_DIR/averaged.ckpt-0 \
#        --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
#        --decode_from_file=$DECODE_FILE \
#        --decode_to_file=$DECODE_OUTPUT 2>&1 | tee -a $TRAIN_DIR/decode-log-${LAST_CHECKPOINTS}.log
#fi

# Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
if [ $? -eq 0 ]; then
    LAST_CHECKPOINTS=$LAST_CHECKPOINTS
    DECODE_OUTPUT=$TRAIN_DIR/decode-translation-${LAST_CHECKPOINTS}.txt
    t2t-bleu --translation=$DECODE_OUTPUT --reference=$REF_FILE 2>&1 | tee -a $TRAIN_DIR/decode-log-${LAST_CHECKPOINTS}.log
    sh ./get_bleu.sh $DECODE_OUTPUT $REF_FILE $LANG 2>/dev/null | tee -a $TRAIN_DIR/decode-log-${LAST_CHECKPOINTS}.log
	sacrebleu $REF_FILE -i $DECODE_OUTPUT -m bleu -b -w 4 2>&1 | tee -a $TRAIN_DIR/decode-log-${LAST_CHECKPOINTS}.log
fi
#
#if [ $? -eq 0 ]; then
#    LAST_CHECKPOINTS=50000
#    DECODE_OUTPUT=$TRAIN_DIR/decode-translation-${LAST_CHECKPOINTS}.txt
#    t2t-bleu --translation=$DECODE_OUTPUT --reference=$REF_FILE 2>&1 | tee -a $TRAIN_DIR/decode-log-${LAST_CHECKPOINTS}.log
#    sh ./get_bleu.sh $DECODE_OUTPUT $REF_FILE $LANG 2>/dev/null | tee -a $TRAIN_DIR/decode-log-${LAST_CHECKPOINTS}.log
#fi

