#/bin/bash

PROBLEM=translate_enfr_wmt32k

TMP_DIR=./tmp/t2t_datagen
DATA_DIR=./t2t_data/$PROBLEM

mkdir -p $DATA_DIR $TMP_DIR
cp data/*.tgz data/*.tar $TMP_DIR
cp -r src $0 $DATA_DIR

# Generate data
time t2t-datagen \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM \
    --t2t_usr_dir=`pwd`/src \
    2>&1 | tee $DATA_DIR/output.log

