#!/bin/bash
source ~/.bashrc
source activate t2tvenv
mkdir log


MODEL=CollFormer  #CollFormer、CollMHA、RealFormer、 MHADR、ConcreteGate、MixedMA、TalkingHA
#nohup bash_script GPU_id save_path_name parms_setting nGPUs dataset_name max_epochs log_file_name
nohup bash train_tiny_vi.sh 0 envi_tiny_${MODEL}  "max_length=128,num_hidden_layers=6,num_heads=8,modelname=$MODEL,usedecay=-1,reuse_n=1" 1 translate_envi_iwslt32k 50000 > log/envi_tiny_$MODEL.log &
nohup bash train_base_fr.sh 1 enfr_base_${MODEL}  "max_length=128,num_hidden_layers=6,num_heads=8,modelname=$MODEL,usedecay=-1,reuse_n=1" 1 translate_enfr_wmt32k 250000 > log/enfr_base_${MODEL}.log &
nohup bash train_base_de.sh 2 ende_base_${MODEL} "max_length=128,num_hidden_layers=6,num_heads=8,modelname=$MODEL,usedecay=-1,reuse_n=1" 1 translate_ende_wmt32k 250000 > log/ende_base_${MODEL}.log &
nohup bash train_large_fr.sh 2 ende_large_${MODEL} "max_length=128,num_hidden_layers=6,num_heads=8,modelname=$MODEL,usedecay=-1,reuse_n=1" 1 translate_ende_wmt32k 250000 > log/ende_base_${MODEL}.log &


# MODEL=MHADR  #CollMHA、RealFormer、 MHADR、ConcreteGate、MixedMA、TalkingHA
# #nohup bash_script GPU_id save_path_name parms_setting nGPUs dataset_name max_epochs log_file_name
# nohup bash train_tiny_vi.sh 0 envi_tiny_${MODEL} 'max_length=128,num_hidden_layers=6,num_heads=8,modelname=$MODEL' 1 translate_envi_iwslt32k 50000 > log/envi_tiny_$MODEL.log &
# nohup bash train_base_fr.sh 1 enfr_base_${MODEL} 'max_length=128,num_hidden_layers=6,num_heads=8,modelname=$MODEL' 1 translate_enfr_wmt32k 250000 > log/enfr_base_$MODEL.log &
# nohup bash train_base_de.sh 2 ende_base_${MODEL} 'max_length=128,num_hidden_layers=6,num_heads=8,modelname=$MODEL' 1 translate_ende_wmt32k 250000 > log/ende_base_$MODEL.log &


# MODEL=CollFormer
# nohup bash train_tiny.sh 0 envi_tiny_${MODEL}_0  "max_length=128,num_hidden_layers=6,usedecay=0.8,reuse_n=0,modelname=$MODEL" 1 > envi_tiny_${MODEL}_0.log &
# nohup bash train_tiny.sh 1 envi_tiny_${MODEL}_1  "max_length=128,num_hidden_layers=6,usedecay=0.8,reuse_n=1,modelname=$MODEL" 1 > envi_tiny_${MODEL}_1.log &
# nohup bash train_tiny.sh 2 envi_tiny_${MODEL}_2  "max_length=128,num_hidden_layers=6,usedecay=0.8,reuse_n=2,modelname=$MODEL" 1 > envi_tiny_${MODEL}_2.log &
# nohup bash train_tiny.sh 3 envi_tiny_${MODEL}_3  "max_length=128,num_hidden_layers=6,usedecay=0.8,reuse_n=3,modelname=$MODEL" 1 > envi_tiny_${MODEL}_3.log &
# nohup bash train_tiny.sh 5 envi_tiny_${MODEL}_4  "max_length=128,num_hidden_layers=6,usedecay=0.8,reuse_n=4,modelname=$MODEL" 1 > envi_tiny_$${MODEL}_4.log &
# nohup bash train_tiny.sh 6 envi_tiny_${MODEL}_5  "max_length=128,num_hidden_layers=6,usedecay=0.8,reuse_n=5,modelname=$MODEL" 1 > envi_tiny_${MODEL}_5.log &


# ##auto-learn for different collaborative layers usedecay=1.0 in collformer
# nohup bash train_tiny_vi.sh 0 auto_learn_reuse_n_2 'max_length=128,num_hidden_layers=6,usedecay=1.0,reuse_n=2' 1 translate_envi_iwslt32k 50000 > logauto/auto_learn_reuse_n_2.log &
# nohup bash train_tiny_vi.sh 1 auto_learn_reuse_n_3 'max_length=128,num_hidden_layers=6,usedecay=1.0,reuse_n=3' 1 translate_envi_iwslt32k 50000 > logauto/auto_learn_reuse_n_3.log &
# nohup bash train_tiny_vi.sh 2 auto_learn_reuse_n_4 'max_length=128,num_hidden_layers=6,usedecay=1.0,reuse_n=4' 1 translate_envi_iwslt32k 50000 > logauto/auto_learn_reuse_n_4.log &
# nohup bash train_tiny_vi.sh 3 auto_learn_reuse_n_5 'max_length=128,num_hidden_layers=6,usedecay=1.0,reuse_n=5' 1 translate_envi_iwslt32k 50000 > logauto/auto_learn_reuse_n_5.log &
