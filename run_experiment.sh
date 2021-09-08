#!/bin/bash

gpu=0
TYPE=std # adv, std
model=99       # 1, 2, 3, 99: Conv-lstm
e=10
lda=0.3
lda_inc=0.0
set_=1
python -u main.py -type ${TYPE} \
		-model ${model} \
		-e ${e} \
		-set ${set_} \
		-lda ${lda} \
		-lda_inc ${lda_inc} \
		-type ${TYPE} \
		-gpu $gpu \
		> out_${TYPE}_model_${model}_e${e}_lda${lda}_lstm.txt


