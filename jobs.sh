#!/bin/bash
source envn/bin/activate
# python3 -m pip install keras-nlp
# pip install --upgrade tensorflow
python3 process.py cs_pdt UD_Czech-PDT/cs_pdt-ud-train.conllu UD_Czech-PDT/cs_pdt-ud-test.conllu


