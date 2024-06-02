The data processing part of our code (data/preprocess.py) is based on the code of KG-GPT (https://github.com/jiho283/KG-GPT/) and we use the same dataset as KG-GPT. The dataset, requirements, and data preparation follow the setting of KG-GPT.

You can download FactKG from https://github.com/jiho283/FactKG and MetaQA from https://github.com/yuyuz/MetaQA.

The few-shot examples from all datasets are given by KG-GPT.
make_training_set.py is for the building of our training data. You can save the output file output.json in a txt file output.txt.
pretrain_LM_encoder.py performs the optimization of the encoder.
rewrite.py will rewrite output.txt into output.jsonl
test.py/test-1hop.py/test-2hop.py are the inference code on each dataset, respectively.