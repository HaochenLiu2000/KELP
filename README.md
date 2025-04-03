# Knowledge Graph-Enhanced Large Language Models via Path Selection

The codes are associated with the following paper:

>**Knowledge Graph-Enhanced Large Language Models via Path Selection,**[PDF](https://arxiv.org/pdf/2406.13862)\\
>Haochen Liu, Song Wang, Yaochen Zhu, Yushun Dong, Jundong Li,     
>Annual Meeting of the Association for Computational Linguistics (ACL), 2024.

<p align="center">
<img src="KELP.png" alt="Overview of KELP." width="100%" />
</p>

## 1. Datasets

The dataset, requirements, and data preparation follow the setting of [KG-GPT](https://github.com/jiho283/KG-GPT/). 

Download [FactKG](https://github.com/jiho283/FactKG) and [MetaQA](https://github.com/yuyuz/MetaQA) here.

Place the files `dbpedia_2015_undirected_light.pickle`, `factkg_test.pickle`, `factkg_test.pickle`, `factkg_test.pickle` under `./factkg`.

Place the files or folders `kb.txt`, `1-hop/vanilla`, `2-hop/vanilla` under `./metaQA`.

For data preprocessing, run:

    cd factkg
    python data/preprocess.py --factkg_train factkg_train.pickle --factkg_dev factkg_dev.pickle --factkg_test factkg_test.pickle
    cd ..
    
    cd metaQA
    python data/preprocess.py --setting train --kb kb.txt
    python data/preprocess.py --setting dev --kb kb.txt
    python data/preprocess.py --setting test --kb kb.txt
    cd ..

## 2. Openai Key

Write your own OpenAI API key in factkg/openai_api_key.txt and metaqa/openai_api_key.txt and save them.

## 3. Building of the Training Data

To build the specific training data from the original datasets:

Run

    cd factkg
    python make_training_set.py --setting train
    python make_training_set.py --setting dev
    cd ..

    cd metaQA
    python make_training_set.py --setting train --hop 1
    python make_training_set.py --setting dev --hop 1
    python make_training_set.py --setting train --hop 2
    python make_training_set.py --setting dev --hop 2
    cd ..


## 4. Training

To train our model on dataset:

Run
    cd factkg
    python pretrain_LM_encoder.py
    cd ..

    cd metaQA
    python pretrain_LM_encoder.py --hop 1
    python pretrain_LM_encoder.py --hop 2
    cd ..

## 4. Evaluation

To test the trained model:

Run

    cd factkg
    python test.py --question_model <question_model_path> --question_model_relation_only <question_model_relation_only_path>
    cd ..
    
    cd metaQA
    python test-1hop.py --question_model <question_model_path>
    python test-2hop.py --question_model <question_model_path>
    cd ..

## 5. Acknowledgment

The dataset, requirements, and data preparation follow the setting of [KG-GPT](https://github.com/jiho283/KG-GPT/). 

Thanks to the authors and developers!

## 6. Citation
If you find this work is helpful to your research, please consider citing our paper:
```
@inproceedings{liu-etal-2024-knowledge-graph,
    title = "Knowledge Graph-Enhanced Large Language Models via Path Selection",
    author = "Liu, Haochen and Wang, Song and Zhu, Yaochen and Dong, Yushun and Li, Jundong",
    editor = "Ku, Lun-Wei and Martins, Andre and Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    pages = "6311--6321",
}
```
**Thanks for your interest in our work!**