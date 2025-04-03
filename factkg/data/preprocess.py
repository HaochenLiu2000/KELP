import json
import jsonlines
import pickle
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parsing input arguments.")
    parser.add_argument('--factkg_train', type=str, required=True, help='Path for factkg train set.')
    parser.add_argument('--factkg_dev', type=str, required=True, help='Path for factkg dev set.')
    parser.add_argument('--factkg_test', type=str, required=True, help='Path for factkg test set.')
    
    args = parser.parse_args()

    train_set_path = args.factkg_train
    dev_set_path = args.factkg_dev
    test_set_path = args.factkg_test

    with open(train_set_path, 'rb') as f:
        train_set = pickle.load(f)
    claims_train = list(train_set)
    
    with open(dev_set_path, 'rb') as f:
        dev_set = pickle.load(f)
    claims_dev = list(dev_set)
    
    with open(test_set_path, 'rb') as f:
        test_set = pickle.load(f)
    claims_test = list(test_set)

    with jsonlines.open(f'./extracted_train_set.jsonl', mode='w') as w:
        for i, sample in enumerate(claims_train):
            new_sample = {}
            new_sample["question_id"] = i+1
            new_sample["question"] = sample
            new_sample["types"] = test_set[sample]["types"]
            new_sample["entity_set"] = test_set[sample]["Entity_set"]
            new_sample["Label"] = test_set[sample]["Label"]
            w.write(new_sample)
            
    with jsonlines.open(f'./extracted_dev_set.jsonl', mode='w') as w:
        for i, sample in enumerate(claims_dev):
            new_sample = {}
            new_sample["question_id"] = i+1
            new_sample["question"] = sample
            new_sample["types"] = test_set[sample]["types"]
            new_sample["entity_set"] = test_set[sample]["Entity_set"]
            new_sample["Label"] = test_set[sample]["Label"]
            w.write(new_sample)
            
    with jsonlines.open(f'./extracted_test_set.jsonl', mode='w') as w:
        for i, sample in enumerate(claims_test):
            new_sample = {}
            new_sample["question_id"] = i+1
            new_sample["question"] = sample
            new_sample["types"] = test_set[sample]["types"]
            new_sample["entity_set"] = test_set[sample]["Entity_set"]
            new_sample["Label"] = test_set[sample]["Label"]
            w.write(new_sample)