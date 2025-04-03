import pickle
import json
import random
import openai
import os
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer, losses
from transformers import DistilBertModel, DistilBertTokenizer
from tqdm import tqdm
import json
import jsonlines
import torch
import argparse
parser = argparse.ArgumentParser(description="Parsing input arguments.")
parser.add_argument('--question_model', type=str, required=True)
parser.add_argument('--question_model_relation_only', type=str, required=True)
args = parser.parse_args()
question_model_path = args.question_model
question_model_path_relation_only = args.question_model_relation_only

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
question_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
question_model.load_state_dict(torch.load(question_model_path))
question_model.to(device)
question_model2 = DistilBertModel.from_pretrained('distilbert-base-uncased')
question_model2.load_state_dict(torch.load(question_model_path_relation_only))
question_model2.to(device)
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

openai.api_key = open_file('./openai_api_key.txt')

with open('dbpedia_2015_undirected_light.pickle', 'rb') as f:
    kg = pickle.load(f)

                                        
questions_dict = {}
entity_set_dict = {}
label_set_dict = {}

question_data='extracted_test_set.jsonl'
note='note.txt'
error_file='error.json'
with open(question_data, 'r') as f:
    for line in f:
        if not line:
            continue
        dataset = json.loads(line)
        questions_dict[dataset["question_id"]] = dataset["question"]
        entity_set_dict[dataset["question_id"]] = dataset["entity_set"]
        label_set_dict[dataset["question_id"]] = dataset["Label"]



model_name = 'gpt-3.5-turbo-0613'
max_tokens = 400
temperature = 0.2
top_p = 0.1


def llm(prompt,max_tokens=max_tokens):
    for _ in range(3):
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=max_tokens,
                temperature=0.2,
                top_p = 0.1,
                timeout=30
            )
            generated_text = response["choices"][0]["message"]["content"]
            return generated_text
        except Exception as e:
            if _==2:
                print("[ERROR]", e)
            time.sleep(5)
            


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def context_query(question_list,ground_truth_list,context_texts):
    #print(question_list)
    #print(ground_truth_list)
    #print(context_texts)
    
    
    
    prompt="""Verify the  following claims.\n The context is the evidence of triplets may help your verifying.\n
            Each context contains triplets in the form of [head, relation, tail] and it means "head's relation is tail.".
            Choose one of {True, False}, and give me the one-sentence evidence. Examples:\n
            """
    
    prompt+="""
            Context 1: [['Ahamad_Kadhim', 'clubs', "Al-Zawra'a SC"], ] Claim 1: Ahmad Kadhim Assad's club is Al-Zawra'a SC.
            Answer 1: True, based on the evidence set, Ahmad Kadhim Assad's club is Al-Zawra'a SC.
            Context 2: [['Bananaman', 'firstAired', '"1983-10-03"'], ['Bananaman', 'starring', 'Tim_Brooke-Taylor'], ] Claim 2: Yeah! I know that a TV show, which starred Tim Brooke-Taylor, first aired on 3rd October 1983!
            Answer 2: True, the claim is supported by the evidence since Bananaman refers to the TV show.
            Context 3: [['Jamie_Lawrence', 'composer', 'Death_on_a_Factory_Farm'], ['Death_on_a_Factory_Farm', 'director', 'Sarah_Teale'], ] Claim 3: Really? Jamie Lawrence is the music composer of the 83 minute 'Death on a Factory Farm' film, directed by Sarah Teale!
            Answer 3: False, there is no evidence for the 83 minute length.
            Context 4: [[], ] Claim 4: Do you know Milan Hodža? he had a religion.
            Answer 4: False, there is no evidence that Milan had a religion.
            Context 5: [[], ] Claim 5: No, but the leader of the United States is not Olena Serdiuk.
            Answer 5: True, based on the evidence set, there is no information that the leader of the United States is Olena Serdiuk.
            Context 6: [['Brandon_Carter', 'almaMater', 'University_of_Cambridge'], ['Brandon_Carter', 'birthPlace', 'England'], ['University_of_Cambridge', 'viceChancellor', 'Leszek_Borysiewicz'], ] Claim 6: Brandon Carter was born in England and graduated from the University of Cambridge where the current Chancellor is Leszek Borysiewicz.
            Answer 6: True, everything of the claim is supported by the evidence set.
            Context 7: [['Unpublished_Story', 'director', 'Harold_French'], ['Unpublished_Story', 'cinematography', 'Bernard_Knowles'], ] Claim 7: 'A film' was produced by Anatole de Grunwald, directed by Harold French, with cinematography done by Bernard Knowles.
            Answer 7: False, there is no information about the producer of 'Unpublished_Story'.
            Context 8: [['200_Public_Square', 'location', 'Cleveland'], ['200_Public_Square', 'floorCount', '"45"'], ['Cleveland', 'country', 'United_States'], ] Claim 8: Yes, with a floor count of 45, 200 Public Square is located in Cleveland in the United States.
            Answer 8: True, everything of the claim is supported by the evidence set.\n"""
            #Context 9: [['Bananaman', 'starring', 'Bill_Oddie'], ['Bananaman', 'network', 'Broadcasting_House'], ['Bananaman', 'locationCity', 'Broadcasting_House'], ] Claim 9: Bananaman the TV series starred by a person was shown on the company and the company headquarters is called Broadcasting House.
            #Answer 9: True, everything of the claim is supported by the evidence set.
            #Context 10: [['Azerbaijan', 'leaderName', 'Artur_Rasizade'], ["Baku_Turkish_Martyrs'_Memorial", 'designer', '"Hüseyin Bütüner and Hilmi Güner"'], ["Baku_Turkish_Martyrs'_Memorial", 'location', 'Azerbaijan'], ] Claim 10: The place, designed by Huseyin Butuner and Hilmi Guner, is located in a country, where the leader is Artur Rasizade.
            #Answer 10: True, everything of the claim is supported by the evidence set.
            #Context 11: [['AIDAstella', 'shipBuilder', 'Meyer_Werft'], ['AIDAstella', 'shipOperator', 'AIDA_Cruises'], ] Claim 11: AIDA Cruise line operated the ship which was built by Meyer Werft in Townsend, Poulshot, Wiltshire.
            #Answer 11: False, there is no evidence for Townsend, Poulshot, Wiltshire.
            #Context 12: [[], ] Claim 12: An academic journal with code IJPHDE is also Acta Math. Hungar.
            #Answer 12: False, there is no evidence that the academic journal is also Acta Math. Hungar.
    
          
    prompt+='Now verify the following '+str(len(question_list))+' claims in the same way of these examples.\n'
    j=0
    for question in question_list:
        j+=1
        prompt+='Context '+str(j)+f': {context_texts[0]}'+f' Claim '+str(j)+f': {question}'+'\n'
    prompt+='Answer '+str(j)+': '     
    result = llm(prompt)
    context_correct_list=len(question_list)*[False]
    
    if 'false' in result.lower():
        prompt="""Verify the claim. Is this claim True? or False?
                Choose one of {True, False}. If you are unsure, please choose the option you think is most likely."""
        
        prompt+="""
            Claim: Ahmad Kadhim Assad's club is Al-Zawra'a SC.
            Answer: True.
            Claim: Yeah! I know that a TV show, which starred Tim Brooke-Taylor, first aired on 3rd October 1983!
            Answer: True.
            Claim: Really? Jamie Lawrence is the music composer of the 83 minute 'Death on a Factory Farm' film, directed by Sarah Teale!
            Answer: False.
            Claim: Do you know Milan Hodža? he had a religion.
            Answer: False.
            Claim: No, but the leader of the United States is not Olena Serdiuk.
            Answer: True.
            Claim: Brandon Carter was born in England and graduated from the University of Cambridge where the current Chancellor is Leszek Borysiewicz.
            Answer: True.
            Claim: 'A film' was produced by Anatole de Grunwald, directed by Harold French, with cinematography done by Bernard Knowles.
            Answer: False.
            Claim: Yes, with a floor count of 45, 200 Public Square is located in Cleveland in the United States.
            Answer: True.
            Claim: Bananaman the TV series starred by a person was shown on the company and the company headquarters is called Broadcasting House.
            Answer: True.
            Claim: The place, designed by Huseyin Butuner and Hilmi Guner, is located in a country, where the leader is Artur Rasizade.
            Answer: True.
            Claim: AIDA Cruise line operated the ship which was built by Meyer Werft in Townsend, Poulshot, Wiltshire.
            Answer: False.
            Claim: An academic journal with code IJPHDE is also Acta Math. Hungar.
            Answer: False.
            Claim: """
        prompt+=question_list[0]+'\n'+'Answer: '
        result = llm(prompt)
        
    for j in range(len(question_list)):
        for lab in ground_truth_list[j]:
            if (lab and ('true' in result.lower())) or ((not lab) and ('false' in result.lower())):
                context_correct_list[j]=True
                break
    return context_correct_list


def find_top_k_elements(lst, k):
    indexed_lst = list(enumerate(lst))
    sorted_lst = sorted(indexed_lst, key=lambda x: x[1], reverse=True)
    top_k_elements = sorted_lst[:k]
    top_k_values = [value for index, value in top_k_elements]
    top_k_indices = [index for index, value in top_k_elements]
    return top_k_values, top_k_indices






criterion=nn.CosineSimilarity()


dataset_len=len(questions_dict)
a=1
k1=4
k2=4
data_num=range(a,dataset_len+1,1)
total_correct=0

question_id_list=[]
question_list=[]
entity_set_list=[]
ground_truth_list=[]
contexts_list=[]

for ii in tqdm(data_num):
    question = questions_dict[ii]
    entity_set = entity_set_dict[ii]
    ground_truth =label_set_dict[ii]
    
    triplets=[]
    cossim=[]
    i=0
    pos_list=[]
    question_input = tokenizer([question], return_tensors="pt", padding=True, truncation=True).to(device)
    question_embedding = question_model(**question_input).last_hidden_state.mean(dim=1)
    question_embedding2 = question_model2(**question_input).last_hidden_state.mean(dim=1)
    
    subgraph = []
    for entity in entity_set:
        if entity in kg:
            for relation, object in kg[entity].items():
                if [str(relation)] not in subgraph:
                    subgraph.append([str(relation)])
                m=0
                for obj in kg[entity][relation]:
                    for relation2, object2 in kg[obj].items():
                        if [str(relation),str(relation2)] not in subgraph:
                            subgraph.append([str(relation),str(relation2)])
                            m+=1
    for triplet in subgraph:
        if len(triplet)==1:
            pos=triplet[0]+'.'
        elif len(triplet)==2:
            pos=triplet[0]+', '+triplet[1]+'.'
        pos_list.append(pos)
        
        i+=1
        if i>=300:
            
            positive_input = tokenizer(pos_list, return_tensors="pt", padding=True, truncation=True).to(device)
            positive_embedding = question_model(**positive_input).last_hidden_state.mean(dim=1)
            similarity_scores_pos=criterion(question_embedding, positive_embedding).tolist()
            
            cossim+=similarity_scores_pos
            
            pos_list=[]
            i=0
    if len(pos_list)>0:
        positive_input = tokenizer(pos_list, return_tensors="pt", padding=True, truncation=True).to(device)
        positive_embedding = question_model(**positive_input).last_hidden_state.mean(dim=1)
        similarity_scores_pos=criterion(question_embedding, positive_embedding).tolist()
        
        cossim+=similarity_scores_pos
        pos_list=[]
    indexed_lst = list(enumerate(cossim))
    context_texts="["
    top_k_values, top_k_indices=find_top_k_elements(indexed_lst,k1)
    for i in top_k_indices:
        if len(subgraph[i])==1:
            for entity in entity_set:
                if entity in kg:
                    if subgraph[i][0] in kg[entity]:
                        relation=subgraph[i][0]
                        cossim=[]
                        objlist=list(kg[entity][relation])
                        for obj in objlist:
                            pos=str(entity)+' '+str(relation)+' '+str(obj)+'.'
                            pos_list.append(pos)
                            if len(pos_list)>=300:         
                                positive_input = tokenizer(pos_list, return_tensors="pt", padding=True, truncation=True).to(device)
                                positive_embedding = question_model2(**positive_input).last_hidden_state.mean(dim=1)
                                similarity_scores_pos=criterion(question_embedding2, positive_embedding).tolist()
                                cossim+=similarity_scores_pos
                                pos_list=[]
                        if len(pos_list)>0:
                            positive_input = tokenizer(pos_list, return_tensors="pt", padding=True, truncation=True).to(device)
                            positive_embedding = question_model2(**positive_input).last_hidden_state.mean(dim=1)
                            similarity_scores_pos=criterion(question_embedding2, positive_embedding).tolist()
                            cossim+=similarity_scores_pos
                            pos_list=[]
                        indexed_lst2 = list(enumerate(cossim))
                        if len(indexed_lst2)<=k2:
                            top_k_indices2=list(range(len(indexed_lst2)))     
                        else:
                            top_k_values2, top_k_indices2=find_top_k_elements(indexed_lst2, k2)
                        for j in top_k_indices2:
                            obj=objlist[j]
                            if str(relation)[0]!='~':
                                context_texts+='['+str(entity)+', '+str(relation)+', '+str(obj)+'], '
                            else:
                                context_texts+='['+str(obj)+', '+str(relation)[1:]+', '+str(entity)+'], '
        elif len(subgraph[i])==2:
            save_triplets=[]
            for entity in entity_set:
                if entity in kg:
                    if (subgraph[i][0] in kg[entity]):
                        relation=subgraph[i][0]
                        relation2=subgraph[i][1]
                        cossim=[]
                        objlist1=list(kg[entity][relation])
                        for obj in objlist1:
                            if relation2 in kg[obj]:
                                for obj2 in kg[obj][relation2]:
                                    pos=str(entity)+' '+str(relation)+' '+str(obj)+', '+str(obj)+' '+str(relation2)+' '+str(obj2)+'.'
                                    save_triplets+=[(str(entity),str(relation),str(obj),str(relation2),str(obj2))]
                                    pos_list.append(pos)
                                    if len(pos_list)>=300:         
                                        positive_input = tokenizer(pos_list, return_tensors="pt", padding=True, truncation=True).to(device)
                                        positive_embedding = question_model2(**positive_input).last_hidden_state.mean(dim=1)
                                        similarity_scores_pos=criterion(question_embedding2, positive_embedding).tolist()
                                        cossim+=similarity_scores_pos
                                        pos_list=[]
                                if len(pos_list)>0:
                                    positive_input = tokenizer(pos_list, return_tensors="pt", padding=True, truncation=True).to(device)
                                    positive_embedding = question_model2(**positive_input).last_hidden_state.mean(dim=1)
                                    similarity_scores_pos=criterion(question_embedding2, positive_embedding).tolist()
                                    cossim+=similarity_scores_pos
                                    pos_list=[]
                        indexed_lst2 = list(enumerate(cossim))
                        if len(indexed_lst2)<=k2:
                            top_k_indices2=list(range(len(indexed_lst2)))     
                        else:
                            top_k_values2, top_k_indices2=find_top_k_elements(indexed_lst2, k2)
                        for j in top_k_indices2:
                            triplet=save_triplets[j]
                            if str(triplet[1])[0]!='~':
                                context_texts+='['+str(triplet[0])+', '+str(triplet[1])+', '+str(triplet[2])+'], '
                            else:
                                context_texts+='['+str(triplet[2])+', '+str(triplet[1])[1:]+', '+str(triplet[0])+'], '
                            if str(triplet[3])[0]!='~':
                                context_texts+='['+str(triplet[2])+', '+str(triplet[3])+', '+str(triplet[4])+'], '
                            else:
                                context_texts+='['+str(triplet[4])+', '+str(triplet[3])[1:]+', '+str(triplet[2])+'], '
                        
                        
                
    context_texts+="]"
    question_id_list.append(ii)
    question_list.append(question)
    entity_set_list.append(entity_set)
    ground_truth_list.append(ground_truth)
    contexts_list.append(context_texts)  
    if len(question_id_list)>=1:   
        cor_list=context_query(question_list,ground_truth_list,contexts_list)
        if cor_list=="answer length error":
            for i in range(len(question_list)):
                result_dict={}
                result_dict['question_id']=question_id_list[i]
                result_dict['question']=question_list[i]
                result_dict['entity_set']=entity_set_list[i]
                result_dict['ground_truth']=ground_truth_list[i]
                with open(error_file, 'a') as f:
                    json.dump(result_dict, f, indent=3)
                    f.write('\n')
            question_id_list=[]
            question_list=[]
            entity_set_list=[]
            ground_truth_list=[]
            contexts_list=[]
        else:     
            count_true = cor_list.count(True)
            total_correct+=count_true
            with open(note, 'w') as file:
                file.write(str(total_correct)+'/'+str(ii)+'    ')
                file.write(str(total_correct/ii))
                file.write('\n')
            question_id_list=[]
            question_list=[]
            entity_set_list=[]
            ground_truth_list=[]
            contexts_list=[]
            
if len(question_id_list)>=1:          
    cor_list=context_query(question_list,ground_truth_list,contexts_list)
    if cor_list=="answer length error":
        for i in range(len(question_list)):
            result_dict={}
            result_dict['question_id']=question_id_list[i]
            result_dict['question']=question_list[i]
            result_dict['entity_set']=entity_set_list[i]
            result_dict['ground_truth']=ground_truth_list[i]
            with open(error_file, 'a') as f:
                json.dump(result_dict, f, indent=3)
                f.write('\n')    
        question_id_list=[]
        question_list=[]
        entity_set_list=[]
        ground_truth_list=[]
        contexts_list=[] 
    else:   
        count_true = cor_list.count(True)
        total_correct+=count_true
        with open(note, 'w') as file:
            file.write(str(total_correct)+'/'+str(ii)+'    ')
            file.write(str(total_correct/ii))
            file.write('\n')
        question_id_list=[]
        question_list=[]
        entity_set_list=[]
        ground_truth_list=[]
        contexts_list=[]

print('Acc: ',total_correct,'/',ii,'=',total_correct/ii)