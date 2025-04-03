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
args = parser.parse_args()
question_model_path = args.question_model

question_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
question_model.load_state_dict(torch.load(question_model_path))
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
question_model.to(device)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

openai.api_key = open_file('./openai_api_key.txt')

with open('data/metaqa_kg.pickle', 'rb') as f:
    kg = pickle.load(f)
hop=1
                                 
questions_dict = {}
entity_set_dict = {}
label_set_dict = {}
if hop==1:
    question_data='data/onehop_test_set.jsonl'
    note='note.txt'
    error_file='one-hop-errors.json'
if hop==2:
    question_data='data/twohop_test_set.jsonl'
    note='note.txt'
    error_file='two-hop-errors.json'
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
            

def build_subgraph(entity_set, knowledge_graph):
    subgraph = set()
    for entity in entity_set:
        if entity in knowledge_graph:
            for relation, object in knowledge_graph[entity].items():
                for obj in knowledge_graph[entity][relation]:
                    subgraph.add((str(entity), str(relation), str(obj)))
                    #for relation2, object2 in knowledge_graph[obj].items():
                    #    for obj2 in knowledge_graph[obj][relation2]:
                    #        subgraph.add((str(entity), str(relation), str(obj), str(relation2), str(obj2)))
        
    return subgraph

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def context_query(question_list,ground_truth_list,context_texts):
    
    for _ in range(3):
        prompt="""Answer the following questions.\n The context is the evidence of triplets may help your verifying.\n
                Each context contains triplets in the form of [head, relation, tail] and it means "head's relation is tail.".
                If you think a question can have multiple answers, you must choose one and answer it. Enter when you start answering the next question. Examples:\n
                """
        prompt+="""
        Context 1: [['Coming Home', 'has_genre', 'Drama'], ['Coming Home', 'has_genre', 'War'], ['Coming Home', 'has_tags', 'vietnam'], ['Coming Home', 'has_tags', 'hal ashby'], ] Question 1: what words describe [Coming Home]?
        Answer 1: 'hal ashby'
        Context 2: [['Chungking Express', 'starred_actors', 'Faye Wong'], ['Chinese Odyssey 2002', 'starred_actors', 'Faye Wong'], ] Question 2: what films does [Faye Wong] appear in?
        Answer 2: 'Chungking Express'
        Context 3: [['Code Unknown', 'has_tag', 'haneke'], ['The Piano Teacher', 'has_tag', 'haneke'], ['Funny Games', 'has_tag', 'haneke'], ['Time of the Wolf', 'has_tag', 'haneke'], ] Question 3: what films are about [haneke]?
        Answer 3: 'The Piano Teacher'
        Context 4: [['Inescapable', 'directed_by', 'Ruba Nadda'], ['Inescapable', 'written_by', 'Ruba Nadda'], ['Inescapable', 'starred_actors', 'Marisa Tomei'], ['Inescapable', 'starred_actors', 'Joshua Jackson'], ['Inescapable', 'starred_actors', 'Alexander Siddig'], ] Question 4: who acted in the movie [Inescapable]?
        Answer 4: 'Marisa Tomei'
        Context 5: [['Things to Come', 'directed_by', 'William Cameron Menzies'], ] Question 5: can you name a film directed by [William Cameron Menzies]?
        Answer 5: 'Things to Come'
        Context 6: [['Witness for the Prosecution', 'has_tags', 'bd-r'], ['Witness for the Prosecution', 'has_genre', 'Drama'], ['Witness for the Prosecution', 'has_tags', 'courtroom'], ] Question 6: what sort of movie is [Witness for the Prosecution]?
        Answer 6: 'Drama'
        Context 7: [['The Mouse That Roared', 'has_genre', 'Comedy'], ['The Mouse That Roared', 'has_tags', 'satirical'], ['The Mouse That Roared', 'has_tags', 'peter sellers'], ] Question 7: what type of film is [The Mouse That Roared]?
        Answer 7: 'Comedy'
        Context 8: [['Blackboards', 'in_language', 'Kurdish'], ['Blackboards', 'has_genre', 'War'], ['Blackboards', 'has_tags', 'samira makhmalbaf'], ] Question 8: what is the primary language in the film [Blackboards]?
        Answer 8: 'Kurdish'
        Context 9: [['The Truth of Lie', 'written_by', 'Roland Reber'], ['The Truth of Lie', 'directed_by', 'Roland Reber'], ['The Truth of Lie, 'has_genre', 'Thriller'], ] Question 9: who is the creator of the film script for [The Truth of Lie]?
        Answer 9: 'Roland Reber'
        Context 10: [['The Return of Doctor X', 'written_by', 'William J. Makin'], ['The Return of Doctor X', 'release_year', '1939'], ['The Return of Doctor X', 'has_tags', 'humphrey bogart'], ] Question 10: what was the release year of the film [The Return of Doctor X]?
        Answer 10: '1939'
        Context 11: [['Topper', 'has_tags', 'ghosts'], ['Topper', 'has_tags', 'norman z. mcleod'], ['Topper', 'has_genre', 'Comedy'], ] Question 11: which topics is movie [Topper] about?
        Answer 11: 'ghosts'
        Context 12: [['The Mouse on the Moon', 'has_tags', 'bd-r'], ['The Mouse on the Moon', 'has_genre', 'Comedy'], ['The Mouse on the Moon', 'written_by', 'Leonard Wibberley'], ] Question 12: describe the movie [The Mouse on the Moon] in a few words?
        Answer 12: 'bd-r'\n"""
        
        
              
        prompt+='Now answer the following '+str(len(question_list))+' questions in the same way of these examples.\n'
        j=0
        for question in question_list:
            j+=1
            prompt+='Context '+str(j)+f': {context_texts[0]}'+f' Question '+str(j)+f': {question}'+'\n'
        prompt+='Answer '+str(j)+': '     
        result = llm(prompt)
        context_answer_list=len(question_list)*["No correct answer"]
        context_correct_list=len(question_list)*[False]
        answer_list=result.split('\n')
        answer_list = [item for item in answer_list if item != ""]
        if len(answer_list)==len(question_list):
            break
    if len(answer_list)!=len(question_list):
        return "answer length error"
    for j in range(len(question_list)):
        for lab in ground_truth_list[j]:
            if lab.lower() in answer_list[j].lower():
                context_answer_list[j] = lab.lower()
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
k1=5
#k2=4
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
    
    subgraph=list(build_subgraph(entity_set,kg))
    triplets=[]
    cossim=[]
    i=0
    pos_list=[]
    question_input = tokenizer([question], return_tensors="pt", padding=True, truncation=True).to(device)
    question_embedding = question_model(**question_input).last_hidden_state.mean(dim=1)
    for triplet in subgraph:
        if len(triplet)==3:
            if triplet[1][0]=='~':
                pos=triplet[2]+' '+triplet[1][1:]+' '+triplet[0]+'.'
            else:
                pos=triplet[0]+' '+triplet[1]+' '+triplet[2]+'.'
        elif len(triplet)==5:
            if triplet[1][0]=='~':
                pos=triplet[2]+' '+triplet[1][1:]+' '+triplet[0]+', '
            else:
                pos=triplet[0]+' '+triplet[1]+' '+triplet[2]+', '
            if triplet[3][0]=='~':
                pos+=triplet[4]+' '+triplet[3][1:]+' '+triplet[2]+'.'
            else:
                pos+=triplet[2]+' '+triplet[3]+' '+triplet[4]+'.'
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
    indexed_lst = list(enumerate(cossim))
    sorted_lst = sorted(indexed_lst, key=lambda x: x[1], reverse=True)
    already_got_list=[]
    values=[]
    
    context_texts="["
    for index, value in sorted_lst:
        already_got_list.append((subgraph[index][1],subgraph[index][2]))
        triplets.append(subgraph[index])
        values.append(value)
        if len(subgraph[index])==3:
            if subgraph[index][1][0]=='~':
                context_texts+='['+subgraph[index][2]+', '+subgraph[index][1][1:]+', '+subgraph[index][0]+'], '
            else:
                context_texts+='['+subgraph[index][0]+', '+subgraph[index][1]+', '+subgraph[index][2]+'], '
        elif len(subgraph[index])==5:
            if subgraph[index][1][0]=='~':
                context_texts+='['+subgraph[index][2]+', '+subgraph[index][1][1:]+', '+subgraph[index][0]+'], '
            else:
                context_texts+='['+subgraph[index][0]+', '+subgraph[index][1]+', '+subgraph[index][2]+'], '
            if subgraph[index][3][0]=='~':
                context_texts+='['+subgraph[index][4]+', '+subgraph[index][3][1:]+', '+subgraph[index][2]+'], '
            else:
                context_texts+='['+subgraph[index][2]+', '+subgraph[index][3]+', '+subgraph[index][4]+'], '
        
        if len(already_got_list)>=k1:
            break
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