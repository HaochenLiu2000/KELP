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

question_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
question_model.load_state_dict(torch.load('model.pth'))
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
question_model.to(device)
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

openai.api_key = open_file('./openai_api_key.txt')
with open('data/metaqa_kg.pickle', 'rb') as f:
    kg = pickle.load(f)
hop=2
                        
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
                    for relation2, object2 in knowledge_graph[obj].items():
                        for obj2 in knowledge_graph[obj][relation2]:
                            subgraph.add((str(entity), str(relation), str(obj), str(relation2), str(obj2)))
        
    return subgraph

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def context_query(question_list,ground_truth_list,context_texts):
    #print(question_list)
    #print(ground_truth_list)
    #print(context_texts)
    
    #prompt = open_file('./meta_2hop_prompts/verify_claim_with_evidence.txt').replace('<<<<CLAIM>>>>', question_list[0]).replace('<<<<EVIDENCE_SET>>>>', context_texts[0])
    
    #print(prompt)
    for _ in range(3):
        prompt="""Answer the following questions.\n The context is the evidence of triplets may help your verifying.\n
                Each context contains triplets in the form of [head, relation, tail] and it means "head's relation is tail.".
                If you think a question can have multiple answers, you must choose one and answer it. Enter when you start answering the next question. Examples:\n
                """
        prompt+="""
        Context 1: [['The Grey Fox', 'directed_by', 'Phillip Borsos'], ['The Grey Fox', 'release_year', '1982'], ['One Magic Christmas', 'written_by', 'Phillip Borsos'], ['One Magic Christmas', 'release_year', '1985'], ] Question 1: when were the movies written by [Phillip Borsos] released?
        Answer 1: '1985'
        Context 2: [['Jesus Henry Christ', 'directed_by', 'Dennis Lee'], ['Jesus Henry Christ', 'has_genre', 'Comedy'], ['Jesus Henry Christ', 'written_by', 'Dennis Lee'], ['Fireflies in the Garden', 'written_by', 'Dennis Lee'], ] Question 2: which movies share the screenwriter with [Jesus Henry Christ]?
        Answer 2: 'Fireflies in the Garden'
        Context 3: [['The Counselor', 'directed_by', 'Ridley Scott'], ['Legend', 'directed_by', 'Ridley Scott'], ['Body of Lies', 'directed_by', 'Ridley Scott'], ['Blade Runner', 'directed_by', 'Ridley Scott'], ['Someone to Watch Over Me', 'directed_by', 'Ridley Scott'], ['Gladiator', 'directed_by', 'Ridley Scott'], ['Black Hawk Down', 'directed_by', 'Ridley Scott'], ['Black Rain', 'directed_by', 'Ridley Scott'], ['Robin Hood', 'directed_by', 'Ridley Scott'], ['Gladiator', 'directed_by', 'Rowdy Herrington'], ] Question 3: which directors co-directed movies with [Ridley Scott]?
        Answer 3: 'Rowdy Herrington'
        Context 4: [['First Monday in October', 'written_by', 'Robert E. Lee'], ['First Monday in October', 'written_by', 'Jerome Lawrence'], ['Inherit the Wind', 'written_by', 'Jerome Lawrence'], ] Question 4: the scriptwriter of [First Monday in October] also wrote movies?
        Answer 4: 'Inherit the Wind'
        Context 5: [['Tale of Tales', 'directed_by', 'Yuriy Norshteyn'], ['Tale of Tales', 'written_by', 'Yuriy Norshteyn'], ['Hedgehog in the Fog', 'written_by', 'Sergei Kozlov'], ['Hedgehog in the Fog', 'directed_by', 'Yuriy Norshteyn'], ] Question 5: which person wrote the films directed by [Yuriy Norshteyn]?
        Answer 5: 'Sergei Kozlov'
        Context 6: [['Ronal the Barbarian', 'written_by', 'Philip Einstein Lipski'], ['Ronal the Barbarian', 'directed_by', 'Kresten Vestbjerg Andersen'], ['Ronal the Barbarian', 'written_by', 'Kresten Vestbjerg Andersen'], ['Ronal the Barbarian', 'written_by', 'Thorbjørn Christoffersen'], ] Question 6: who are the writers of the movies directed by [Kresten Vestbjerg Andersen]?
        Answer 6: 'Philip Einstein Lipski'
        Context 7: [['Novocaine', 'has_genre', 'Comedy'], ['Novocaine', 'written_by', 'David Atkins'], ['Novocaine', 'directed_by', 'David Atkins'], ] Question 7: the movies directed by [David Atkins] were in which genres?
        Answer 7: 'Comedy'
        Context 8: [['Man of the House', 'release_year', '2005'], ['Man of the House', 'written_by', 'Scott Lobdell'], ['Man of the House', 'release_year', '1995'], ['Man of the House', 'has_genre', 'Comedy'], ] Question 8: the films written by [Scott Lobdell] were released in which years?
        Answer 8: '1995'
        Context 9: [['Terence Hill', 'starred_actors', 'They Call Me Trinity'], ['Terence Hill', 'starred_actors', 'They Call Me Renegade'], ['Terence Hill', 'starred_actors', 'Go for It'], ['They Call Me Renegade', 'in_language', 'Italian'], ] Question 9: what are the languages spoken in the films starred by [Terence Hill]?
        Answer 9: 'Italian'
        Context 10: [['Project X', 'directed_by', 'Jonathan Kaplan'], ['Project X', 'starred_actors', 'Jonathan Daniel Brown'], ['Project X', 'starred_actors', 'Oliver Cooper'], ['Project X', 'directed_by', 'Nima Nourizadeh'], ['Project X', 'written_by', 'Matt Drake'], ] Question 10: who is listed as director of [Oliver Cooper] acted films?
        Answer 10: 'Nima Nourizadeh'
        Context 11: [['The Dream Team', 'starred_actors', 'Michael Keaton'], ['The Dream Team', 'starred_actors', 'Peter Boyle'], ['The Dream Team', 'starred_actors', 'Christopher Lloyd'], ['The Dream Team', 'directed_by', 'Howard Zieff'], ['The Dream Team', 'written_by', 'David Loucka'], ['The Dream Team', 'starred_actors', 'Stephen Furst'], ] Question 11: who co-starred with [Stephen Furst]?
        Answer 11: 'Peter Boyle'
        Context 12: [['Casey Jones', 'written_by', 'Polaris Banks'], ['Casey Jones', 'directed_by', 'Polaris Banks'], ['Casey Jones', 'has_genre', 'Short'], ['Casey Jones', 'release_year', '2011'], ] Question 12: what types are the films written by [Polaris Banks]?
        Answer 12: 'Short'\n"""
        
        
              
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
    already_got=[0]*int(k1-1)
    already_got_list=[]
    values=[]
    
    context_texts="["
    for index, value in sorted_lst:
        if (subgraph[index][1],subgraph[index][2]) not in already_got_list:
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
        else:
            for i in range(len(already_got_list)):
                if (len(subgraph[index])==5) and (already_got_list[i]==(subgraph[index][1],subgraph[index][2])) and (already_got[i]<k2-1):
                    already_got[i]+=1
                    triplets.append((subgraph[index][2],subgraph[index][3],subgraph[index][4]))
                    values.append(value)
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