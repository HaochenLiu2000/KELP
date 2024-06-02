import pickle
import json
import random
import openai
import os
from tqdm import tqdm
import time
openai.api_key = os.environ["OPENAI_API_KEY"]
with open('data/metaqa_kg.pickle', 'rb') as f:
    kg = pickle.load(f)
questions_dict = {}
entity_set_dict = {}
label_set_dict = {}
hop=1

if hop==1:
    question_data='data/onehop_train_set.jsonl'
    save_file='onehop.json'
if hop==2:
    question_data='data/twohop_train_set.jsonl'
    save_file='twohop.json'

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


def build_subgraph_rels_ranking(question, entity_set, knowledge_graph):
    one_hop_neighbors = get_one_hop_neighbors_rels_ranking(question, entity_set, knowledge_graph)
    if one_hop_neighbors=="No":
        return "No"
    two_hop_neighbors = get_two_hop_neighbors_rels_ranking(question, one_hop_neighbors, knowledge_graph)
    if two_hop_neighbors=="No":
        return "No"
    if hop==1:
        return one_hop_neighbors#+two_hop_neighbors
    #if hop==2:
    #    #return one_hop_neighbors+two_hop_neighbors
    #    return two_hop_neighbors

def get_one_hop_neighbors_rels_ranking(question, entity_set, knowledge_graph):
    neighbors_rel = set()
    for entity in entity_set:
        if entity in knowledge_graph:
            for relation, object in knowledge_graph[entity].items():
                if (str(entity), str(relation)) not in neighbors_rel and (',' not in str(entity) and (',' not in str(relation))): 
                    neighbors_rel.add((str(entity), str(relation)))
    top_neighbors_rel=get_top_related_triplets_rels_ranking(question,neighbors_rel,1)
    if top_neighbors_rel=="No":
        return "No"
    neighbors_ent=[set() for _ in range(len(top_neighbors_rel))]
    num_ent=0
    for i in range(len(top_neighbors_rel)):
        if len(top_neighbors_rel[i])!=2:
            continue
        if top_neighbors_rel[i][0] in knowledge_graph and top_neighbors_rel[i][1] in knowledge_graph[top_neighbors_rel[i][0]]:
            for obj in knowledge_graph[top_neighbors_rel[i][0]][top_neighbors_rel[i][1]]:
                if (str(top_neighbors_rel[i][0]), str(top_neighbors_rel[i][1]), str(obj)) not in neighbors_ent[i] and (',' not in str(entity) and (',' not in str(relation))) and (',' not in str(obj)): 
                    neighbors_ent[i].add((str(top_neighbors_rel[i][0]), str(top_neighbors_rel[i][1]), str(obj)))
                    num_ent+=1
    top_neighbors_ent=get_top_related_triplets_ents_ranking(question,neighbors_ent,1,num_ent)
    if top_neighbors_ent=="No":
        return "No"
    if not top_neighbors_ent:
        return "No"
    
    #if top_neighbors_ent:
    #    a=random.choice(list(top_neighbors_ent))
    #    if len(a)==3:
    #        h0,r0,h_n=a
    #        if h_n in knowledge_graph:
    #            r_n=random.choice(list(knowledge_graph[h_n]))
    #            t_n=random.choice(list(knowledge_graph[h_n][r_n]))
    #            top_neighbors_ent.append((str(h_n), str(r_n), str(t_n)))
    
    
    return top_neighbors_ent

def get_two_hop_neighbors_rels_ranking(question, top_1hop_triplets, knowledge_graph):
    neighbors_rel = set()
    for triplet in top_1hop_triplets:
        try:
            h, r, t = triplet
        except Exception as e:
            return "No"
        if t in knowledge_graph:
            for relation, object in knowledge_graph[t].items():
                if relation!='~'+r and r!='~'+relation:
                    if (str(t), str(relation)) not in neighbors_rel and (',' not in str(t)) and (',' not in str(relation)):
                        neighbors_rel.add((str(t), str(relation)))
    top_neighbors_rel=get_top_related_triplets_rels_ranking(question,neighbors_rel,2)
    if top_neighbors_rel=="No":
        return "No"
    neighbors_ent=[set() for _ in range(len(top_neighbors_rel))]
    num_ent=0
    for i in range(len(top_neighbors_rel)):
        if top_neighbors_rel[i][0] in knowledge_graph and top_neighbors_rel[i][1] in knowledge_graph[top_neighbors_rel[i][0]]:
            for obj in knowledge_graph[top_neighbors_rel[i][0]][top_neighbors_rel[i][1]]:
                if (str(top_neighbors_rel[i][0]), str(top_neighbors_rel[i][1]), str(obj)) not in neighbors_ent[i]  and (',' not in str(top_neighbors_rel[i][0])) and (',' not in str(top_neighbors_rel[i][1])) and (',' not in str(obj)):
                    neighbors_ent[i].add((str(top_neighbors_rel[i][0]), str(top_neighbors_rel[i][1]), str(obj)))
                    num_ent+=1
    top_neighbors_ent=get_top_related_triplets_ents_ranking(question,neighbors_ent,1,num_ent)
    if top_neighbors_ent=="No":
        return "No"
    for i in range(len(top_neighbors_ent)):
        for triplet in top_1hop_triplets:
            h, r, t = triplet
            if t==top_neighbors_ent[i][0]:
                top_neighbors_ent[i]=(str(h),str(r),str(top_neighbors_ent[i][0]),str(top_neighbors_ent[i][1]),str(top_neighbors_ent[i][2]))
                break
    if top_1hop_triplets:
        h0,r0,h_n=random.choice(list(top_1hop_triplets))
        if h_n in knowledge_graph:
            r_n=random.choice(list(knowledge_graph[h_n]))
            t_n=random.choice(list(knowledge_graph[h_n][r_n]))
            top_neighbors_ent.append((str(h0),str(r0),str(h_n), str(r_n), str(t_n)))
    
    return top_neighbors_ent

def get_top_related_triplets_rels_ranking(question, triplets, hop):
    if len(triplets)<=5:
        ranked_triplets=[triplet for triplet in triplets]
        return ranked_triplets
    
    prompt = f"Each of these word sets shows an entity and one of its corresponding relation. Select the 5-top word sets which are most semantically related to a given question. You should list the selected word sets from rank 1 to rank 5. Your answer should be in the form of '(XXX,XXX);(XXX,XXX);(XXX,XXX);(XXX,XXX);(XXX,XXX)'. Question: {question}\nWord sets: "
    for triplet in triplets:
        prompt += f"{triplet};"
    
    response = llm(prompt,len(triplets)*50)
    if response is None:
        return "No"
    try:
        ranked_triplets = [tuple(word.strip('\'" ') for word in triplet.split('(')[1].split(')')[0].split(',')) for triplet in response.split(';') if triplet]
    except Exception as e:
        return "No"
    return ranked_triplets

def get_top_related_triplets_ents_ranking(question, neighbors_ent, hop,num_ent):
    ranked_triplets=[]
    if num_ent<=5:    
        for triplets in neighbors_ent:
            ranked_triplets+=[triplet for triplet in triplets]
        return ranked_triplets
    num_sel=0
    i=0
    while(num_sel<5 and i<len(neighbors_ent)):
        if len(neighbors_ent[i])==1:
            ranked_triplets+=[triplet for triplet in neighbors_ent[i]]
            num_sel+=1
            i+=1
        elif len(neighbors_ent[i])==2 and 5-num_sel>1:
            ranked_triplets+=[triplet for triplet in neighbors_ent[i]]
            num_sel+=2
            i+=1
        else:
            if 5-num_sel>1:
                prompt = f"These word sets shows the relations of some entities. Select the 2-top word sets which are most semantically related to a given question. You should list the selected word sets from rank 1 to rank 2. Your answer should be in the form of '(XXX,XXX,XXX);(XXX,XXX,XXX)'. Question: {question}\nWord sets: "
            else:
                prompt = f"These word sets shows the relations of some entities. Select the best word sets which are most semantically related to a given question. Your answer should be in the form of '(XXX,XXX,XXX)'. Question: {question}\nWord sets: "

            for triplet in neighbors_ent[i]:
                prompt += f"{triplet};"
            
            response = llm(prompt,30+len(neighbors_ent[i])*50)
            if response is None:
                return "No"
            try:
                add_info=[tuple(word.strip('\'" ') for word in triplet.split('(')[1].split(')')[0].split(',')) for triplet in response.split(';') if triplet]
            except Exception as e:
                return "No"
            ranked_triplets += add_info
            num_sel+=len(add_info)
            i+=1
    return ranked_triplets


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

original_prompt=open_file('./meta_'+str(hop)+'hop_prompts/verify_claim_no_evidence.txt')
context_prompt=open_file('./meta_'+str(hop)+'hop_prompts/verify_claim_with_evidence.txt')

def original_query(question,ground_truth):
    prompt = original_prompt.replace('<<<<CLAIM>>>>', question)
    
    result_original = llm(prompt)
    original_correct=False
    
    for lab in ground_truth:
        if lab.lower() in result_original.lower():
            original_correct=True
            break
    
    return original_correct

def context_query(question,ground_truth,triplet,already_pos,already_neg, pos_sam, neg_sam):
    context='['
    if len(triplet)==3:
        if triplet[1][0]=='~':
            context+='['+triplet[2]+', '+triplet[1][1:]+', '+triplet[0]+']'
        else:
            context+='['+triplet[0]+', '+triplet[1]+', '+triplet[2]+']'
    elif len(triplet)==5:
        if triplet[1][0]=='~':
            context+='['+triplet[2]+', '+triplet[1][1:]+', '+triplet[0]+'], '
        else:
            context+='['+triplet[0]+', '+triplet[1]+', '+triplet[2]+'], '
        if triplet[3][0]=='~':
            context+='['+triplet[4]+', '+triplet[3][1:]+', '+triplet[2]+']'
        else:
            context+='['+triplet[2]+', '+triplet[3]+', '+triplet[4]+']'
    else:
        return 'triplet error', already_pos, already_neg, pos_sam, neg_sam
    context+=']'
    prompt = context_prompt.replace('<<<<CLAIM>>>>', question).replace('<<<<EVIDENCE_SET>>>>', context)
    result = llm(prompt)
    context_answer="No correct answer"
    context_correct=False
    
    for lab in ground_truth:
        if lab.lower() in result.lower():
            context_answer = lab.lower()
            context_correct=True
            break
    
    if already_pos==False and context_correct==True:
        already_pos=True
        pos_sam=triplet
    if already_neg==False and context_correct==False:
        already_neg=True
        neg_sam=triplet
    
    return context_correct, already_pos, already_neg, pos_sam, neg_sam



dataset_len=len(questions_dict)
a=1
data_num=range(a,dataset_len+1,1)


for ii in tqdm(data_num):
    if ii%5!=0:
        continue
    question = questions_dict[ii]
    entity_set = entity_set_dict[ii]
    ground_truth =label_set_dict[ii]
    
    original_correct=original_query(question,ground_truth)
    if original_correct==True:
        continue
    
    subgraph = build_subgraph_rels_ranking(question, entity_set, kg)
    if subgraph=="No":
        continue
    
    num_triplets_to_test = len(subgraph)
    if num_triplets_to_test==0:
        continue
    
    
    
    
    already_pos,already_neg=False, False
    pos_sam,neg_sam=False, False
    tl=list(range(num_triplets_to_test))
    random.shuffle(tl)
    for i in tl:
        context_correct, already_pos, already_neg, pos_sam, neg_sam=context_query(question,ground_truth,subgraph[i],already_pos,already_neg, pos_sam, neg_sam)
        if context_correct=='triplet error':
            continue
        if pos_sam==True and neg_sam==True:
            break
    if already_pos==True and already_neg==True:
        result_dict={}
        result_dict['question_id']=ii
        result_dict['question']=question
        result_dict['entity_set']=entity_set
        result_dict['ground_truth']=ground_truth
        result_dict['pos_triplet']=pos_sam
        result_dict['neg_triplet']=neg_sam
        with open(save_file, 'a') as f:
            json.dump(result_dict, f, indent=5)
            f.write('\n')      