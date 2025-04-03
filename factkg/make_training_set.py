import pickle
import json
import random
import openai
import os
from tqdm import tqdm
import time

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

openai.api_key = open_file('./openai_api_key.txt')

with open('dbpedia_2015_undirected_light.pickle', 'rb') as f:
    kg = pickle.load(f)
  
questions_dict = {}
entity_set_dict = {}
label_set_dict = {}
with open('extracted_train_set.jsonl', 'r') as f:
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

    return one_hop_neighbors+two_hop_neighbors

def get_one_hop_neighbors_rels_ranking(question, entity_set, knowledge_graph):
    neighbors_rel = set()
    for entity in entity_set:
        if entity in knowledge_graph:
            for relation, object in knowledge_graph[entity].items():
                if (str(entity), str(relation)) not in neighbors_rel and (',' not in str(entity) and (',' not in str(relation))): 
                    neighbors_rel.add((str(entity), str(relation)))
    if len(neighbors_rel)>40:
        neighbors_rel=random.sample(neighbors_rel, 40)
    top_neighbors_rel=get_top_related_triplets_rels_ranking(question,neighbors_rel,1)
    if top_neighbors_rel=="No":
        return "No"
    if not top_neighbors_rel:
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
    for i in range(len(top_neighbors_rel)):
        if len(neighbors_ent[i])>20:
            neighbors_ent[i]=random.sample(neighbors_ent[i],20)
    top_neighbors_ent=get_top_related_triplets_ents_ranking(question,neighbors_ent,1,num_ent)
    if top_neighbors_ent=="No":
        return "No"
    if not top_neighbors_ent:
        return "No"
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
    if len(neighbors_rel)>40:
        neighbors_rel=random.sample(neighbors_rel, 40)
    top_neighbors_rel=get_top_related_triplets_rels_ranking(question,neighbors_rel,2)
    if top_neighbors_rel=="No":
        return "No"
    neighbors_ent=[set() for _ in range(len(top_neighbors_rel))]
    num_ent=0
        
    for i in range(len(top_neighbors_rel)):
        #print(i)
        if len(top_neighbors_rel[i])!=2:
            continue
        if top_neighbors_rel[i][0] in knowledge_graph and top_neighbors_rel[i][1] in knowledge_graph[top_neighbors_rel[i][0]]:
            for obj in knowledge_graph[top_neighbors_rel[i][0]][top_neighbors_rel[i][1]]:
                if (str(top_neighbors_rel[i][0]), str(top_neighbors_rel[i][1]), str(obj)) not in neighbors_ent[i]  and (',' not in str(top_neighbors_rel[i][0])) and (',' not in str(top_neighbors_rel[i][1])) and (',' not in str(obj)):
                    neighbors_ent[i].add((str(top_neighbors_rel[i][0]), str(top_neighbors_rel[i][1]), str(obj)))
                    num_ent+=1
    for i in range(len(top_neighbors_rel)):
        if len(neighbors_ent[i])>20:
            neighbors_ent[i]=random.sample(neighbors_ent[i],20)
    top_neighbors_ent=get_top_related_triplets_ents_ranking(question,neighbors_ent,1,num_ent)
    if top_neighbors_ent=="No":
        return "No"
    for i in range(len(top_neighbors_ent)):
        for triplet in top_1hop_triplets:
            h, r, t = triplet
            if len(top_neighbors_ent[i])==3 and t==top_neighbors_ent[i][0]:
                top_neighbors_ent[i]=(str(h),str(r),str(top_neighbors_ent[i][0]),str(top_neighbors_ent[i][1]),str(top_neighbors_ent[i][2]))
                break
    if top_1hop_triplets:
        h0,r0,h_n=random.choice(list(top_1hop_triplets))
        if h_n in knowledge_graph:
            r_n=random.choice(list(knowledge_graph[h_n]))
            t_n=random.choice(list(knowledge_graph[h_n][r_n]))
            top_neighbors_ent.append((str(h0),str(r0),str(h_n), str(r_n), str(t_n)))
        if h0 in knowledge_graph:
            r_n1=random.choice(list(knowledge_graph[h0]))
            t_n1=random.choice(list(knowledge_graph[h0][r_n1]))
            top_neighbors_ent.append((str(h0),str(r_n1),str(t_n1)))
    return top_neighbors_ent

def get_top_related_triplets_rels_ranking(question, triplets, hop):
    if len(triplets)<=5:
        ranked_triplets=[triplet for triplet in triplets]
        return ranked_triplets
    
    prompt = f"Each of these word sets shows an entity and one of its corresponding relation. Select the 5-top word sets which are most semantically related to the given sentence. You should list the selected word sets from rank 1 to rank 5. Your answer should be in the form of '(XXX,XXX);(XXX,XXX);(XXX,XXX);(XXX,XXX);(XXX,XXX)'. Sentence: {question}\nWord sets: "
    triplets_ran=triplets
    for _ in range(3):
        try:
            triplets_ran=random.sample(triplets_ran,len(triplets)-5)
            prompt1=prompt
            for triplet in triplets_ran:
                prompt1 += f"{triplet};"
            response = llm(prompt1,len(triplets_ran)*50)
        except Exception as e:
            continue
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
                prompt = f"These word sets shows the relations of some entities. Select the 2-top word sets which are most semantically related to the given sentence. You should list the selected word sets from rank 1 to rank 2. Your answer should be in the form of '(XXX,XXX,XXX);(XXX,XXX,XXX)'. Sentence: {question}\nWord sets: "
            else:
                prompt = f"These word sets shows the relations of some entities. Select the best word sets which are most semantically related to the given sentence. Your answer should be in the form of '(XXX,XXX,XXX)'. Sentence: {question}\nWord sets: "

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



def original_query(question_list,ground_truth_list):
    for _ in range(3):
        prompt="""Verify these claims. Is this claim True? or False?\n
                Choose one of {True, False}. Enter when you start answering the next question. Example:\n
                Claim: """
        prompt+="""
        Claim 1: Ahmad Kadhim Assad's club is Al-Zawra'a SC.
        Claim 2: Yeah! I know that a TV show, which starred Tim Brooke-Taylor, first aired on 3rd October 1983!
        Claim 3: Really? Jamie Lawrence is the music composer of the 83 minute 'Death on a Factory Farm' film, directed by Sarah Teale!
        Claim 4: 'A film' was produced by Anatole de Grunwald, directed by Harold French, with cinematography done by Bernard Knowles.
        Answer 1: True
        Answer 2: True
        Answer 3: False
        Answer 4: False
        Now answer the following questions. Do not repeat the question again and just response like the example.
        
        """        
                
        j=0
        for question in question_list:
            j+=1
            prompt+=f'Claim '+str(j)+f': {question}\n'        
        
        result_original = llm(prompt)
        original_answer_list=len(question_list)*["No correct answer"]
        original_correct_list=len(question_list)*[False]
        answer_list=result_original.split('\n')
        if len(answer_list)==len(question_list):
            break
    if len(answer_list)!=len(question_list):
        return "answer length error","answer length error"
    for i in range(len(question_list)):
        for lab in ground_truth_list[i]:
            if (lab and ('true' in answer_list[i].lower())) or ((not lab) and ('false' in answer_list[i].lower())):
                original_answer_list[i] = lab
                original_correct_list[i]=True
                break
    q_choose_list = [i for i, value in enumerate(original_correct_list) if value==False]
    
    return q_choose_list, original_correct_list

def context_query(question_list,ground_truth_list,subgraph_list,i,q_choose_list, already_pos_list, already_neg_list):
    question_list2=[question_list[j] for j in q_choose_list]
    ground_truth_list2=[ground_truth_list[j] for j in q_choose_list]
    subgraph_list2=[subgraph_list[j] for j in q_choose_list]
    context_texts=[]
    for subgraph in subgraph_list2:
        triplet=subgraph[i]
        context='['
        if len(triplet)==3:
            if str(triplet[1])[0]!='~':
                context+='['+str(triplet[0])+', '+str(triplet[1])+', '+str(triplet[2])+'], '
            else:
                context+='['+str(triplet[2])+', '+str(triplet[1])[1:]+', '+str(triplet[0])+'], '
        elif len(triplet)==5:
            if str(triplet[1])[0]!='~':
                context+='['+str(triplet[0])+', '+str(triplet[1])+', '+str(triplet[2])+'], '
            else:
                context+='['+str(triplet[2])+', '+str(triplet[1])[1:]+', '+str(triplet[0])+'], '
            if str(triplet[3])[0]!='~':
                context+='['+str(triplet[2])+', '+str(triplet[3])+', '+str(triplet[4])+'], '
            else:
                context+='['+str(triplet[4])+', '+str(triplet[3])[1:]+', '+str(triplet[2])+'], '
        context+=']'
        context_texts.append(context)
    for _ in range(3):    
        
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
            Answer 8: True, everything of the claim is supported by the evidence set.
            Context 9: [['Bananaman', 'starring', 'Bill_Oddie'], ['Bananaman', 'network', 'Broadcasting_House'], ['Bananaman', 'locationCity', 'Broadcasting_House'], ] Claim 9: Bananaman the TV series starred by a person was shown on the company and the company headquarters is called Broadcasting House.
            Answer 9: True, everything of the claim is supported by the evidence set.
            Context 10: [['Azerbaijan', 'leaderName', 'Artur_Rasizade'], ["Baku_Turkish_Martyrs'_Memorial", 'designer', '"Hüseyin Bütüner and Hilmi Güner"'], ["Baku_Turkish_Martyrs'_Memorial", 'location', 'Azerbaijan'], ] Claim 10: The place, designed by Huseyin Butuner and Hilmi Guner, is located in a country, where the leader is Artur Rasizade.
            Answer 10: True, everything of the claim is supported by the evidence set.
            Context 11: [['AIDAstella', 'shipBuilder', 'Meyer_Werft'], ['AIDAstella', 'shipOperator', 'AIDA_Cruises'], ] Claim 11: AIDA Cruise line operated the ship which was built by Meyer Werft in Townsend, Poulshot, Wiltshire.
            Answer 11: False, there is no evidence for Townsend, Poulshot, Wiltshire.
            Context 12: [[], ] Claim 12: An academic journal with code IJPHDE is also Acta Math. Hungar.
            Answer 12: False, there is no evidence that the academic journal is also Acta Math. Hungar.\n"""
        
              
        prompt+='Now verify the following '+str(len(question_list2))+' claims in the same way of these examples.\n'
        j=0
        for question in question_list2:
            j+=1
            prompt+='Context '+str(j)+f': {context_texts[0]}'+f' Claim '+str(j)+f': {question}'+'\n'
        prompt+='Answer '+str(j)+': '     
        
        
        result = llm(prompt)
        context_answer_list=len(question_list2)*["No correct answer"]
        context_correct_list=len(question_list2)*[False]
        answer_list=result.split('\n')
        if len(answer_list)==len(question_list2):
            break
    if len(answer_list)!=len(question_list2):
        return "answer length error","answer length error", already_pos_list, already_neg_list, pos_sam_list, neg_sam_list
    for j in range(len(question_list2)):
        for lab in ground_truth_list2[j]:
            if (lab and ('true' in answer_list[j].lower())) or ((not lab) and ('false' in answer_list[j].lower())):
                context_answer_list[j] = lab
                context_correct_list[j]=True
                break
    for j in range(len(question_list2)):
        if already_pos_list[q_choose_list[j]]==False and context_correct_list[j]==True:
            already_pos_list[q_choose_list[j]]=True
            pos_sam_list[q_choose_list[j]]=[subgraph_list[q_choose_list[j]][i], context_texts[j]]
    for j in range(len(question_list2)):
        if already_neg_list[q_choose_list[j]]==False and context_correct_list[j]==False:
            already_neg_list[q_choose_list[j]]=True
            neg_sam_list[q_choose_list[j]]=[subgraph_list[q_choose_list[j]][i], context_texts[j]]
    q_choose_list = [v for j, v in enumerate(q_choose_list) if (already_pos_list[v] and already_neg_list[v])==False and i+1<len(subgraph_list2[j])]
    return q_choose_list, context_correct_list, already_pos_list, already_neg_list, pos_sam_list, neg_sam_list


dataset_len=len(questions_dict)
a=1
data_num=list(range(1,dataset_len+1,10))+list(range(2,dataset_len+1,10))

question_id_list=[]
question_list=[]
entity_set_list=[]
ground_truth_list=[]
subgraph_list=[]
num_triplets_to_test_list=[]

questions_num_in_list=0
for ii in tqdm(data_num):
    question = questions_dict[ii]
    entity_set = entity_set_dict[ii]
    ground_truth =label_set_dict[ii]
    
    subgraph = build_subgraph_rels_ranking(question, entity_set, kg)
    
    if subgraph=="No":
        continue
    
    num_triplets_to_test = len(subgraph)
    if num_triplets_to_test==0:
        continue
    questions_num_in_list+=1
    question_id_list.append(ii)
    question_list.append(question)
    entity_set_list.append(entity_set)
    ground_truth_list.append(ground_truth)
    subgraph_list.append(subgraph)
    num_triplets_to_test_list.append(num_triplets_to_test)
    
    if questions_num_in_list>=1:
        q_choose_list, original_correct_list=original_query(question_list,ground_truth_list)
        if q_choose_list=="answer length error":
            question_id_list=[]
            question_list=[]
            entity_set_list=[]
            ground_truth_list=[]
            subgraph_list=[]
            num_triplets_to_test_list=[]
            questions_num_in_list=0
            continue
        
        already_pos_list,already_neg_list=[False]*len(question_list), [False]*len(question_list)
        pos_sam_list,neg_sam_list=[False]*len(question_list), [False]*len(question_list)
        for i in range(max(num_triplets_to_test_list)):
            if len(q_choose_list)>0:
                q_choose_list, context_correct_list, already_pos_list, already_neg_list, pos_sam_list, neg_sam_list=context_query(question_list,ground_truth_list,subgraph_list,i,q_choose_list,already_pos_list,already_neg_list)
            if q_choose_list=="answer length error":
                break
        with open('output.jsonl', 'a') as f:
            for i in range(len(question_list)):
                if already_pos_list[i] and already_neg_list[i]:
                    result_dict = {
                        'question_id': question_id_list[i],
                        'question': question_list[i],
                        'entity_set': entity_set_list[i],
                        'ground_truth': ground_truth_list[i],
                        'pos_triplet': pos_sam_list[i][0],
                        'pos_context': pos_sam_list[i][1],
                        'neg_triplet': neg_sam_list[i][0],
                        'neg_context': neg_sam_list[i][1]
                    }
                    json_line = json.dumps(result_dict)
                    f.write(json_line + '\n')
        
        question_id_list=[]
        question_list=[]
        entity_set_list=[]
        ground_truth_list=[]
        subgraph_list=[]
        num_triplets_to_test_list=[]
        questions_num_in_list=0