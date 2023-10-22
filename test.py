from datasets import load_dataset
import fig
import time
from torch.optim.lr_scheduler import LambdaLR

import torch
import numpy as np
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

def train_switch_base_8():
    from rouge_score import rouge_scorer
    # 创建ROUGE评分器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    dataset = load_dataset("samsum")

    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=5e-05,
                                betas=(0.9,0.999),
                                eps=1e-08)
    # t_total = args.num_steps
    scheduler = WarmupLinearSchedule(optimizer, 3, 6)
    # scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    model.train()

    # model = reload_model(method[-1],model,0)
    model = model.cuda()
    index = 0
    batch_size = 1
    scores_list = {'rouge1':0,'rouge2':0,'rougeL':0}
    dataset_length = len(dataset['test'])
    for i in range(0,dataset_length,batch_size):
        
        batchs = dataset['test'][i:i+batch_size]
        doc, sum, _ = batchs['dialogue'], batchs['summary'], batchs['id']

        input_ids = tokenizer(doc, truncation=True,padding=True, return_tensors="pt").input_ids 
        input_ids = input_ids.cuda()
        
        # input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids

        labels = tokenizer.batch_encode_plus(sum, truncation=True,padding=True, return_tensors="pt")
        # input_ids['input_ids'] = input_ids['input_ids'].cuda()
        # input_ids['attention_mask'] = input_ids['attention_mask'].cuda()
        labels['input_ids'] = labels['input_ids'].cuda()

        outputs = model(input_ids=input_ids, labels=labels['input_ids'])
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        optimizer.step()
        scheduler.step()
        # optimizer.step()
        optimizer.zero_grad()

        outputs = list(outputs.squeeze(0).cpu().numpy())
        outputs = tokenizer.convert_ids_to_tokens(outputs)
        outputs = [token for token in outputs if not token.startswith("<") and not token.endswith(">")]
        outputs = tokenizer.convert_tokens_to_string(outputs)
        scores = scorer.score(outputs, sum[0])
        for key in scores.keys():
            scores_list[key] += scores[key].fmeasure/dataset_length
        print(scores_list)

        index += 1
    print('scores: ',scores_list)
train_switch_base_8()

def eval_score():
    from rouge_score import rouge_scorer
    # 创建ROUGE评分器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    dataset = load_dataset("samsum")

    tokenizer = AutoTokenizer.from_pretrained("emre/switch-base-8-finetuned-samsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("emre/switch-base-8-finetuned-samsum")
    
    model.eval()

    # model = reload_model(method[-1],model,0)
    model = model.cuda()
    index = 0
    batch_size = 1
    scores_list = {'rouge1':0,'rouge2':0,'rougeL':0}
    dataset_length = len(dataset['test'])
    for i in range(0,dataset_length,batch_size):
        
        batchs = dataset['test'][i:i+batch_size]
        doc, sum, _ = batchs['dialogue'], batchs['summary'], batchs['id']
        input_ids = tokenizer(doc, return_tensors="pt").input_ids 
        input_ids = input_ids.cuda()
        outputs = model.generate(input_ids,token_aggregate_layer=0,fuse_emb_method=0)  
        outputs = list(outputs.squeeze(0).cpu().numpy())
        outputs = tokenizer.convert_ids_to_tokens(outputs)
        outputs = [token for token in outputs if not token.startswith("<") and not token.endswith(">")]
        outputs = tokenizer.convert_tokens_to_string(outputs)
        scores = scorer.score(outputs, sum[0])
        for key in scores.keys():
            scores_list[key] += scores[key].fmeasure/dataset_length
        print(scores_list)
        index += 1
    print('scores: ',scores_list)
# eval_score()

def batch_eval_score(method,start_layer):
    from rouge_score import rouge_scorer
    from cluster_expert import reload_model
    # 创建ROUGE评分器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    dataset = load_dataset("samsum")

    tokenizer = AutoTokenizer.from_pretrained("emre/switch-base-8-finetuned-samsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("emre/switch-base-8-finetuned-samsum")
    # method = ['original', 'naive_8_to_1', 'naive_8_to_2', 'naive_8_to_4']
    # start_layer = [0,1,2,3,4,5,6,7,8,9,10,11]
    
    model.eval()

    model = reload_model(method,model,start_layer)
    model = model.cuda()
    index = 0
    batch_size = 12
    scores_list = {'rouge1':0,'rouge2':0,'rougeL':0}
    dataset_length = len(dataset['test'])
    for i in range(0,dataset_length,batch_size):
        batchs = dataset['test'][i:i+batch_size]
        doc, sum, _ = batchs['dialogue'], batchs['summary'], batchs['id']
        input_ids = tokenizer.batch_encode_plus(doc, truncation=True,padding=True, return_tensors="pt") 
        input_ids['input_ids'] = input_ids['input_ids'].cuda()
        input_ids['attention_mask'] = input_ids['attention_mask'].cuda()
        # input_ids = input_ids.cuda()
        outputs = model.generate(**input_ids) 
        outputs = outputs.squeeze(0).cpu().numpy()
        for output_index in range(0,len(sum)):
            output = list(outputs[output_index])
            output = tokenizer.convert_ids_to_tokens(output)
            output = [token for token in output if not token.startswith("<") and not token.endswith(">")]
            output = tokenizer.convert_tokens_to_string(output)
            scores = scorer.score(output, sum[output_index])
            for key in scores.keys():
                scores_list[key] += scores[key].fmeasure/dataset_length
            index += 1
            # print(scores_list,index)
    del model
    del dataset
    del tokenizer
    return scores_list

def batch_token_eval_score(method,start_layer,bs=0):
    from rouge_score import rouge_scorer
    # from cluster_expert import reload_model
    # 创建ROUGE评分器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    dataset = load_dataset("samsum")

    tokenizer = AutoTokenizer.from_pretrained("emre/switch-base-8-finetuned-samsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("emre/switch-base-8-finetuned-samsum")
    # method = ['original', 'naive_8_to_1', 'naive_8_to_2', 'naive_8_to_4']
    # start_layer = [0,1,2,3,4,5,6,7,8,9,10,11]
    
    model.eval()

    # model = reload_model(method,model,start_layer)
    model = model.cuda()
    index = 0
    if bs != 0:
        batch_size = bs
    else:
        batch_size = 8
    scores_list = {'rouge1':0,'rouge2':0,'rougeL':0}
    dataset_length = len(dataset['test'])

    time_list = []
    aggregation_time_list = []
    for i in range(0,dataset_length,batch_size):
        batchs = dataset['test'][i:i+batch_size]
        doc, sum_1, _ = batchs['dialogue'], batchs['summary'], batchs['id']
        input_ids = tokenizer.batch_encode_plus(doc, truncation=True,padding=True, return_tensors="pt") 
        input_ids['input_ids'] = input_ids['input_ids'].cuda()
        input_ids['attention_mask'] = input_ids['attention_mask'].cuda()
        # input_ids = input_ids.cuda()

        time_start = time.perf_counter()
        outputs = model.generate(**input_ids,token_aggregate_layer=start_layer*2,fuse_emb_method=method,return_dict_in_generate=True) 
        time_stop = time.perf_counter()
        time_list.append(time_stop - time_start)
        aggregation_time_list.append(sum(outputs.total_aggregation_time))
        outputs = outputs.sequences.squeeze(0).cpu().numpy()
        for output_index in range(0,len(sum_1)):
            if len(sum_1) == 1:
                output = list(outputs)
            else:
                output = list(outputs[output_index])
            output = tokenizer.convert_ids_to_tokens(output)
            output = [token for token in output if not token.startswith("<") and not token.endswith(">")]
            output = tokenizer.convert_tokens_to_string(output)
            scores = scorer.score(output, sum_1[output_index])
            for key in scores.keys():
                scores_list[key] += scores[key].fmeasure/dataset_length
            index += 1
            print(scores_list, len(time_list), len(aggregation_time_list), index)
        # break
    # A = sum(time_list)
    # print(A)
    del model
    del dataset
    del tokenizer
    return (scores_list, time_list, aggregation_time_list)

# batch_token_eval_score(0,0)
# a1,b1,c1 = batch_token_eval_score(2,0,8)
# a2,b2,c2 = batch_token_eval_score(2,20,8)
# b2,c2=0,0
# print(sum(b1),sum(c1),sum(b2),sum(c2))
# print(batch_token_eval_score(2,6))

#     aggreagation expert code
# methods = ['original', 'naive_8_to_1', 'naive_8_to_2', 'naive_8_to_4']



# methods = [1,2]
# start_layers = [0,1,2,3,4,5,6,7,8,9,10,11,20]
# batch_size = [1,2,4,6,8]
# score_dict = {}
# for method in methods:
#     # if method == 0:
#     #     score_list = batch_token_eval_score(method,0)
#     #     score_dict[str(method)+'_'+str(0)] = score_list
#     #     with open("./experiment/token_aggregate_dict_time.pickle", "wb") as file:
#     #         print(score_dict)
#     #         pickle.dump(score_dict, file)
#     #     continue

#     # if method == 1:
#     #     continue
#     # for start_layer in start_layers:
#     #     score_list = batch_token_eval_score(method,start_layer)
#     #     score_dict[str(method)+'_'+str(start_layer)] = score_list
#     #     with open("./experiment/token_aggregate_dict_time.pickle", "wb") as file:
#     #         print(score_dict)
#     #         pickle.dump(score_dict, file)

#     if method == 1:
#         continue
#     for bs in batch_size:
#         score_list = batch_token_eval_score(method,0,bs)
#         score_dict['aggregation_batch_size_'+str(bs)] = score_list
#         with open("./experiment/token_aggregate_dict_time_bs.pickle", "wb") as file:
#             print(score_dict)
#             pickle.dump(score_dict, file)

#         score_list = batch_token_eval_score(method,20,bs)
#         score_dict['non_aggregation_batch_size_'+str(bs)] = score_list
#         with open("./experiment/token_aggregate_dict_time_bs.pickle", "wb") as file:
#             print(score_dict)
#             pickle.dump(score_dict, file)



def expert_dis():
    dataset = load_dataset("samsum")

    tokenizer = AutoTokenizer.from_pretrained("emre/switch-base-8-finetuned-samsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("emre/switch-base-8-finetuned-samsum")

    model = model.cuda()
    model.eval()
    index = 0
    batch_size = 1

    expert_list,token_list = [], []

    for i in range(0,len(dataset['test']),batch_size):
        
        batchs = dataset['test'][i:i+batch_size]
        doc, sum, id = batchs['dialogue'], batchs['summary'], batchs['id']
        input_ids = tokenizer.batch_encode_plus(doc, truncation=True,padding=True, return_tensors="pt") 
        labels = tokenizer.batch_encode_plus(sum, truncation=True,padding=True, return_tensors="pt")
        # input_ids = tokenizer(doc, return_tensors="pt").input_ids 
        # labels = tokenizer(sum, return_tensors="pt").input_ids 
        input_ids['input_ids'] = input_ids['input_ids'].cuda()
        input_ids['attention_mask'] = input_ids['attention_mask'].cuda()
        labels['input_ids'] = labels['input_ids'].cuda()
        # print(doc,sum,id)
        outputs = model(**input_ids,labels=labels['input_ids']) 
        # outputs = model(**input_ids, labels=labels['input_ids'])
        
        encoder_router = outputs.encoder_router_logits
        decoder_router = outputs.decoder_router_logits
        # encoder_token = outputs.encoder_token_embed
        # decoder_token = outputs.decoder_token_embed

        if index == 0:
            for i in encoder_router:
                if len(i) == 2:
                    print(i[1].shape)
                    expert_list.append([i[1]])
            for i in decoder_router:
                if len(i) == 2:
                    print(i[1].shape)
                    expert_list.append([i[1]])  

            # for i in encoder_token:
            #     if not isinstance(i, tuple):
            #         print(i.shape)
            #         token_list.append(i)
            # for i in decoder_token:
            #     if not isinstance(i, tuple):
            #         print(i.shape)
            #         token_list.append(i)  
        else:
            k = 0
            for i in encoder_router:
                if len(i) == 2:
                    print(i[1].shape)
                    expert_list[k].append(i[1])
                    k+=1
            for i in decoder_router:
                if len(i) == 2:
                    print(i[1].shape)
                    expert_list[k].append(i[1])
                    k+=1
            # k = 0
            # for i in encoder_token:
            #     if not isinstance(i, tuple):
            #         print(i.shape)
            #         token_list[k] = torch.hstack((token_list[k], i))
            #         k+=1
            # for i in decoder_token:
            #     if not isinstance(i, tuple):
            #         print(i.shape)
            #         token_list[k] = torch.hstack((token_list[k], i))
            #         k+=1

        # outputs = model.generate(input_ids)
        # print(outputs.shape)
        
        # outputs = list(outputs.squeeze(0).cpu().numpy())
        # outputs = tokenizer.batch_decode(outputs)
        # outputs = tokenizer.convert_ids_to_tokens(outputs)
        # outputs = tokenizer.convert_tokens_to_string(outputs)
        # print(sum, outputs)

        index += 1

        # if index == 10:
        #     break
    del model
    del dataset
    del tokenizer
    return expert_list
    # fig.tokenpcafig(expert_list,token_list,index)
# data = expert_dis()
# with open("./experiment/router_dict.pkl", "wb") as file:
#     # print(score_dict)
#     pickle.dump(data, file)
# fig.expert_times()