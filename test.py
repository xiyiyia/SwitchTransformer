from datasets import load_dataset
import fig
import time
from torch.optim.lr_scheduler import LambdaLR

import torch
import numpy as np
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
import wandb

from transformers import TrainingArguments, Trainer
from rouge_score import rouge_scorer
import numpy as np
import evaluate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler,DataCollatorForSeq2Seq
import os

import nltk

def train_switch_base_8():
    def save_model(model,name):
        model_to_save = model.module if hasattr(model, 'module') else model
        model_checkpoint = os.path.join('/home/ubuntu/SwitchTransformer/pth/', "%s_checkpoint.bin" % name)
        torch.save(model_to_save.state_dict(), model_checkpoint)
        print("Saved model checkpoint to [DIR: /home/ubuntu/SwitchTransformer/pth/]")
        # logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)
    # nltk.download('punkt')
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    
    dataset = load_dataset("samsum")
    metric = evaluate.load("rouge")

    # tokenizer = AutoTokenizer.from_pretrained("emre/switch-base-8-finetuned-samsum")
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
    # model = AutoModelForSeq2SeqLM.from_pretrained("emre/switch-base-8-finetuned-samsum")

    max_input_length = 1024
    max_target_length = 128

    def preprocess_function(examples):
        # inputs = [doc for doc in examples['dialogue']]
        model_inputs = tokenizer(examples['dialogue'], max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        labels = tokenizer(text_target=examples["summary"], max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    wandb.init(    # set the wandb project where this run will be logged
    project="switch-8-samsum",
    
    # track hyperparameters and run metadata
    
    config={
    "learning_rate": 5e-05,
    "architecture": "switch-8",
    "dataset": "samsum",
    "epochs": 10,
    }
    )
    
    batch_size=6
    # dataset = load_dataset("yelp_review_full")
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    
    tokenized_datasets.set_format("torch")
    # tokenized_datasets = tokenized_datasets.remove_columns(["valid"])
    tokenized_datasets = tokenized_datasets.remove_columns(["dialogue"])
    tokenized_datasets = tokenized_datasets.remove_columns(["id"])
    tokenized_datasets = tokenized_datasets.remove_columns(["summary"])
    
    train_dataset = tokenized_datasets["train"].shuffle(seed=42) # .select(range(1000))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42) # .select(range(1000))

    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=5e-05,
                                betas=(0.9,0.999),
                                eps=1e-08)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    # lr_scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # model.train()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    progress_bar = tqdm(range(num_training_steps))
    model = model.to(device)
    best_acc = 0
    model_name='switch_samsum_8'
    
    for epoch in range(num_epochs):
        model.train()
        step = 0
        loss_all = 0
        for batch in train_dataloader:
            # break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            step += 1
            del batch
            del loss
            wandb.log({'batch_loss': loss_all/step})
            
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model.generate(batch['input_ids'])# (**batch)

            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        result = metric.compute()

        wandb.log({'loss': loss_all/step, 'rouge1': result['rouge1']})
        if best_acc < result['rouge1']:
            save_model(model,model_name)
            best_acc = result['rouge1']
        # break
    # print(result)
    wandb.finish()
    del model
    del dataset
    del tokenizer

train_switch_base_8()

def eval_score():
    from rouge_score import rouge_scorer
    # 创建ROUGE评分器
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    dataset = load_dataset("samsum")

    tokenizer = AutoTokenizer.from_pretrained("emre/switch-base-8-finetuned-samsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("emre/switch-base-8-finetuned-samsum")
    metric = evaluate.load("rouge")
    model.eval()

    # model = reload_model(method[-1],model,0)
    model = model.cuda()
    index = 0
    batch_size = 1
    scores_list1 = {'rouge1':0,'rouge2':0,'rougeL':0,'rougeLsum':0}
    scores_list = {'rouge1':{},'rouge2':{},'rougeL':{}}
    for key in scores_list.keys():
        for keys in range(0,3):
            if keys == 0:
                scores_list[key]['precision'] =0
            if keys == 1:
                scores_list[key]['recall'] =0
            if keys == 2:
                scores_list[key]['fmeasure'] =0
                
    dataset_length = len(dataset['test'])
    for i in range(0,dataset_length,batch_size):
        
        batchs = dataset['test'][i:i+batch_size]
        doc, sum, _ = batchs['dialogue'], batchs['summary'], batchs['id']
        input_ids = tokenizer(doc, return_tensors="pt").input_ids 
        input_ids = input_ids.cuda()
        outputs = model.generate(input_ids)  
        outputs = list(outputs.squeeze(0).cpu().numpy())
        outputs = tokenizer.convert_ids_to_tokens(outputs)
        outputs = [token for token in outputs if not token.startswith("<") and not token.endswith(">")]
        outputs = tokenizer.convert_tokens_to_string(outputs)
        # scores1 = metric.compute(predictions=outputs, references=sum[0][0:len(outputs)])
        scores = scorer.score(outputs, sum[0])
        # print(outputs, sum[0])
        # for key in scores1.keys():
        #     scores_list1[key] += scores1[key]/dataset_length
        for key in scores.keys():
            for keys in range(0,3):
                if keys == 0:
                    scores_list[key]['precision'] += scores[key].precision/dataset_length
                if keys == 1:
                    scores_list[key]['recall'] += scores[key].recall/dataset_length
                if keys == 2:
                    scores_list[key]['fmeasure'] += scores[key].fmeasure/dataset_length
            # scores_list[key] += scores[key]/dataset_length
        
        index += 1
    # scores_list = metric.compute(use_stemmer=True)
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