from datasets import load_dataset
import fig
import time
from torch.optim.lr_scheduler import LambdaLR

import torch
import numpy as np
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,SwitchTransformersForConditionalGeneration
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


config1 = {
    'Avg':{'model':'Avg_selection_model.bin','purne_layer':[1,3,5,7,9,11,13,15,17,19,21,23],'require_experts':
     [[5, 6, 7, 1], [1, 7, 0], [4, 7, 0], [2, 3, 6, 0], [2, 4, 0], [1, 2, 4, 0], [4, 0], [3, 6, 0], [3, 7, 0], [7, 5], [4, 6, 3], [7, 5]]},
    'Greedy':{'model':'greedy_selection_model.bin','purne_layer':[1,3,5,7,9,11,13,15,17,19,21,23],'require_experts':
                     [[5, 7, 6, 2, 0, 4, 3, 1], [7, 0, 5, 4, 6, 2, 1], [0, 4, 3, 5, 2, 7], 
                      [2, 0, 6, 7, 3], [0, 4, 7, 2], [4, 0, 2], [0, 4], [0, 3], [7, 3], [5, 7], [4, 6], [5, 7]]},
    
    None:{'model':'switch_samsum_8_checkpoint.bin'},
    '9-2':{
        'model':'switch_samsum_8_layer_9_expert_2_checkpoint.bin',
        'purne_layer':[9, 11, 13, 15,17,19,21, 23],
        'require_experts':[[0,2],[0],[0,1,2,3,4,6],[0,1,3,4,6],[1,2,3,4],[0,3,7],[0,7],[7]]},
        '11-1':{
            'model':'switch_samsum_8_layer_11_expert_1_checkpoint.bin',
        'purne_layer':[11, 13, 15,17,19,21, 23],
        'require_experts':[[0],[0,1,2,3,4,6],[0,1,3,4,6],[1,2,3,4],[0,3,7],[0,7],[7]]},
        '13-6':{
            'model':'switch_samsum_8_layer_15_expert_5_checkpoint.bin',
        'purne_layer':[13,15,17,19,21, 23],
        'require_experts':[[0,1,2,3,4,6],[0,1,3,4,6],[1,2,3,4],[0,3,7],[0,7],[7]]},
        '15-5':{
            'model':'switch_samsum_8_layer_15_expert_5_checkpoint.bin',
        'purne_layer':[13,15,17,19,21, 23],
        'require_experts':[[0,1,2,3,4,6],[0,1,3,4,6],[1,2,3,4],[0,3,7],[0,7],[7]]},
        '17-4':{
            'model':'switch_samsum_8_layer_17_expert_4_checkpoint.bin',
        'purne_layer':[17,19,21, 23],
        'require_experts':[[1,2,3,4],[0,3,7],[0,7],[7]]},
        '19-3':{
            'model':'switch_samsum_8_layer_19_expert_3_checkpoint.bin',
        'purne_layer':[19,21, 23],
        'require_experts':[[0,3,7],[0,7],[7]]},
        '21-2':{
            'model':'switch_samsum_8_layer_21_expert_2_checkpoint.bin',
        'purne_layer':[21, 23],
        'require_experts':[[0,7],[7]]},
        '23-1':{
            'model':'switch_samsum_8_layer_23_expert_1_checkpoint.bin',
        'purne_layer':[23],
        'require_experts':[[7]]}
        }


def purning_model(option=None):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("google/switch-base-8")
    config.pyramid=True

    if option is not None:
        config.purne_layer = config1[option]['purne_layer']
        config.require_experts = config1[option]['require_experts']

    mymodel = SwitchTransformersForConditionalGeneration(config=config)
    myparam_model = mymodel.state_dict()
    model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
    param_model = model.state_dict()
    gene_key = []
    if len(config.purne_layer) !=0:
        for layer in config.purne_layer:
            num_experts = config.require_experts[config.purne_layer.index(layer)]
            if layer > 11:
                for i in num_experts:
                    gene_key += ['decoder.block.'+str(layer-12)+'.layer.2.mlp.experts.expert_'+str(i)+'.wi.weight',\
                                'decoder.block.'+str(layer-12)+'.layer.2.mlp.experts.expert_'+str(i)+'.wo.weight']
                gene_key += ['decoder.block.'+str(layer)+'.layer.2.mlp.router.classifier.weight']
            else:
                for i in num_experts:
                    gene_key += ['encoder.block.'+str(layer)+'.layer.1.mlp.experts.expert_'+str(i)+'.wi.weight',\
                                'encoder.block.'+str(layer)+'.layer.1.mlp.experts.expert_'+str(i)+'.wo.weight']
                gene_key += ['encoder.block.'+str(layer)+'.layer.1.mlp.router.classifier.weight']
        for key in param_model.keys():
            if key not in gene_key and key in myparam_model.keys():
                myparam_model[key] = param_model[key]
        for layer in config.purne_layer:
            num_experts = config.require_experts[config.purne_layer.index(layer)]
            if layer > 11:
                index = 0
                for i in num_experts:
                    myparam_model['decoder.block.'+str(layer-12)+'.layer.2.mlp.experts.expert_'+str(index)+'.wi.weight'] = \
                    param_model['decoder.block.'+str(layer-12)+'.layer.2.mlp.experts.expert_'+str(i)+'.wi.weight']
                    myparam_model['decoder.block.'+str(layer-12)+'.layer.2.mlp.experts.expert_'+str(index)+'.wo.weight'] = \
                    param_model['decoder.block.'+str(layer-12)+'.layer.2.mlp.experts.expert_'+str(i)+'.wo.weight']
                    index += 1
                myparam_model['decoder.block.'+str(layer-12)+'.layer.2.mlp.router.classifier.weight'] = \
                    param_model['decoder.block.'+str(layer-12)+'.layer.2.mlp.router.classifier.weight'][num_experts,:]
            else:
                index = 0
                for i in num_experts:
                    myparam_model['encoder.block.'+str(layer)+'.layer.1.mlp.experts.expert_'+str(index)+'.wi.weight'] = \
                    param_model['encoder.block.'+str(layer)+'.layer.1.mlp.experts.expert_'+str(i)+'.wi.weight']
                    myparam_model['encoder.block.'+str(layer)+'.layer.1.mlp.experts.expert_'+str(index)+'.wo.weight'] = \
                    param_model['encoder.block.'+str(layer)+'.layer.1.mlp.experts.expert_'+str(i)+'.wo.weight']
                    index += 1
                myparam_model['encoder.block.'+str(layer)+'.layer.1.mlp.router.classifier.weight'] = \
                    param_model['encoder.block.'+str(layer)+'.layer.1.mlp.router.classifier.weight'][num_experts,:]
        mymodel.load_state_dict(myparam_model)
    return mymodel
def train_switch_base_8(some_args):
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
    model = purning_model(some_args) # AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
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
    name=config1[some_args]['model'],
    # track hyperparameters and run metadata
    
    config={
    "learning_rate": 5e-05,
    "architecture": "switch-8",
    "dataset": "samsum",
    "epochs": 10,
    }
    )
    
    batch_size=4
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
    num_epochs = 8
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
    model_name=config1[some_args]['model']
    
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
            wandb.log({'batch_loss': loss_all/step})
            # break
        # dict_router = {}
        # index = 0
        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model.generate(batch['input_ids'])# (**batch)
                # outputs = model(**batch)
            # logits = outputs.logits
            # predictions = torch.argmax(logits, dim=-1)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            # dict_router[index]=(outputs.encoder_router_logits,outputs.decoder_router_logits)
            # index += 1
            # with open("./experiment/router_dict_finetune_o.pkl", "wb") as file:
            #     # print(score_dict)
            #     pickle.dump(dict_router, file)
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

train_switch_base_8('Avg')


def eval_switch_base_8(option=None):
    # def save_model(model,name):
    #     model_to_save = model.module if hasattr(model, 'module') else model
    #     model_checkpoint = os.path.join('/home/ubuntu/SwitchTransformer/pth/', "%s_checkpoint.bin" % name)
    #     torch.save(model_to_save.state_dict(), model_checkpoint)
    #     print("Saved model checkpoint to [DIR: /home/ubuntu/SwitchTransformer/pth/]")
    #     # logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)
    # # nltk.download('punkt')
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
    model = purning_model(option) # AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")

    
    # model_checkpoint = os.path.join('/home/ubuntu/SwitchTransformer/pth/' + config1.option['model'])
    model.load_state_dict(torch.load('/home/ubuntu/SwitchTransformer/pth/' + config1[option]['model']))
    print("Load model checkpoint from [DIR: /home/ubuntu/SwitchTransformer/pth/]")
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
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

    batch_size=16
    # dataset = load_dataset("yelp_review_full")
    
    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    
    tokenized_datasets.set_format("torch")
    # tokenized_datasets = tokenized_datasets.remove_columns(["valid"])
    tokenized_datasets = tokenized_datasets.remove_columns(["dialogue"])
    tokenized_datasets = tokenized_datasets.remove_columns(["id"])
    tokenized_datasets = tokenized_datasets.remove_columns(["summary"])
    
    # train_dataset = tokenized_datasets["train"].shuffle(seed=42) # .select(range(1000))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42) # .select(range(1000))

    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )
    # train_dataloader = DataLoader(
    #     train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    # )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)


    # lr_scheduler = WarmupLinearSchedule(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # print(torch.cuda.memory_summary(device='cuda'))
    model = model.to(device)
    best_acc = 0
    model_name='switch_samsum_8_layer_9_expert_2'
    # print(torch.cuda.memory_summary(device='cuda'))
    model.eval()
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())/1048576
    # param_size =
    start_time = time.perf_counter()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model.generate(batch['input_ids'])# (**batch)
            # outputs = model(**batch)
        # logits = outputs.logits
        # predictions = torch.argmax(logits, dim=-1)
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        # dict_router[index]=(outputs.encoder_router_logits,outputs.decoder_router_logits)
        # index += 1
        # with open("./experiment/router_dict_finetune_o.pkl", "wb") as file:
        #     # print(score_dict)
        #     pickle.dump(dict_router, file)
    result = metric.compute()
    end_time = time.perf_counter()
    real_time = end_time - start_time
    avg_time = real_time/len(eval_dataloader)
    # wandb.log({'para_size':param_size,'avg_time':avg_time, 'inference_time:': real_time,'rouge1': result['rouge1']})

    # wandb.finish()
    del model
    del dataset
    del tokenizer
    print(param_size, (avg_time, real_time), result['rouge1'])
    return param_size, (avg_time, real_time), result['rouge1']


def freq():
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

    option=None
    # tokenizer = AutoTokenizer.from_pretrained("emre/switch-base-8-finetuned-samsum")
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    model = purning_model(option) # AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")

    
    # model_checkpoint = os.path.join('/home/ubuntu/SwitchTransformer/pth/' + config1.option['model'])
    # model.load_state_dict(torch.load('/home/ubuntu/SwitchTransformer/pth/' + config1[option]['model']))
    # print("Load model checkpoint from [DIR: /home/ubuntu/SwitchTransformer/pth/]")
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
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
    
    batch_size=4
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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.eval()
    model = model.to(device)


    # dict_1 = {}
    index= 0 
    expert_list = []
    # for batch in train_dataloader:
    #     # break
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     outputs = model(**batch)
    #     # dict_1[index] = (outputs.encoder_router_logits, outputs.decoder_router_logits)
    #     # break
    #     encoder_router = outputs.encoder_router_logits
    #     decoder_router =  outputs.decoder_router_logits
    #     if index == 0:
    #         for i in encoder_router:
    #             if len(i) == 2:
    #                 # print(i[1].shape)
    #                 expert_list.append([i[1]])
    #         for i in decoder_router:
    #             if len(i) == 2:
    #                 # print(i[1].shape)
    #                 expert_list.append([i[1]])  
    #     else:
    #         k = 0
    #         for i in encoder_router:
    #             if len(i) == 2:
    #                 # print(i[1].shape)
    #                 expert_list[k].append(i[1])
    #                 k+=1
    #         for i in decoder_router:
    #             if len(i) == 2:
    #                 # print(i[1].shape)
    #                 expert_list[k].append(i[1])
    #                 k+=1
    #     index += 1
    # file_name = "./result/freq_o_train.pkl"
    # with open(file_name, 'wb') as file:
    #     pickle.dump(expert_list, file)

    # list_1 = []
    # expert_list
    for batch in eval_dataloader:
        # break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # dict_1[index] = (outputs.encoder_router_logits, outputs.decoder_router_logits)
        # break
        encoder_router = outputs.encoder_router_logits
        decoder_router =  outputs.decoder_router_logits
        if index == 0:
            for i in encoder_router:
                if len(i) == 2:
                    # print(i[1].shape)
                    expert_list.append([i[1]])
            for i in decoder_router:
                if len(i) == 2:
                    # print(i[1].shape)
                    expert_list.append([i[1]])  
        else:
            k = 0
            for i in encoder_router:
                if len(i) == 2:
                    # print(i[1].shape)
                    expert_list[k].append(i[1])
                    k+=1
            for i in decoder_router:
                if len(i) == 2:
                    # print(i[1].shape)
                    expert_list[k].append(i[1])
                    k+=1
        index += 1
    file_name = "./result/freq_o_eval.pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(expert_list, file)

    model.load_state_dict(torch.load('/home/ubuntu/SwitchTransformer/pth/switch_samsum_8_checkpoint.bin'))
    print("Load model checkpoint from [DIR: /home/ubuntu/SwitchTransformer/pth/]")


    expert_list
    for batch in train_dataloader:
        # break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # dict_1[index] = (outputs.encoder_router_logits, outputs.decoder_router_logits)
        # break
        encoder_router = outputs.encoder_router_logits
        decoder_router =  outputs.decoder_router_logits
        if index == 0:
            for i in encoder_router:
                if len(i) == 2:
                    # print(i[1].shape)
                    expert_list.append([i[1]])
            for i in decoder_router:
                if len(i) == 2:
                    # print(i[1].shape)
                    expert_list.append([i[1]])  
        else:
            k = 0
            for i in encoder_router:
                if len(i) == 2:
                    # print(i[1].shape)
                    expert_list[k].append(i[1])
                    k+=1
            for i in decoder_router:
                if len(i) == 2:
                    # print(i[1].shape)
                    expert_list[k].append(i[1])
                    k+=1
        index += 1
    file_name = "./result/freq_f_train.pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(expert_list, file)


    expert_list
    for batch in eval_dataloader:
        # break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # dict_1[index] = (outputs.encoder_router_logits, outputs.decoder_router_logits)
        # break
        encoder_router = outputs.encoder_router_logits
        decoder_router =  outputs.decoder_router_logits
        if index == 0:
            for i in encoder_router:
                if len(i) == 2:
                    # print(i[1].shape)
                    expert_list.append([i[1]])
            for i in decoder_router:
                if len(i) == 2:
                    # print(i[1].shape)
                    expert_list.append([i[1]])  
        else:
            k = 0
            for i in encoder_router:
                if len(i) == 2:
                    # print(i[1].shape)
                    expert_list[k].append(i[1])
                    k+=1
            for i in decoder_router:
                if len(i) == 2:
                    # print(i[1].shape)
                    expert_list[k].append(i[1])
                    k+=1
        index += 1
    file_name = "./result/freq_f_eval.pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(expert_list, file)

    del model
    del dataset
    del tokenizer
# freq()
def shell_eval():
    import pickle
    ok_dict = {}
    file_name = "./result/1026.pkl"
    for key in config1.keys():
        memory, realt, rg = eval_switch_base_8(key)
        ok_dict[key] = (memory,realt,rg)

        with open(file_name, 'wb') as file:
            pickle.dump(ok_dict, file)
# shell_eval()