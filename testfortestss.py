from datasets import load_dataset
import fig
import time
from torch.optim.lr_scheduler import LambdaLR

import torch
import numpy as np
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,SwitchTransformersForConditionalGeneration,BertForQuestionAnswering
import pickle
import wandb

import numpy as np
# import evaluate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler,DataCollatorForSeq2Seq
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
from transformers import DefaultDataCollator
# import datasets
import nltk
import evaluate
import collections
from transformers import AutoConfig
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
def save_model(model,name):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(dir_path + '/pth/', "%s_checkpoint.bin" % name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    print("Saved model checkpoint to [DIR: /home/ubuntu/SwitchTransformer/pth/]")

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

def Create_MoE_Model(model_name, num_experts):
    if model_name == 'bert':
        config = AutoConfig.from_pretrained("bert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        config.moe=True
        config.num_experts=num_experts
        mymoe = BertForQuestionAnswering(config=config)
        return mymoe, tokenizer
        modelForLoad = BertForQuestionAnswering(config=config)
        if num_experts == 0:
            return modelForLoad,tokenizer

        # print(modelForLoad.state_dict().keys(),mymoe.state_dict().keys())
        mymoeParam = mymoe.state_dict()
        bertParam = modelForLoad.state_dict()
        # original weight = ['bert.encoder.layer.10.intermediate.dense.weight', 'bert.encoder.layer.10.intermediate.dense.bias', 
        # 'bert.encoder.layer.10.output.dense.weight', 'bert.encoder.layer.10.output.dense.bias']
        # desity weight = ['bert.encoder.layer.0.moe_linear.experts.0.htoh4.weight', 'bert.encoder.layer.0.moe_linear.experts.0.htoh4.bias', 
        # 'bert.encoder.layer.0.moe_linear.experts.0.h4toh.weight', 'bert.encoder.layer.0.moe_linear.experts.0.h4toh.bias',]
        # original_layer_normal = ['bert.encoder.layer.11.output.LayerNorm.weight', 'bert.encoder.layer.11.output.LayerNorm.bias']
        # desitny weight = ['bert.encoder.layer.0.moe_linear.layer_norm.weight', 'bert.encoder.layer.0.moe_linear.layer_norm.bias']
        bertLayerLength=12
        # copy linear weight, bias and layernormal
        for layer in range(bertLayerLength):
            for expert_id in range(num_experts):
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.intermediate.dense.weight'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.intermediate.dense.bias'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.dense.weight'].unsqueeze(0).detach().clone()
                mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.dense.bias'].unsqueeze(0).detach().clone()
            mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.layer_norm.weight'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.LayerNorm.weight'].detach().clone()
            mymoeParam['bert.encoder.layer.'+str(layer)+'.moe_linear.layer_norm.bias'] = bertParam['bert.encoder.layer.'+str(layer)+'.output.LayerNorm.bias'].detach().clone() 
        mymoe.load_state_dict(mymoeParam)
        return mymoe, tokenizer
    elif model_name == 'xl':
        from transformers import TransfoXLForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")
        config = AutoConfig.from_pretrained("transfo-xl-wt103")
        config.moe=True
        config.num_experts=num_experts
        mymoe = TransfoXLForSequenceClassification(config=config)
        return mymoe, tokenizer
        # config.num_labels = 2
        modelForLoad = TransfoXLForSequenceClassification(config=config)
        if num_experts == 0:
            return modelForLoad,tokenizer
        # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        # outputs = modelForLoad(**inputs)

        # print(modelForLoad.state_dict().keys(),mymoe.state_dict().keys())
        mymoeParam = mymoe.state_dict()
        bertParam = modelForLoad.state_dict()
        # original weight = ['transformer.layers.0.pos_ff.CoreNet.0.weight', 'transformer.layers.0.pos_ff.CoreNet.0.bias', 
        # 'transformer.layers.0.pos_ff.CoreNet.3.weight', 
        # 'transformer.layers.0.pos_ff.CoreNet.3.bias']
        # desity weight = ['transformer.h.11.moe_linear.experts.15.htoh4.weight', 'transformer.h.11.moe_linear.experts.15.htoh4.bias', 
        # 'transformer.h.11.moe_linear.experts.15.h4toh.weight', 'transformer.h.11.moe_linear.experts.15.h4toh.bias',]
        # original_layer_normal = ['transformer.layers.0.pos_ff.layer_norm.weight', 'transformer.layers.0.pos_ff.layer_norm.bias']
        # desitny weight = ['transformer.h.11.moe_linear.layer_norm.weight', 'transformer.h.11.moe_linear.layer_norm.bias',]
        bertLayerLength=18
        # copy linear weight, bias and layernormal
        for layer in range(bertLayerLength):
            for expert_id in range(num_experts):
                mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.weight'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.CoreNet.0.weight'].unsqueeze(0).detach().clone()
                mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.bias'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.CoreNet.0.bias'].unsqueeze(0).detach().clone()
                mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.weight'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.CoreNet.3.weight'].unsqueeze(0).detach().clone()
                mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.bias'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.CoreNet.3.bias'].unsqueeze(0).detach().clone()
            mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.layer_norm.weight'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.layer_norm.weight'].detach().clone()
            mymoeParam['transformer.layers.'+str(layer)+'.moe_linear.layer_norm.bias'] = bertParam['transformer.layers.'+str(layer)+'.pos_ff.layer_norm.bias'].detach().clone() 
        mymoe.load_state_dict(mymoeParam)
        return mymoe, tokenizer
    elif model_name == 'gpt':
        from transformers import GPT2LMHeadModel, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        config = AutoConfig.from_pretrained("gpt2")
        config.moe=True
        config.num_experts=num_experts
        mymoe = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
        return mymoe, tokenizer
    
        modelForLoad = GPT2LMHeadModel.from_pretrained("gpt2",config=config)
        if num_experts == 0:
            return modelForLoad,tokenizer
        # tokenizer = AutoTokenizer.from_pretrained("cwh/gpt2-medium-finetuned-wikitext2")
        # model = AutoModelForCausalLM.from_pretrained("cwh/gpt2-medium-finetuned-wikitext2")

        # print(modelForLoad.state_dict().keys(),mymoe.state_dict().keys())
        mymoeParam = mymoe.state_dict()
        bertParam = modelForLoad.state_dict()
        # original weight = ['transformer.h.0.mlp.c_fc.weight', 'transformer.h.0.mlp.c_fc.bias',
        #  'transformer.h.0.mlp.c_proj.weight', 'transformer.h.0.mlp.c_proj.bias']
        # desity weight = ['transformer.h.11.moe_linear.experts.15.htoh4.weight', 'transformer.h.11.moe_linear.experts.15.htoh4.bias', 
        # 'transformer.h.11.moe_linear.experts.15.h4toh.weight', 'transformer.h.11.moe_linear.experts.15.h4toh.bias',]
        # original_layer_normal = ['transformer.h.0.ln_2.weight', 'transformer.h.0.ln_2.bias']
        # desitny weight = ['transformer.h.11.moe_linear.layer_norm.weight', 'transformer.h.11.moe_linear.layer_norm.bias',]
        bertLayerLength=12
        # copy linear weight, bias and layernormal
        for layer in range(bertLayerLength):
            for expert_id in range(num_experts):
                mymoeParam['transformer.h.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.weight'] = bertParam['transformer.h.'+str(layer)+'.mlp.c_fc.weight'].T.unsqueeze(0).detach().clone()
                mymoeParam['transformer.h.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.htoh4.bias'] = bertParam['transformer.h.'+str(layer)+'.mlp.c_fc.bias'].unsqueeze(0).detach().clone()
                mymoeParam['transformer.h.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.weight'] = bertParam['transformer.h.'+str(layer)+'.mlp.c_proj.weight'].T.unsqueeze(0).detach().clone()
                mymoeParam['transformer.h.'+str(layer)+'.moe_linear.experts.'+str(expert_id)+'.h4toh.bias'] = bertParam['transformer.h.'+str(layer)+'.mlp.c_proj.bias'].unsqueeze(0).detach().clone()
            mymoeParam['transformer.h.'+str(layer)+'.moe_linear.layer_norm.weight'] = bertParam['transformer.h.'+str(layer)+'.ln_2.weight'].detach().clone()
            mymoeParam['transformer.h.'+str(layer)+'.moe_linear.layer_norm.bias'] = bertParam['transformer.h.'+str(layer)+'.ln_2.bias'].detach().clone() 
        mymoe.load_state_dict(mymoeParam)
        return mymoe, tokenizer
    else:
        print('no model ' + model_name)
    print('success to load ' + model_name)
# Create_MoE_Model('gpt',8)
def train_GPT_MoE():
    from transformers import DataCollatorWithPadding
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    
    dataset = load_dataset("samsum")
    metric = evaluate.load("rouge")

    model,tokenizer = Create_MoE_Model('gpt',2) # AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")


    def preprocess_function(examples):
        # inputs = [doc for doc in examples['dialogue']]
        model_inputs = tokenizer(examples['dialogue'], padding="max_length", max_length=1024, truncation=True)

        # Setup the tokenizer for targets
        labels = tokenizer(text_target=examples["summary"], padding="max_length", truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    
    batch_size=1
    # dataset = load_dataset("yelp_review_full")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(tokenizer))

    tokenized_datasets = dataset.map(preprocess_function,  batched=True)
    tokenized_datasets.set_format("torch")
    # tokenized_datasets = tokenized_datasets.remove_columns(["valid"])
    tokenized_datasets = tokenized_datasets.remove_columns(["dialogue"])
    tokenized_datasets = tokenized_datasets.remove_columns(["id"])
    tokenized_datasets = tokenized_datasets.remove_columns(["summary"])
    
    train_dataset = tokenized_datasets["train"].shuffle(seed=42) # .select(range(1000))
    eval_dataset = tokenized_datasets["test"]# .shuffle(seed=42) # .select(range(1000))



    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=None,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=5e-05,
                                betas=(0.9,0.999),
                                eps=1e-08)
    num_epochs = 1
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
    model_name='gpt'

    wandb.init(    # set the wandb project where this run will be logged
    project="switch-8-samsum",
    name='gpt',
    # track hyperparameters and run metadata
    
    config={
    "learning_rate": 5e-05,
    "architecture": "gpt",
    "dataset": "samsum",
    "epochs": num_epochs,
    }
    )

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
            break
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
            break
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
def train_Bert_MoE():
    def compute_metrics(start_logits, end_logits, features, examples):

        n_best = 20
        max_answer_length = 30

        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)

    model,tokenizer = Create_MoE_Model('bert',1)

    max_length = 384
    stride = 128


    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs
    
    datasets = load_dataset("squad")
    # raw_datasets  = raw_datasets.train_test_split(test_size=0.2)
    # raw_datasets  = raw_datasets.rename_column("test", "validation")
    metric = evaluate.load("squad")
    # tokenized_squad = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    train_dataset = datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )
    eval_dataset = datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=datasets["validation"].column_names,
    )
    validation_dataset = eval_dataset.remove_columns(["example_id", "offset_mapping"])

    data_collator = DefaultDataCollator()

    batch_size=4
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(validation_dataset, collate_fn=data_collator, batch_size=batch_size)
    num_epochs = 1
    model_name="bert" # config1[some_args]['model']
    # metric = evaluate.load("squad_v2" if data_args.version_2_with_negative else "squad")
    wandb.init(    # set the wandb project where this run will be logged
    project="switch-8-samsum",
    name='bert', # config1[some_args]['model'],
    # track hyperparameters and run metadata
    
    config={
    "learning_rate": 3e-5,
    "architecture": model_name,
    "dataset": "samsum",
    "epochs": num_epochs,
    }
    )

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=3e-5)
    # ,
                                # betas=(0.9,0.999),
                                # eps=1e-08)
    # num_epochs = 8
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    progress_bar = tqdm(range(num_training_steps))
    model = model.to(device)
    best_acc = 0
    
    
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
        # question_answerer = pipeline("question-answering", model=model)
        start_logits = []
        end_logits = []
        # accelerator.print("Evaluation!")
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(validation_dataset)]
        end_logits = end_logits[: len(validation_dataset)]
        # metrics = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
        metrics = compute_metrics(start_logits, end_logits, eval_dataset, datasets["validation"])
        # {'exact_match': 83.0, 'f1': 88.25}
        wandb.log({'loss': loss_all/step, 'exact_match':metrics['exact_match'],'f1':metrics['f1']}) # 'rouge1': result['rouge1']})
        if best_acc < metrics['f1']:
            save_model(model,model_name)
            best_acc = metrics['exact_match']
    

    wandb.finish()
    del model
    del datasets
    del tokenizer
def train_xl_MoE():
    from transformers import DataCollatorWithPadding
    dataset = load_dataset("glue", "cola")
    # dataset = load_dataset("imdb",split="train[10:20]")
    # dataset  = dataset.train_test_split(test_size=0.2)
    
    model,tokenizer = Create_MoE_Model('xl',2)
    tokenizer.model_max_length = 250
    print(model)
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], padding="max_length",truncation=True)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(len(tokenizer))
    tokenized_datasets = dataset.map(preprocess_function,  batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metric = evaluate.load("accuracy")
    model_name = 'xl'
    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     predictions = np.argmax(predictions, axis=1)
    #     return metric.compute(predictions=predictions, references=labels)
    # id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    # label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    tokenized_datasets.set_format("torch")
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence"])
    tokenized_datasets = tokenized_datasets.remove_columns(["idx"])
    train_batch_size = 1
    eval_batch_size = 1
    train_dataloader = DataLoader(tokenized_datasets["train"].shuffle(seed=42), collate_fn=data_collator,shuffle=True, batch_size=train_batch_size)
    eval_dataloader = DataLoader(tokenized_datasets["validation"], collate_fn=data_collator, batch_size=eval_batch_size)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=5e-5,
                                betas=(0.9,0.999),
                                eps=1e-08)
    num_epochs = 1
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    progress_bar = tqdm(range(num_training_steps))
    model = model.to(device)
    best_acc = 0
    

    wandb.init(    # set the wandb project where this run will be logged
    project="switch-8-samsum",
    name='xl',
    # track hyperparameters and run metadata
    
    config={
    "learning_rate": 5e-05,
    "architecture": "xl",

    "dataset": "samsum",
    "epochs": 8,
    }
    )

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
            break

        model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            break
        metrics = metric.compute()
        wandb.log({'loss': loss_all/step, 'acc':metrics['accuracy']}) # 'rouge1': result['rouge1']})
        if best_acc < metrics['accuracy']:
            save_model(model,model_name)
            best_acc = metrics['accuracy']
    

    wandb.finish()
    del model
    del dataset
    del tokenizer
# train_GPT_MoE()
# train_Bert_MoE()
# train_xl_MoE()

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
# train_switch_base_8('Avg')
def train_bert_base_8():
    from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
    from huggingface_hub import notebook_login
    from transformers import pipeline
    # notebook_login()
    def save_model(model,name):
        model_to_save = model.module if hasattr(model, 'module') else model
        model_checkpoint = os.path.join('/home/ubuntu/SwitchTransformer/pth/', "%s_checkpoint.bin" % name)
        torch.save(model_to_save.state_dict(), model_checkpoint)
        print("Saved model checkpoint to [DIR: /home/ubuntu/SwitchTransformer/pth/]")

    def compute_metrics(start_logits, end_logits, features, examples):

        n_best = 20
        max_answer_length = 30

        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)
    # squad = load_dataset("squad", split="train[:10]")
    # squad = squad.train_test_split(test_size=0.2)
    # metric = evaluate.load("rouge")
    # squad.set_format("torch")
    # tokenizer = AutoTokenizer.from_pretrained("emre/switch-base-8-finetuned-samsum"
    
    # tokenized_datasets = tokenized_datasets.remove_columns(["valid"])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = purning_model(some_args) # AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
    # model = AutoModelForSeq2SeqLM.from_pretrained("emre/switch-base-8-finetuned-samsum")

    # max_input_length = 1024
    # max_target_length = 128

    max_length = 384
    stride = 128


    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs
    # def preprocess_function(examples):
    #     questions = [q.strip() for q in examples["question"]]
    #     inputs = tokenizer(
    #         questions,
    #         examples["context"],
    #         max_length=384,
    #         truncation="only_second",
    #         return_overflowing_tokens=True,
    #         return_offsets_mapping=True,
    #         padding="max_length",
    #     )

    #     offset_mapping = inputs["offset_mapping"]
        
    #     answers = examples["answers"]
    #     start_positions = []
    #     end_positions = []

    #     # add
    #     # inputsCopy = tokenizer(
    #     #     questions,
    #     #     examples["context"],
    #     #     max_length=384,
    #     #     truncation="only_second",
    #     #     stride=128,
    #     #     return_overflowing_tokens=True,
    #     #     return_offsets_mapping=True,
    #     #     padding="max_length",
    #     # )
    #     example_ids = []
    #     sample_map = inputs["overflow_to_sample_mapping"]
    #     # sample_map = inputsCopy.pop("overflow_to_sample_mapping")

    #     for i, offset in enumerate(offset_mapping):
            
    #         # add
    #         sample_idx = sample_map[i]
    #         example_ids.append(examples["id"][sample_idx])

    #         sequence_ids = inputs.sequence_ids(i)
    #         offset = inputs["offset_mapping"][i]
    #         inputs["offset_mapping"][i] = [
    #             o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
    #         ]


            
    #         answer = answers[i]
    #         start_char = answer["answer_start"][0]
    #         end_char = answer["answer_start"][0] + len(answer["text"][0])
    #         sequence_ids = inputs.sequence_ids(i)

    #         # Find the start and end of the context
    #         idx = 0
    #         while sequence_ids[idx] != 1:
    #             idx += 1
    #         context_start = idx
    #         while sequence_ids[idx] == 1:
    #             idx += 1
    #         context_end = idx - 1

    #         # If the answer is not fully inside the context, label it (0, 0)
    #         if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
    #             start_positions.append(0)
    #             end_positions.append(0)
    #         else:
    #             # Otherwise it's the start and end token positions
    #             idx = context_start
    #             while idx <= context_end and offset[idx][0] <= start_char:
    #                 idx += 1
    #             start_positions.append(idx - 1)

    #             idx = context_end
    #             while idx >= context_start and offset[idx][1] >= end_char:
    #                 idx -= 1
    #             end_positions.append(idx + 1)

    #     inputs["start_positions"] = start_positions
    #     inputs["end_positions"] = end_positions
    #     inputs["example_id"] = example_ids

        

    #     return inputs
    
    raw_datasets = load_dataset("squad")
    # raw_datasets  = raw_datasets.train_test_split(test_size=0.2)
    # raw_datasets  = raw_datasets.rename_column("test", "validation")
    metric = evaluate.load("squad")
    # tokenized_squad = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    train_dataset = raw_datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    eval_dataset = raw_datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )
    validation_dataset = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    # eval_dataset = raw_datasets["test"].map(
    #     preprocess_validation_examples,
    #     batched=True,
    #     remove_columns=raw_datasets["test"].column_names,
    # )
    # validation_dataset = eval_dataset.remove_columns(["example_id", "offset_mapping"])

    # tokenized_squad.set_format("torch")
    # tokenized_squadCopy = tokenized_squad.remove_columns(["example_id"])
    # tokenized_squadCopy = tokenized_squadCopy.remove_columns(["overflow_to_sample_mapping"])
    # tokenized_squadCopy = tokenized_squadCopy.remove_columns(["offset_mapping"])
    data_collator = DefaultDataCollator()
    # train_dataset = tokenized_squadCopy["train"].shuffle(seed=42) # .select(range(1000))
    # eval_dataset = tokenized_squadCopy["test"] # .select(range(1000))
    
    # label_pad_token_id = tokenizer.pad_token_id
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     model=model,
    #     label_pad_token_id=label_pad_token_id,
    #     pad_to_multiple_of=None,
    # )
    batch_size=32
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(validation_dataset, collate_fn=data_collator, batch_size=batch_size)

    # metric = evaluate.load("squad_v2" if data_args.version_2_with_negative else "squad")
    wandb.init(    # set the wandb project where this run will be logged
    project="switch-8-samsum",
    name='bert-test', # config1[some_args]['model'],
    # track hyperparameters and run metadata
    
    config={
    "learning_rate": 3e-5,
    "architecture": "switch-8",
    "dataset": "samsum",
    "epochs": 10,
    }
    )
    
    # batch_size=4
    # dataset = load_dataset("yelp_review_full")
    
    # tokenized_datasets = dataset.map(preprocess_function, batched=True)

    
    # tokenized_datasets.set_format("torch")
    # tokenized_datasets = tokenized_datasets.remove_columns(["valid"])
    # tokenized_datasets = tokenized_datasets.remove_columns(["dialogue"])
    # tokenized_datasets = tokenized_datasets.remove_columns(["id"])
    # tokenized_datasets = tokenized_datasets.remove_columns(["summary"])
    
    # train_dataset = tokenized_datasets["train"].shuffle(seed=42) # .select(range(1000))
    # eval_dataset = tokenized_datasets["test"].shuffle(seed=42) # .select(range(1000))

    # label_pad_token_id = tokenizer.pad_token_id
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     model=model,
    #     label_pad_token_id=label_pad_token_id,
    #     pad_to_multiple_of=None,
    # )
    # train_dataloader = DataLoader(
    #     train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    # )
    # eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=3e-5)
    # ,
                                # betas=(0.9,0.999),
                                # eps=1e-08)
    num_epochs = 8
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    progress_bar = tqdm(range(num_training_steps))
    model = model.to(device)
    best_acc = 0
    model_name="bert" # config1[some_args]['model']
    
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
        # question_answerer = pipeline("question-answering", model=model)
        start_logits = []
        end_logits = []
        # accelerator.print("Evaluation!")
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(validation_dataset)]
        end_logits = end_logits[: len(validation_dataset)]
        # metrics = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
        metrics = compute_metrics(start_logits, end_logits, eval_dataset, raw_datasets["validation"])
        # {'exact_match': 83.0, 'f1': 88.25}
        wandb.log({'loss': loss_all/step, 'exact_match':metrics['exact_match'],'f1':metrics['f1']}) # 'rouge1': result['rouge1']})
        if best_acc < metrics['f1']:
            save_model(model,model_name)
            best_acc = metrics['exact_match']
    

    wandb.finish()
    del model
    del raw_datasets
    del tokenizer
# train_bert_base_8()
def train_bert_moe_8():
    from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
    from huggingface_hub import notebook_login
    from transformers import pipeline
    # notebook_login()
    def save_model(model,name):
        model_to_save = model.module if hasattr(model, 'module') else model
        model_checkpoint = os.path.join('/home/ubuntu/SwitchTransformer/pth/', "%s_checkpoint.bin" % name)
        torch.save(model_to_save.state_dict(), model_checkpoint)
        print("Saved model checkpoint to [DIR: /home/ubuntu/SwitchTransformer/pth/]")

    def compute_metrics(start_logits, end_logits, features, examples):

        n_best = 20
        max_answer_length = 30

        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
                end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": ""})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return metric.compute(predictions=predicted_answers, references=theoretical_answers)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = purning_model(some_args) # AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
    # model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
    # model = AutoModelForSeq2SeqLM.from_pretrained("emre/switch-base-8-finetuned-samsum")
    # config = AutoConfig.from_pretrained("bert-base-uncased")
    # # config.moe=True
    # mymodel = BertForQuestionAnswering(config=config)
    # config.moe=True
    # config.num_experts=8
    # mymoe = BertForQuestionAnswering(config=config)
    # max_input_length = 1024
    # max_target_length = 128
    # print(mymoe,mymodel)
    max_length = 384
    stride = 128


    def preprocess_training_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs
    
    raw_datasets = load_dataset("squad")
    # raw_datasets  = raw_datasets.train_test_split(test_size=0.2)
    # raw_datasets  = raw_datasets.rename_column("test", "validation")
    metric = evaluate.load("squad")
    # tokenized_squad = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    train_dataset = raw_datasets["train"].map(
        preprocess_training_examples,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    eval_dataset = raw_datasets["validation"].map(
        preprocess_validation_examples,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )
    validation_dataset = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    # eval_dataset = raw_datasets["test"].map(
    #     preprocess_validation_examples,
    #     batched=True,
    #     remove_columns=raw_datasets["test"].column_names,
    # )
    # validation_dataset = eval_dataset.remove_columns(["example_id", "offset_mapping"])

    # tokenized_squad.set_format("torch")
    # tokenized_squadCopy = tokenized_squad.remove_columns(["example_id"])
    # tokenized_squadCopy = tokenized_squadCopy.remove_columns(["overflow_to_sample_mapping"])
    # tokenized_squadCopy = tokenized_squadCopy.remove_columns(["offset_mapping"])
    data_collator = DefaultDataCollator()
    # train_dataset = tokenized_squadCopy["train"].shuffle(seed=42) # .select(range(1000))
    # eval_dataset = tokenized_squadCopy["test"] # .select(range(1000))
    
    # label_pad_token_id = tokenizer.pad_token_id
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     model=model,
    #     label_pad_token_id=label_pad_token_id,
    #     pad_to_multiple_of=None,
    # )
    batch_size=32
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size
    )
    eval_dataloader = DataLoader(validation_dataset, collate_fn=data_collator, batch_size=batch_size)

    # metric = evaluate.load("squad_v2" if data_args.version_2_with_negative else "squad")
    # wandb.init(    # set the wandb project where this run will be logged
    # project="switch-8-samsum",
    # name='bert-test', # config1[some_args]['model'],
    # # track hyperparameters and run metadata
    
    # config={
    # "learning_rate": 3e-5,
    # "architecture": "switch-8",
    # "dataset": "samsum",
    # "epochs": 10,
    # }
    # )
    
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=3e-5)
    # ,
                                # betas=(0.9,0.999),
                                # eps=1e-08)
    num_epochs = 8
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    progress_bar = tqdm(range(num_training_steps))
    model = model.to(device)
    best_acc = 0
    model_name="bert" # config1[some_args]['model']
    
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
            # wandb.log({'batch_loss': loss_all/step})
            break
        # dict_router = {}
        # index = 0
        model.eval()
        # question_answerer = pipeline("question-answering", model=model)
        start_logits = []
        end_logits = []
        # accelerator.print("Evaluation!")
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(validation_dataset)]
        end_logits = end_logits[: len(validation_dataset)]
        # metrics = compute_metrics(start_logits, end_logits, validation_dataset, raw_datasets["validation"])
        metrics = compute_metrics(start_logits, end_logits, eval_dataset, raw_datasets["validation"])
        # {'exact_match': 83.0, 'f1': 88.25}
        # wandb.log({'loss': loss_all/step, 'exact_match':metrics['exact_match'],'f1':metrics['f1']}) # 'rouge1': result['rouge1']})
        if best_acc < metrics['f1']:
            save_model(model,model_name)
            best_acc = metrics['exact_match']
    

    # wandb.finish()
    del model
    del raw_datasets
    del tokenizer
# train_bert_moe_8()
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
    del datasets
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
def shell_eval():
    import pickle
    ok_dict = {}
    file_name = "./result/1026.pkl"
    for key in config1.keys():
        memory, realt, rg = eval_switch_base_8(key)
        ok_dict[key] = (memory,realt,rg)

        with open(file_name, 'wb') as file:
            pickle.dump(ok_dict, file)