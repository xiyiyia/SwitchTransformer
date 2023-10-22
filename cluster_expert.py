from datasets import load_dataset
import fig


import torch
import numpy as np
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def reload_model(method,model,start_layer):
    import random
    model_state_dict = model.state_dict()
    if method == 'naive_8_to_4':
        all_numbers = list(range(0, 8))
        unique_combinations, number_list = [], []
        # 生成4个不重复的组合
        while len(unique_combinations) < 4:
            # 从所有数字中随机选择两个数字
            selected_numbers = random.sample(all_numbers, 2)
            # 对选中的数字排序，以确保组合的唯一性
            bool_1 = False
            for number in selected_numbers:
                if number in number_list:
                    bool_1 = True
                    break
                # number_list.append(number)
            if bool_1 == False:
                unique_combinations.append(selected_numbers)
                for number in selected_numbers:
                    number_list.append(number)
            else:
                continue
        print(unique_combinations)
        if start_layer < 6:
            for block in range(start_layer*2,12,2):
                for (expert_id_1,expert_id_2) in unique_combinations:
                    combine_key_1_wi = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_1)+'.wi.weight'
                    combine_key_1_wo = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_1)+'.wo.weight'
                    combine_key_2_wi = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_2)+'.wi.weight'
                    combine_key_2_wo = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_2)+'.wo.weight'
                    model_state_dict[combine_key_1_wi] = ((model_state_dict[combine_key_1_wi] + model_state_dict[combine_key_2_wi])/2.0).clone().detach()
                    model_state_dict[combine_key_1_wo] = ((model_state_dict[combine_key_1_wo] + model_state_dict[combine_key_2_wo])/2.0).clone().detach()
                    model_state_dict[combine_key_2_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_2_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
        if start_layer >= 6:
            for block in range((start_layer-6)*2,12,2):
                for (expert_id_1,expert_id_2) in unique_combinations:
                    combine_key_1_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_1)+'.wi.weight'
                    combine_key_1_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_1)+'.wo.weight'
                    combine_key_2_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_2)+'.wi.weight'
                    combine_key_2_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_2)+'.wo.weight'
                    model_state_dict[combine_key_1_wi] = ((model_state_dict[combine_key_1_wi] + model_state_dict[combine_key_2_wi])/2.0).clone().detach()
                    model_state_dict[combine_key_1_wo] = ((model_state_dict[combine_key_1_wo] + model_state_dict[combine_key_2_wo])/2.0).clone().detach()
                    model_state_dict[combine_key_2_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_2_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                # print(model.state_dict()['encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight'].shape)
                # break
        else:
            for block in range(0,12,2):
                for (expert_id_1,expert_id_2) in unique_combinations:
                    combine_key_1_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_1)+'.wi.weight'
                    combine_key_1_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_1)+'.wo.weight'
                    combine_key_2_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_2)+'.wi.weight'
                    combine_key_2_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_2)+'.wo.weight'
                    model_state_dict[combine_key_1_wi] = ((model_state_dict[combine_key_1_wi] + model_state_dict[combine_key_2_wi])/2.0).clone().detach()
                    model_state_dict[combine_key_1_wo] = ((model_state_dict[combine_key_1_wo] + model_state_dict[combine_key_2_wo])/2.0).clone().detach()
                    model_state_dict[combine_key_2_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_2_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                # print(model.state_dict()['encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight'].shape)
                # break 
    if method == 'naive_8_to_2':
        all_numbers = list(range(0, 8))
        unique_combinations, number_list = [], []
        # 生成4个不重复的组合

        selected_numbers = random.sample(all_numbers, 4)
        unique_combinations.append(selected_numbers)
        for i in all_numbers:
            if i not in unique_combinations[0]:
                number_list.append(i)
        unique_combinations.append(number_list)
        print(unique_combinations)
        if start_layer < 6:
            for block in range(start_layer*2,12,2):
                for (expert_id_1,expert_id_2,expert_id_3,expert_id_4) in unique_combinations:
                    combine_key_1_wi = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_1)+'.wi.weight'
                    combine_key_1_wo = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_1)+'.wo.weight'
                    combine_key_2_wi = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_2)+'.wi.weight'
                    combine_key_2_wo = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_2)+'.wo.weight'
                    combine_key_3_wi = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_3)+'.wi.weight'
                    combine_key_3_wo = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_3)+'.wo.weight'
                    combine_key_4_wi = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_4)+'.wi.weight'
                    combine_key_4_wo = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_4)+'.wo.weight'

                    model_state_dict[combine_key_1_wi] = ((model_state_dict[combine_key_1_wi] + model_state_dict[combine_key_2_wi] + \
                                                          model_state_dict[combine_key_3_wi] + model_state_dict[combine_key_4_wi])/4.0).clone().detach()
                    model_state_dict[combine_key_1_wo] = ((model_state_dict[combine_key_1_wo] + model_state_dict[combine_key_2_wo] + \
                                                          model_state_dict[combine_key_3_wo] + model_state_dict[combine_key_4_wo])/4.0).clone().detach()
                    model_state_dict[combine_key_2_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_2_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_3_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_3_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_4_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_4_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
        if start_layer >= 6:
            for block in range((start_layer-6)*2,12,2):
                for (expert_id_1,expert_id_2,expert_id_3,expert_id_4) in unique_combinations:
                    combine_key_1_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_1)+'.wi.weight'
                    combine_key_1_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_1)+'.wo.weight'
                    combine_key_2_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_2)+'.wi.weight'
                    combine_key_2_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_2)+'.wo.weight'
                    combine_key_3_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_3)+'.wi.weight'
                    combine_key_3_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_3)+'.wo.weight'
                    combine_key_4_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_4)+'.wi.weight'
                    combine_key_4_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_4)+'.wo.weight'

                    model_state_dict[combine_key_1_wi] = ((model_state_dict[combine_key_1_wi] + model_state_dict[combine_key_2_wi] + \
                                                          model_state_dict[combine_key_3_wi] + model_state_dict[combine_key_4_wi])/4.0).clone().detach()
                    model_state_dict[combine_key_1_wo] = ((model_state_dict[combine_key_1_wo] + model_state_dict[combine_key_2_wo] + \
                                                          model_state_dict[combine_key_3_wo] + model_state_dict[combine_key_4_wo])/4.0).clone().detach()
                    model_state_dict[combine_key_2_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_2_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_3_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_3_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_4_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_4_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                # print(model.state_dict()['encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight'].shape)
                # break
        else:
            for block in range(0,12,2):
                for (expert_id_1,expert_id_2,expert_id_3,expert_id_4) in unique_combinations:
                    combine_key_1_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_1)+'.wi.weight'
                    combine_key_1_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_1)+'.wo.weight'
                    combine_key_2_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_2)+'.wi.weight'
                    combine_key_2_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_2)+'.wo.weight'
                    combine_key_3_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_3)+'.wi.weight'
                    combine_key_3_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_3)+'.wo.weight'
                    combine_key_4_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_4)+'.wi.weight'
                    combine_key_4_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_4)+'.wo.weight'
                    model_state_dict[combine_key_1_wi] = ((model_state_dict[combine_key_1_wi] + model_state_dict[combine_key_2_wi] + \
                                                          model_state_dict[combine_key_3_wi] + model_state_dict[combine_key_4_wi])/4.0).clone().detach()
                    model_state_dict[combine_key_1_wo] = ((model_state_dict[combine_key_1_wo] + model_state_dict[combine_key_2_wo] + \
                                                          model_state_dict[combine_key_3_wo] + model_state_dict[combine_key_4_wo])/4.0).clone().detach()
                    model_state_dict[combine_key_2_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_2_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_3_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_3_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_4_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_4_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                # print(model.state_dict()['encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight'].shape)
                # break
    if method == 'naive_8_to_1':
        unique_combinations = [list(range(0, 8))]
        if start_layer < 6:
            for block in range(start_layer*2,12,2):
                for (expert_id_1,expert_id_2,expert_id_3,expert_id_4,expert_id_5,expert_id_6,expert_id_7,expert_id_8) in unique_combinations:
                    combine_key_1_wi = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_1)+'.wi.weight'
                    combine_key_1_wo = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_1)+'.wo.weight'
                    combine_key_2_wi = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_2)+'.wi.weight'
                    combine_key_2_wo = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_2)+'.wo.weight'
                    combine_key_3_wi = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_3)+'.wi.weight'
                    combine_key_3_wo = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_3)+'.wo.weight'
                    combine_key_4_wi = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_4)+'.wi.weight'
                    combine_key_4_wo = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_4)+'.wo.weight'
                    combine_key_5_wi = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_5)+'.wi.weight'
                    combine_key_5_wo = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_5)+'.wo.weight'
                    combine_key_6_wi = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_6)+'.wi.weight'
                    combine_key_6_wo = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_6)+'.wo.weight'
                    combine_key_7_wi = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_7)+'.wi.weight'
                    combine_key_7_wo = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_7)+'.wo.weight'
                    combine_key_8_wi = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_8)+'.wi.weight'
                    combine_key_8_wo = 'encoder.block.'+str(block+1)+'.layer.1.mlp.experts.expert_'+str(expert_id_8)+'.wo.weight'

                    model_state_dict[combine_key_1_wi] = ((model_state_dict[combine_key_1_wi] + model_state_dict[combine_key_2_wi] + \
                                                          model_state_dict[combine_key_3_wi] + model_state_dict[combine_key_4_wi] +\
                                                            model_state_dict[combine_key_5_wi] + model_state_dict[combine_key_6_wi] +\
                                                                model_state_dict[combine_key_7_wi] + model_state_dict[combine_key_8_wi]) \
                                                            /8.0).clone().detach()
                    model_state_dict[combine_key_1_wo] = ((model_state_dict[combine_key_1_wo] + model_state_dict[combine_key_2_wo] + \
                                                          model_state_dict[combine_key_3_wo] + model_state_dict[combine_key_4_wo] +\
                                                            model_state_dict[combine_key_5_wo] + model_state_dict[combine_key_6_wo] +\
                                                                model_state_dict[combine_key_7_wo] + model_state_dict[combine_key_8_wo]) \
                                                            /8.0).clone().detach()
                    
                    model_state_dict[combine_key_2_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_2_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_3_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_3_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_4_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_4_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_5_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_5_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_6_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_6_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_7_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_7_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_8_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_8_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
        if start_layer >= 6:
            for block in range((start_layer-6)*2,12,2):
                for (expert_id_1,expert_id_2,expert_id_3,expert_id_4,expert_id_5,expert_id_6,expert_id_7,expert_id_8) in unique_combinations:
                    combine_key_1_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_1)+'.wi.weight'
                    combine_key_1_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_1)+'.wo.weight'
                    combine_key_2_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_2)+'.wi.weight'
                    combine_key_2_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_2)+'.wo.weight'
                    combine_key_3_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_3)+'.wi.weight'
                    combine_key_3_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_3)+'.wo.weight'
                    combine_key_4_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_4)+'.wi.weight'
                    combine_key_4_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_4)+'.wo.weight'
                    combine_key_5_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_5)+'.wi.weight'
                    combine_key_5_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_5)+'.wo.weight'
                    combine_key_6_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_6)+'.wi.weight'
                    combine_key_6_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_6)+'.wo.weight'
                    combine_key_7_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_7)+'.wi.weight'
                    combine_key_7_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_7)+'.wo.weight'
                    combine_key_8_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_8)+'.wi.weight'
                    combine_key_8_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_8)+'.wo.weight'
                    model_state_dict[combine_key_1_wi] = ((model_state_dict[combine_key_1_wi] + model_state_dict[combine_key_2_wi] + \
                                                          model_state_dict[combine_key_3_wi] + model_state_dict[combine_key_4_wi] +\
                                                            model_state_dict[combine_key_5_wi] + model_state_dict[combine_key_6_wi] +\
                                                                model_state_dict[combine_key_7_wi] + model_state_dict[combine_key_8_wi]) \
                                                            /8.0).clone().detach()
                    model_state_dict[combine_key_1_wo] = ((model_state_dict[combine_key_1_wo] + model_state_dict[combine_key_2_wo] + \
                                                          model_state_dict[combine_key_3_wo] + model_state_dict[combine_key_4_wo] +\
                                                            model_state_dict[combine_key_5_wo] + model_state_dict[combine_key_6_wo] +\
                                                                model_state_dict[combine_key_7_wo] + model_state_dict[combine_key_8_wo]) \
                                                            /8.0).clone().detach()
                    model_state_dict[combine_key_2_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_2_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_3_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_3_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_4_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_4_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_5_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_5_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_6_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_6_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_7_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_7_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_8_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_8_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                # print(model.state_dict()['encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight'].shape)
                # break
        else:
            for block in range(0,12,2):
                for (expert_id_1,expert_id_2,expert_id_3,expert_id_4,expert_id_5,expert_id_6,expert_id_7,expert_id_8) in unique_combinations:
                    combine_key_1_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_1)+'.wi.weight'
                    combine_key_1_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_1)+'.wo.weight'
                    combine_key_2_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_2)+'.wi.weight'
                    combine_key_2_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_2)+'.wo.weight'
                    combine_key_3_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_3)+'.wi.weight'
                    combine_key_3_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_3)+'.wo.weight'
                    combine_key_4_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_4)+'.wi.weight'
                    combine_key_4_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_4)+'.wo.weight'
                    combine_key_5_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_5)+'.wi.weight'
                    combine_key_5_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_5)+'.wo.weight'
                    combine_key_6_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_6)+'.wi.weight'
                    combine_key_6_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_6)+'.wo.weight'
                    combine_key_7_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_7)+'.wi.weight'
                    combine_key_7_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_7)+'.wo.weight'
                    combine_key_8_wi = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_8)+'.wi.weight'
                    combine_key_8_wo = 'decoder.block.'+str(block+1)+'.layer.2.mlp.experts.expert_'+str(expert_id_8)+'.wo.weight'
                    model_state_dict[combine_key_1_wi] = ((model_state_dict[combine_key_1_wi] + model_state_dict[combine_key_2_wi] + \
                                                          model_state_dict[combine_key_3_wi] + model_state_dict[combine_key_4_wi] +\
                                                            model_state_dict[combine_key_5_wi] + model_state_dict[combine_key_6_wi] +\
                                                                model_state_dict[combine_key_7_wi] + model_state_dict[combine_key_8_wi]) \
                                                            /8.0).clone().detach()
                    model_state_dict[combine_key_1_wo] = ((model_state_dict[combine_key_1_wo] + model_state_dict[combine_key_2_wo] + \
                                                          model_state_dict[combine_key_3_wo] + model_state_dict[combine_key_4_wo] +\
                                                            model_state_dict[combine_key_5_wo] + model_state_dict[combine_key_6_wo] +\
                                                                model_state_dict[combine_key_7_wo] + model_state_dict[combine_key_8_wo]) \
                                                            /8.0).clone().detach()
                    model_state_dict[combine_key_2_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_2_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_3_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_3_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_4_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_4_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_5_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_5_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_6_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_6_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_7_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_7_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                    model_state_dict[combine_key_8_wi] = (model_state_dict[combine_key_1_wi]).clone().detach()
                    model_state_dict[combine_key_8_wo] = (model_state_dict[combine_key_1_wo]).clone().detach()
                # print(model.state_dict()['encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight'].shape)
                # break
    model.load_state_dict(model_state_dict)
    return model

if __name__ == "__main__":

    # dataset = load_dataset("samsum")

    # tokenizer = AutoTokenizer.from_pretrained("emre/switch-base-8-finetuned-samsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("emre/switch-base-8-finetuned-samsum")
    # print(model.state_dict().keys())

    method = ['original', 'naive_8_to_1', 'naive_8_to_2', 'naive_8_to_4']
    reload_model(method[-1], model,0)
    # for i in model.state_dict().keys():

# model = model.cuda()
# model.eval()
# index = 0
# batch_size = 1

# expert_list,token_list = [], []

# for i in range(0,len(dataset['test']),batch_size):
    
#     batchs = dataset['test'][i:i+batch_size]
#     # print(typei['dialogue'], i['summary'], i['id'])
#     doc, sum, id = batchs['dialogue'], batchs['summary'], batchs['id']
#     # tokenizer.batch_encode_plus
#     input_ids = tokenizer.batch_encode_plus(doc, truncation=True,padding=True, return_tensors="pt") 
#     labels = tokenizer.batch_encode_plus(sum, truncation=True,padding=True, return_tensors="pt")
#     # input_ids = tokenizer(doc, return_tensors="pt").input_ids 
#     # labels = tokenizer(sum, return_tensors="pt").input_ids 
#     input_ids['input_ids'] = input_ids['input_ids'].cuda()
#     input_ids['attention_mask'] = input_ids['attention_mask'].cuda()
#     labels['input_ids'] = labels['input_ids'].cuda()
#     # print(doc,sum,id)
#     outputs = model(**input_ids, labels=labels['input_ids'])
    
#     encoder_router = outputs.encoder_router_logits
#     decoder_router = outputs.decoder_router_logits
#     encoder_token = outputs.encoder_token_embed
#     decoder_token = outputs.decoder_token_embed

#     if index == 0:
#         for i in encoder_router:
#             if len(i) == 2:
#                 print(i[1].shape)
#                 expert_list.append(i[1])
#         for i in decoder_router:
#             if len(i) == 2:
#                 print(i[1].shape)
#                 expert_list.append(i[1])  

#         for i in encoder_token:
#             if not isinstance(i, tuple):
#                 print(i.shape)
#                 token_list.append(i)
#         for i in decoder_token:
#             if not isinstance(i, tuple):
#                 print(i.shape)
#                 token_list.append(i)  
#     else:
#         k = 0
#         for i in encoder_router:
#             if len(i) == 2:
#                 print(i[1].shape)
#                 expert_list[k] = torch.hstack((expert_list[k], i[1]))
#                 k+=1
#         for i in decoder_router:
#             if len(i) == 2:
#                 print(i[1].shape)
#                 expert_list[k] = torch.hstack((expert_list[k], i[1]))
#                 k+=1
#         k = 0
#         for i in encoder_token:
#             if not isinstance(i, tuple):
#                 print(i.shape)
#                 token_list[k] = torch.hstack((token_list[k], i))
#                 k+=1
#         for i in decoder_token:
#             if not isinstance(i, tuple):
#                 print(i.shape)
#                 token_list[k] = torch.hstack((token_list[k], i))
#                 k+=1

#     # outputs = model.generate(input_ids)
#     # print(outputs.shape)
    
#     # outputs = list(outputs.squeeze(0).cpu().numpy())
#     # outputs = tokenizer.batch_decode(outputs)
#     # outputs = tokenizer.convert_ids_to_tokens(outputs)
#     # outputs = tokenizer.convert_tokens_to_string(outputs)
#     # print(sum, outputs)
#     index += 1
#     if index == 10:
#         break
# fig.tokenpcafig(expert_list,token_list,index)
# print(model)