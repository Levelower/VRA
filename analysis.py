import copy
import torch
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from bert4vec import Bert4Vec
from itertools import product
from attack import Attacker
from PIL import Image, ImageFont, ImageDraw
from evaluate import calc_success_num


class Analyzer(object):
    def __init__(self, model_src_path, model_tgt_path, device):
        self.attacker = Attacker(model_src_path, model_tgt_path, device, method=3)

    def search_vision_candidate_set(self, text):
        candidate_set = []

        # tokenize结果不包含结尾标识，tokens结果有结尾标识，将其去除
        tokens = self.attacker.tokenizer_src(text, return_tensors='pt').input_ids.numpy()[0][:-1]
        attack_res = [i.replace('▁','') for i in self.attacker.tokenizer_src.tokenize(text)]

        # 计算token重要度
        importance = self.attacker.get_token_importance(attack_res)

        # 按重要度从高到低排序
        token_ids = np.arange(0, tokens.shape[0])
        tokens_order = sorted(zip(importance, token_ids), reverse=True)
        tokens_order = [pair[1] for pair in tokens_order]

        # 根据文本长度，按比例计算替换的次数上限
        if self.attacker.percent == '1':
            num_changed = 1
        elif self.attacker.percent == '2':
            num_changed = 2
        else:
            num_changed = round(len(text) * float(self.attacker.percent))
        
        # 生成所有的替换组合
        pos_candidate = []
        for i in range(num_changed):
            token = attack_res[tokens_order[i]]

            candidate_char_list, candidate_score_list = self.attacker.get_token_vision_similarity(token)
            
            pos_candidate.append(candidate_char_list)

        replacement_combinations = product(*pos_candidate)

        # 生成候选集
        for replacement_combination in replacement_combinations:
            atk = copy.deepcopy(attack_res)
            for i in range(num_changed):
                atk[tokens_order[i]] = replacement_combination[i]

            atk_text = ''.join(atk)
            candidate_set.append(atk_text)

        return candidate_set

    def search_semantics_candidate_set(self, text):
        candidate_set = []

        # tokenize结果不包含结尾标识，tokens结果有结尾标识，将其去除
        tokens = self.attacker.tokenizer_src(text, return_tensors='pt').input_ids.numpy()[0][:-1]
        attack_res = [i.replace('▁','') for i in self.attacker.tokenizer_src.tokenize(text)]

        # 计算token重要度
        importance = self.attacker.get_token_importance(attack_res)

        # 按重要度从高到低排序
        token_ids = np.arange(0, tokens.shape[0])
        tokens_order = sorted(zip(importance, token_ids), reverse=True)
        tokens_order = [pair[1] for pair in tokens_order]

        # 根据文本长度，按比例计算替换的次数上限
        if self.attacker.percent == '1':
            num_changed = 1
        elif self.attacker.percent == '2':
            num_changed = 2
        else:
            num_changed = round(len(text) * float(self.attacker.percent))
        
        # 生成所有的替换组合
        pos = []
        pos_candidate = []

        cnt = 0
        for idx, i in enumerate(tokens_order):
            context = self.attacker.text_map.get(tokens[i], "")
            if context == "" or context in string.punctuation:
                continue
            candidate_char_list = self.attacker.get_most_similar_topN(context, 20, 0.8)
            if len(candidate_char_list) <= 0:
                continue
            pos_candidate.append(candidate_char_list)
            pos.append(i)
            cnt += 1
            if cnt == num_changed:
                break

        replacement_combinations = product(*pos_candidate)

        # 生成候选集
        for replacement_combination in replacement_combinations:
            atk = copy.deepcopy(attack_res)
            for i in range(num_changed):
                atk[pos[i]] = replacement_combination[i]

            atk_text = ''.join(atk)
            candidate_set.append(atk_text)

        return candidate_set
    
    def calc_cadidate_set_size_and_asr(self, text_src, text_tgt, save_path='result/search_space/test1'):
        text_tgt_translation = self.attacker.translate_tgt(text_tgt)

        original_vision_cadidate = self.search_vision_candidate_set(text_src)
        enhanced_vision_candidate = self.search_vision_candidate_set(text_tgt_translation)
        
        vision_cadidate = original_vision_cadidate + enhanced_vision_candidate
        semantics_candidate = self.search_semantics_candidate_set(text_src)

        vision_size = len(vision_cadidate)
        semantics_size = len(semantics_candidate)

        origin_translate = self.attacker.translate_src(text_src)
        source = origin_translate
        reference = text_tgt

        vision_cadidate_file = open(save_path + '/vision_cadidate.txt', 'w')
        semantics_cadidate_file = open(save_path + '/semantics_cadidate.txt', 'w')
        vision_success_file = open(save_path + '/vision_success.txt', 'w')
        semantics_success_file = open(save_path + '/semantics_success.txt', 'w')

        vision_all, vision_suc = 0, 0
        for text in vision_cadidate:
            candidate = self.attacker.translate_src(text)
            all, suc, asr = calc_success_num(source, candidate, reference)
            vision_all += all
            vision_suc += suc
            if suc == 1:
                vision_success_file.writelines(text + '\n')
                vision_success_file.flush()
            vision_cadidate_file.writelines(text + '\n')
            vision_cadidate_file.flush()

        semantics_all, semantics_suc = 0, 0
        for text in semantics_candidate:
            candidate = self.attacker.translate_src(text)
            all, suc, asr = calc_success_num(source, candidate, reference)
            semantics_all += all
            semantics_suc += suc
            if suc == 1:
                semantics_success_file.writelines(text + '\n')
                semantics_success_file.flush()
            semantics_cadidate_file.writelines(text + '\n')
            semantics_cadidate_file.flush()

        return vision_all, vision_suc, semantics_all, semantics_suc


def draw_space_image(path='result/search_space/test2', fontsize=30, format='svg', all_size=600, suc_size=150):
    vision_cadidate_file = open(path + '/vision_cadidate.txt', 'r')
    semantics_cadidate_file = open(path + '/semantics_cadidate.txt', 'r')
    vision_success_file = open(path + '/vision_success.txt', 'r')
    semantics_success_file = open(path + '/semantics_success.txt', 'r')

    vision_cadidate = vision_cadidate_file.readlines()
    semantics_cadidate = semantics_cadidate_file.readlines()
    vision_success = vision_success_file.readlines()
    semantics_success = semantics_success_file.readlines()

    vision_cadidate = [text.strip() for text in vision_cadidate]
    semantics_cadidate = [text.strip() for text in semantics_cadidate]
    vision_success = [text.strip() for text in vision_success]
    semantics_success = [text.strip() for text in semantics_success]

    vision_success_id = [vision_cadidate.index(text) for text in vision_success]
    semantics_success_id = [semantics_cadidate.index(text) for text in semantics_success]

    model = Bert4Vec(mode='simbert-base', model_name_or_path='./model/simbert-base-chinese')
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=min([11, len(vision_success), len(semantics_success)])-1)

    vision_vecs = model.encode(vision_cadidate, convert_to_numpy=True, normalize_to_unit=False)
    vision_vecs_2 = tsne.fit_transform(vision_vecs)

    semantics_vecs = model.encode(semantics_cadidate, convert_to_numpy=True, normalize_to_unit=False)
    semantics_vecs_2 = tsne.fit_transform(semantics_vecs)

    plt.rc('legend', fontsize=fontsize)
    plt.rcParams['axes.linewidth'] = 3

    plt.figure(figsize=(20, 16))
    axes = plt.subplot(111)
    type1 = axes.scatter(vision_vecs_2[:, 0], vision_vecs_2[:, 1], color='#eca689', s=all_size, label='vision_all')
    type2 = axes.scatter(vision_vecs_2[vision_success_id, 0], vision_vecs_2[vision_success_id, 1], color='#e3716e', s=suc_size, label='vision_suc')
    type3 = axes.scatter(semantics_vecs_2[:, 0], semantics_vecs_2[:, 1], color='#bbd1e7', s=all_size, label='semantics_all')
    type4 = axes.scatter(semantics_vecs_2[semantics_success_id, 0], semantics_vecs_2[semantics_success_id, 1], color='#6d8bc3', s=suc_size, label='semantics_suc')
    plt.xticks([])
    plt.yticks([])
    l1 = axes.legend((type1, type2), (u'vision_all', u'vision_suc'), loc='upper left', bbox_to_anchor=(0.0, 1.0))
    axes.legend((type3, type4), (u'semantics_all', u'semantics_suc'), loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.gca().add_artist(l1)
    axes.set_xlim(-135, 135)
    axes.set_ylim(-110, 120)
    axes.axis('off')
    axes.set_frame_on(False)
    plt.savefig(path + f'/all.{format}', format=format, bbox_inches='tight')

    plt.figure(figsize=(20, 16))
    axes = plt.subplot(111)
    type1 = axes.scatter(vision_vecs_2[:, 0], vision_vecs_2[:, 1], color='#eca689', s=all_size, label='vision_all')
    type2 = axes.scatter(vision_vecs_2[vision_success_id, 0], vision_vecs_2[vision_success_id, 1], color='#e3716e', s=suc_size, label='vision_suc')
    plt.xticks([])
    plt.yticks([])
    axes.legend((type1, type2), (u'vision_all', u'vision_suc'), loc='upper center', bbox_to_anchor=(0.5, 1.2), borderaxespad=0)
    axes.set_xlim(-130, 130)
    axes.set_ylim(-130, 130)
    plt.savefig(path + f'/vision.{format}', format=format, bbox_inches='tight')

    plt.figure(figsize=(20, 16))
    axes = plt.subplot(111)
    type3 = axes.scatter(semantics_vecs_2[:, 0], semantics_vecs_2[:, 1], color='#bbd1e7', s=all_size, label='semantics_all')
    type4 = axes.scatter(semantics_vecs_2[semantics_success_id, 0], semantics_vecs_2[semantics_success_id, 1], color='#6d8bc3', s=suc_size, label='semantics_suc')
    plt.xticks([])
    plt.yticks([])   
    axes.legend((type3, type4), (u'semantics_all', u'semantics_suc'), loc='upper center', bbox_to_anchor=(0.5, 1.2), borderaxespad=0)
    axes.set_xlim(-60, 51)
    axes.set_ylim(-60, 51)
    plt.savefig(path + f'/semantics.{format}', format=format, bbox_inches='tight')


def save_image_difference(text_ref, text_ours, text_adv, text_drtt, text_hotflip, text_ta, text_ss, name):
    def draw_bold_text(draw, pos, text, font, fill, thick=10):
        for dx in range(-thick, thick+1):
            for dy in range(-thick, thick+1):
                draw.text((pos[0]+dx, pos[1]+dy), text, font=font, fill=fill)

    multi = 20
    font = ImageFont.truetype('./utils/chinese_fonts/simsun.ttc', 22 * multi)

    ws = []
    for text in [text_ref, text_ours, text_adv, text_drtt, text_hotflip, text_ta, text_ss]:
        _, _, w, _ = font.getbbox(text)
        ws.append(w)

    margin = 19 * multi
    im_w = max(ws) + margin
    im_h = 41 * multi

    texts = [text_ours, text_adv, text_drtt, text_hotflip, text_ta, text_ss]
    names = ['Ours', 'ADV', 'DRTT', 'HotFlip', 'TA', 'SS']

    ori_image = Image.new("RGB", (im_w, im_h), 'white')
    ori_draw = ImageDraw.Draw(ori_image)
    draw_bold_text(ori_draw, (margin // 2, margin // 2), text_ref, font=font, fill="black")
    ori_image.save(f'./result/diff_image/' + name + f'/ori.png')
        
    for text_atk, name_atk in zip(texts, names):
        # 绘图
        hoffset, woffset = margin // 2, margin // 2
        final_image = Image.new("RGB", (im_w, im_h), 'black')
        final_draw = ImageDraw.Draw(final_image)
        for i, chr in enumerate(text_atk):
            _, _, w, _ = font.getbbox(chr)
            if i > len(text_ref) - 1 or chr != text_ref[i]:
                draw_bold_text(final_draw, (hoffset, woffset), chr, font=font, fill="white")
            hoffset += w

        # 计算差异像素
        image_ref = Image.new("RGB", (im_w, im_h), 'black')
        image_atk = Image.new("RGB", (im_w, im_h), 'black')
        draw_ref = ImageDraw.Draw(image_ref)
        draw_atk = ImageDraw.Draw(image_atk)
        offset = margin // 2
        draw_bold_text(draw_ref, (offset, offset), text_ref, font=font, fill="white")
        draw_bold_text(draw_atk, (offset, offset), text_atk, font=font, fill="white")
        arr_ref = np.array(image_ref.convert('L'))
        arr_atk = np.array(image_atk.convert('L'))
        diff = np.sum(arr_ref != arr_atk) / (arr_ref.shape[0] * arr_ref.shape[1])

        ori_image = Image.new("RGB", (im_w, im_h), 'white')
        ori_draw = ImageDraw.Draw(ori_image)
        draw_bold_text(ori_draw, (margin // 2, margin // 2), text_atk, font=font, fill="black")
        final_image.save(f'./result/diff_image/' + name + f'/atk_{name_atk}_{round(diff, 3)}.png')
        ori_image.save(f'./result/diff_image/' + name + f'/ori_{name_atk}_{round(diff, 3)}.png')


if __name__ == '__main__':
    text_src = '我想向你们致敬'
    text_tgt = 'I just really wanted to acknowledge you'

    text_src = '今天我有好消息要讲。'
    text_tgt = 'And there is some good news to report today.'

    text_src = '他从不愿意与家人争吵'
    text_tgt = 'He never wanted to be in any kind of altercation'

    text_src = '一个手势结束一场危机'
    text_tgt = 'It was a gesture that ended a crisis'

    text_src = '他已断然否认该种说法'
    text_tgt = 'He\'s denied that emphatically'

    text_src = '这个是我自己一个小猜测'
    text_tgt = 'This could be my little assessment'

    text_src = '同样对于我们自己也非常有用'
    text_tgt = 'But it will also be useful for us'

    model_src_path = './model/opus-mt-zh-en'
    model_tgt_path = './model/opus-mt-en-zh'

    device = torch.device('cuda:2')

    analyzer = Analyzer(model_src_path, model_tgt_path, device)
    
    res = analyzer.calc_cadidate_set_size_and_asr(text_src, text_tgt, save_path='result/search_space/test10')
    print('中文: ', text_src)
    print('英文: ', text_tgt)
    print('视觉搜索空间大小: {:<10}有效样本数量: {:<10}成功率: {:<10}'.format(res[0], res[1], res[1] / res[0]))
    print('语义搜索空间大小: {:<10}有效样本数量: {:<10}成功率: {:<10}'.format(res[2], res[3], res[3] / res[2]))
    print('总体搜索空间大小: {:<10}有效样本数量: {:<10}成功率: {:<10}'.format(res[0]+res[2], res[1]+res[3], (res[1]+res[3])/(res[0]+res[2])))

    draw_space_image(path='result/search_space/test9', format='svg')

    text_ref      = '平昌冬奥会圣火抵达韩国'
    text_ours     = '平昌冬奥会圣火扺达韩回。'
    text_adv      = '平晶冬奥会圣火抵达韩guo'
    text_drtt     = '平昌的圣火抵达北京。'
    text_hotflip  = '平昌冬奥会圣火和韩国'
    text_ta       = '平動奥会圣火焰达到了韩国'
    text_ss       = '平尺度法学奥会圣火的成功'

    save_image_difference(text_ref, text_ours, text_adv, text_drtt, text_hotflip, text_ta, text_ss, 'test2')