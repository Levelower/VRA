import copy
import torch
import pickle
import string
import numpy as np
from transformers import MarianMTModel, MarianTokenizer
from vision import vision_model
from saliency import saliency_model
from semantics import semantics_model 
from evaluate import calc_bleu


class Attacker(object):
    def __init__(
        self, model_src_path, model_tgt_path, device, method,
        percent='0.2', thresh=0.95, sc='all', search_method='vision',
        vision_constraint=True
    ):
        # 从本地加载huggingface模型
        self.tokenizer_src = MarianTokenizer.from_pretrained(model_src_path)
        self.model_src = MarianMTModel.from_pretrained(model_src_path)
        self.tokenizer_tgt = MarianTokenizer.from_pretrained(model_tgt_path)
        self.model_tgt = MarianMTModel.from_pretrained(model_tgt_path)

        # 将模型加载至GPU
        self.device = device
        self.model_src = self.model_src.to(self.device)
        self.model_tgt = self.model_tgt.to(self.device)
        
        # 加载视觉相似字列表
        vision_file = './utils/chinese_characters/vision_similar_char_score.pkl'
        with open(vision_file, 'rb') as file:
            self.vision_similar_chars = pickle.load(file)

        radicals_file = './utils/chinese_characters/radicals_similar_char_score.pkl'
        with open(radicals_file, 'rb') as file:
            self.radicals_similar_chars = pickle.load(file)

        # 加载重要度计算模型
        model_saliency_path = './model/chinese-bert-wwm-ext'
        self.saliency_model = saliency_model(model_saliency_path, device=device)

        # 加载视觉相似度模型
        self.vision_model = vision_model()

        # 加载语义相似度计算模型
        model_semantics_path = './model/all-MiniLM-L6-v2'
        self.semantics_model = semantics_model(model_semantics_path, device=device)

        # 语义搜索部分
        self.token_vocab = self.tokenizer_src.get_vocab()
        self.text_map = {label: text for text, label in self.token_vocab.items()}
        self.token_embeddings = self.model_src.get_input_embeddings().weight.data

        # merge方法
        self.set_merge_method(method)

        # 超参
        self.percent = percent
        self.thresh = thresh
        self.sc = sc
        self.search_method = search_method
        self.vision_constraint = vision_constraint

    def set_merge_method(self, method):
        self.attack_method = [
            self.attack1, self.attack2, 
            self.attack3, self.attack4,
            self.attack5, self.attack6,
            self.attack7
        ][method-1]

    # 翻译src到tgt
    def translate_src(self, text, length=None):
        input_ids = self.tokenizer_src.encode(text, return_tensors="pt").to(self.device)
        if length is None:
            output = self.model_src.generate(input_ids)
        else:
            output = self.model_src.generate(input_ids, max_length=length, min_length=length)
            
        output_text = self.tokenizer_src.decode(output[0], skip_special_tokens=True)
        return output_text

    # 翻译tgt到src
    def translate_tgt(self, text, length=None):
        input_ids = self.tokenizer_tgt.encode(text, return_tensors="pt").to(self.device)
        if length is None:
            output = self.model_tgt.generate(input_ids)
        else:
            output = self.model_tgt.generate(input_ids, max_length=length, min_length=length)
        output_text = self.tokenizer_tgt.decode(output[0], skip_special_tokens=True)
        return output_text
    
    # 为句子中的每个token计算重要度
    def get_token_importance(self, words):
        importance = self.saliency_model.scores(words)
        return importance
    
    # 为某个token查找视觉相似的token
    def get_token_vision_similarity(self, text):
        candidate_char_list = []
        candidate_score_list = []
        for idx, char in enumerate(text):
            r_tmp = []
            s_tmp = []

            if self.sc == 'glyph':
                if char in self.vision_similar_chars:
                    for first in self.vision_similar_chars[char]:
                        r_tmp.append(first[0])
                        s_tmp.append(first[1])

            elif self.sc == 'radicals':
                if char in self.radicals_similar_chars.keys():
                    for first in self.radicals_similar_chars[char]:
                        r_tmp.append(first[0])
                        s_tmp.append(first[1])
            else:
                if char in self.radicals_similar_chars.keys():
                    for first in self.radicals_similar_chars[char]:
                        r_tmp.append(first[0])
                        s_tmp.append(first[1])
                
                elif char in self.vision_similar_chars:
                    for first in self.vision_similar_chars[char]:
                        r_tmp.append(first[0])
                        s_tmp.append(first[1])

            if len(r_tmp) > 0:     
                candidate_char_list = candidate_char_list + [text[: idx] + c + text[idx+1:] for c in r_tmp]
                candidate_score_list = candidate_score_list + s_tmp

        return candidate_char_list, candidate_score_list
    
    # 为某个token查找语义相似，即余弦相似度最高的token
    def get_most_similar_topN(self, text, topn=10, thresh=0.9):
        input_token = text

        input_vector_matrix = torch.unsqueeze(
            self.token_embeddings[self.token_vocab[input_token]], 0
        ).repeat(len(self.token_embeddings), 1)
        
        cos_sim_list = torch.cosine_similarity(
            input_vector_matrix, self.token_embeddings, 1
        ).tolist()

        cos_sim_list = [(self.text_map[i], cos_sim_list[i]) for i in range(len(cos_sim_list))]
        cos_sim_list.sort(key=lambda x: x[1], reverse=True)

        cnt = 0
        topn_tokens = []
        for i in range(1, len(cos_sim_list)):
            if cos_sim_list[i][1] > thresh:
                topn_tokens.append(cos_sim_list[i][0])
                cnt += 1
            if cnt >= topn:
                break

        return topn_tokens
    
    # 计算两个句子之间的整体视觉相似程度
    def get_sentence_vision_similarity(self, text1, text2):
        score_overall = self.vision_model.get_lpips_similarity(text1, text2)
        return score_overall

    # 计算两个句子之间的整体语义相似程度
    def get_sentence_semantic_similarity(self, text1, text2):
        cosine_sim = self.semantics_model.get_sentence_similarity([text1, text2])
        return cosine_sim

    # 视觉替换生成对抗样本
    def search_samples_by_vision(self, text):    
        # tokenize结果不包含结尾标识，tokens结果有结尾标识，将其去除
        tokens = self.tokenizer_src(text, return_tensors='pt').input_ids.numpy()[0][:-1]
        attack_res = [i.replace('▁','') for i in self.tokenizer_src.tokenize(text)]

        # 计算token重要度
        importance = self.get_token_importance(attack_res)

        # 按重要度从高到低排序
        token_ids = np.arange(0, tokens.shape[0])
        tokens_order = sorted(zip(importance, token_ids), reverse=True)
        tokens_order = [pair[1] for pair in tokens_order]

        # 根据文本长度，按比例计算替换的次数上限
        if self.percent == '1':
            num_changed = 1
        elif self.percent == '2':
            num_changed = 2
        else:
            num_changed = round(len(text) * float(self.percent))

        # 替换次数计数
        cnt = 0 

        # 按重要度高低依次进行替换，若替换后相似度低于阈值则放弃此次替换，达到替换次数上限后停止
        for token_idx in tokens_order:
            token = attack_res[token_idx]

            candidates, scores = self.get_token_vision_similarity(token)
            
            # 获取替换后感知变化最小的作为对抗样本
            res = []
            if len(candidates) > 0:
                sim_thresh = 0
                for candidate, score in zip(candidates, scores):
                    atk = copy.deepcopy(attack_res)
                    atk[token_idx] = candidate
                    atk_text = ''.join(atk)

                    # 句子整体的相似度、替换词与原词的局部相似度，构成相似度分数
                    sim_score = (self.get_sentence_vision_similarity(text, atk_text) + score) / 2
                    
                    res.append((candidate, sim_score))
                    sim_thresh = max(sim_thresh, sim_score)

                # 得到整体相似度最高的
                max_token = max(res, key=lambda x: x[1])[0] 

                # 控制整体视觉相似度不得低于阈值，否则不替换
                if self.vision_constraint:
                    if sim_thresh > self.thresh:
                        attack_res[token_idx] = max_token
                        cnt = cnt + 1
                else:
                    attack_res[token_idx] = max_token
                    cnt = cnt + 1

                # 达到替换比例则停止
                if cnt >= num_changed:
                    break

        return (''.join(attack_res))
    
    def search_samples_by_semantics(self, text):
        # tokenize结果不包含结尾标识，tokens结果有结尾标识，将其去除
        tokens = self.tokenizer_src(text, return_tensors='pt').input_ids.numpy()[0][:-1]
        attack_res = [i.replace('▁','') for i in self.tokenizer_src.tokenize(text)]

        # 计算token重要度
        importance = self.get_token_importance(attack_res)

        # 按重要度从高到低排序
        token_ids = np.arange(0, tokens.shape[0])
        tokens_order = sorted(zip(importance, token_ids), reverse=True)
        tokens_order = [pair[1] for pair in tokens_order]

        # 根据文本长度，按比例计算替换的次数上限
        if self.percent == '1':
            num_changed = 1
        elif self.percent == '2':
            num_changed = 2
        else:
            num_changed = round(len(text) * float(self.percent))

        # 替换次数计数
        cnt = 0 

        # 根据语义相似度进行替换
        res_attack = [self.text_map.get(t, "") for t in tokens]
        for idx, i in enumerate(tokens_order):
            # 标点符号不替换
            context = self.text_map.get(tokens[i], "")
            if context == "" or context in string.punctuation:
                res_attack[i] = context
                continue

            # 根据余弦相似度计算语义相似的topN
            candidate = self.get_most_similar_topN(context, 20, 0.8)
            if len(candidate) <= 0:
                res_attack[i] = context
                continue

            # 找一个使得句子整体语义相似度下降最多的候选词进行替换
            res = []
            for candi in candidate:
                atk_tmp = [self.token_vocab[j] for j in res_attack]
                atk_tmp[idx] = self.token_vocab.get(candi)
                atk_text = ''.join([self.text_map.get(t, '') for t in atk_tmp])
                atk_text = atk_text.replace('▁', ' ').replace('</s>', '')
                semantics_score = self.semantics_model.get_sentence_similarity([text, atk_text])

                # 整体相似度约束
                lpips_score = self.get_sentence_vision_similarity(text, atk_text)
                sim_score = (lpips_score + self.get_sentence_vision_similarity(res_attack[i], candi)) / 2

                res.append((candi, semantics_score, sim_score))
            
            max_token_info = min(res, key=lambda x: x[1])

            # 控制整体视觉相似度不得低于阈值，否则不替换
            if self.vision_constraint:
                if max_token_info[2] > self.thresh:
                    res_attack[i] = max_token_info[0]
                    cnt += 1
            else:
                res_attack[i] = max_token_info[0]
                cnt += 1

            # 达到替换比例则停止
            if cnt >= num_changed:
                break

        return (''.join(res_attack)).replace('▁', '').replace('</s>', '')
    
    # 只使用视觉替换的方法进行攻击
    def attack_by_vision(self, text):
        attack_res_vision = self.search_samples_by_vision(text)
        return attack_res_vision

    # 使用视觉和语义结合的方法进行攻击
    def attack_by_semantics(self, text_src, text_tgt):
        # 是否使用语义作为待替换样本
        semantics_flag = False

        # 设置原始样本为待替换样本
        source = text_src

        # 将参考译文翻译回来，得到语义相似的待替换样本
        text_tgt_translation = self.translate_tgt(text_tgt)

        # 审查语义相似的待替换样本是否满足语义相似条件
        semantic_similarity = self.get_sentence_semantic_similarity(text_tgt_translation, text_src)

        if text_tgt_translation != text_src and semantic_similarity > 0.9:
            source = text_tgt_translation
            semantics_flag = True
        else:
            return False, None

        # 用待替换样本进行视觉替换得到最终的对抗样本
        attack_res_semantics = self.search_samples_by_vision(source)

        return semantics_flag, attack_res_semantics
    
    # 从上述两种方法的结果中挑选一份攻击性最强的
    def attack1(self, text_src, text_tgt):
        vision_result = self.attack_by_vision(text_src)
        flag, semantics_result = self.attack_by_semantics(text_src, text_tgt)

        if not flag:
            return flag, vision_result
        
        vision_translation = self.translate_src(vision_result)
        semantics_translation = self.translate_src(semantics_result)

        vision_bleu = calc_bleu(vision_translation, text_tgt)
        semantics_bleu = calc_bleu(semantics_translation, text_tgt)

        if vision_bleu < semantics_bleu:
            adversarial_example = vision_result
            flag = False
        else:
            adversarial_example = semantics_result

        return flag, adversarial_example
    
    # 从两种对抗样本中挑选一份翻译结果和原始翻译语义相似度最低的
    def attack2(self, text_src, text_tgt):
        vision_result = self.attack_by_vision(text_src)
        flag, semantics_result = self.attack_by_semantics(text_src, text_tgt)

        if not flag:
            return flag, vision_result

        vision_translation = self.translate_src(vision_result)
        semantics_translation = self.translate_src(semantics_result)

        origin_translation = self.translate_src(text_src)

        vision_similar = self.semantics_model.get_sentence_similarity([
            origin_translation, 
            vision_translation
        ])
        semantics_similar = self.semantics_model.get_sentence_similarity([
            origin_translation, 
            semantics_translation
        ])

        if vision_similar < semantics_similar:
            adversarial_example = vision_result
            flag = False
        else:
            adversarial_example = semantics_result

        return flag, adversarial_example
    
    # 从两种对抗样本中挑选一份翻译结果和参考翻译语义相似度最低的
    def attack3(self, text_src, text_tgt):
        vision_result = self.attack_by_vision(text_src)
        flag, semantics_result = self.attack_by_semantics(text_src, text_tgt)

        if not flag:
            return flag, vision_result

        vision_translation = self.translate_src(vision_result)
        semantics_translation = self.translate_src(semantics_result)

        vision_similar = self.semantics_model.get_sentence_similarity([
            text_tgt, 
            vision_translation
        ])
        semantics_similar = self.semantics_model.get_sentence_similarity([
            text_tgt, 
            semantics_translation
        ])

        if vision_similar < semantics_similar:
            adversarial_example = vision_result
            flag = False
        else:
            adversarial_example = semantics_result

        return flag, adversarial_example
    
    # 从两种对抗样本中挑选一份和原始样本语义相似度最高的
    def attack4(self, text_src, text_tgt):
        vision_result = self.attack_by_vision(text_src)
        flag, semantics_result = self.attack_by_semantics(text_src, text_tgt)

        if not flag:
            return flag, vision_result

        vision_similar = self.semantics_model.get_sentence_similarity([
            text_src, 
            vision_result
        ])
        semantics_similar = self.semantics_model.get_sentence_similarity([
            text_src, 
            semantics_result
        ])

        if vision_similar > semantics_similar:
            adversarial_example = vision_result
            flag = False
        else:
            adversarial_example = semantics_result

        return flag, adversarial_example

    # 只用视觉攻击
    def attack5(self, text_src, text_tgt):
        adversarial_example = self.attack_by_vision(text_src)
        return False, adversarial_example
    
    # 只用语义攻击，但有约束，不符合约束的用视觉替换
    def attack6(self, text_src, text_tgt):
        # 是否使用语义作为待替换样本
        semantics_flag = False

        # 设置原始样本为待替换样本
        source = text_src

        # 将参考译文翻译回来，得到语义相似的待替换样本
        text_tgt_translation = self.translate_tgt(text_tgt)

        # 审查语义相似的待替换样本是否满足语义相似条件
        semantic_similarity = self.get_sentence_semantic_similarity(text_tgt_translation, text_src)

        if text_tgt_translation != text_src and semantic_similarity > 0.9:
            source = text_tgt_translation
            semantics_flag = True

        # 用待替换样本进行视觉替换得到最终的对抗样本
        attack_res_semantics = self.search_samples_by_vision(source)

        return semantics_flag, attack_res_semantics
    
    # 只用语义攻击，但无约束
    def attack7(self, text_src, text_tgt):
        # 将参考译文翻译回来，得到语义相似的待替换样本
        text_tgt_translation = self.translate_tgt(text_tgt)

        source = text_tgt_translation
        semantics_flag = True

        # 用待替换样本进行视觉替换得到最终的对抗样本
        attack_res_semantics = self.search_samples_by_vision(source)

        return semantics_flag, attack_res_semantics
    
    # 从两种对抗样本中挑选一份翻译结果和参考翻译语义相似度最低的
    def attack_by_vision_search(self, text_src, text_tgt):
        flag, adversarial_example = self.attack_method(text_src, text_tgt)
        return flag, adversarial_example
    
    # 使用语义搜索进行攻击
    def attack_by_semantics_search(self, text):
        attack_res_semantics = self.search_samples_by_semantics(text)
        return None, attack_res_semantics
    
    def attack(self, text_src, text_tgt):
        if self.search_method == 'vision':
            flag, adversarial_example = self.attack_by_vision_search(text_src, text_tgt)
        else:
            flag, adversarial_example = self.attack_by_semantics_search(text_src)

        return flag, adversarial_example


if __name__ == '__main__':
    model_src_path = './model/opus-mt-zh-en'
    model_tgt_path = './model/opus-mt-en-zh'

    # 消融实验（组件消融：A单纯语义+B替换比例）
    percent='0.15'
    thresh=0.95
    sc='all'
    search_method='semantics'
    vision_constraint=False

    # # 消融实验（组件消融：A单纯语义+B整体视觉约束）
    # percent='0.15'
    # thresh=0.95
    # sc='all'
    # search_method='semantics'
    # vision_constraint=True

    # # 消融实验（组件消融：A语义&视觉+B替换比例）
    # percent='0.15'
    # thresh=0.95
    # sc='all'
    # search_method='vision'
    # vision_constraint=False

    # # 消融实验（组件消融：A语义&视觉+B整体视觉约束）
    # percent='0.15'
    # thresh=0.95
    # sc='all'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（超参消融：替换比例：1字符）
    # percent='1'
    # thresh=0.95
    # sc='all'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（超参消融：替换比例：2字符）
    # percent='2'
    # thresh=0.95
    # sc='all'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（超参消融：替换比例：10%）
    # percent='0.1'
    # thresh=0.95
    # sc='all'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（超参消融：替换比例：15%）
    # percent='0.15'
    # thresh=0.95
    # sc='all'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（超参消融：替换比例：20%）
    # percent='0.2'
    # thresh=0.95
    # sc='all'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（超参消融：整体感知相似约束：0.92）
    # percent='0.15'
    # thresh=0.92
    # sc='all'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（超参消融：整体感知相似约束：0.93）
    # percent='0.15'
    # thresh=0.93
    # sc='all'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（超参消融：整体感知相似约束：0.94）
    # percent='0.15'
    # thresh=0.94
    # sc='all'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（超参消融：整体感知相似约束：0.95）
    # percent='0.15'
    # thresh=0.95
    # sc='all'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（超参消融：整体感知相似约束：0.96）
    # percent='0.15'
    # thresh=0.96
    # sc='all'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（超参消融：整体感知相似约束：0.97）
    # percent='0.15'
    # thresh=0.97
    # sc='all'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（超参消融：整体感知相似约束：0.98）
    # percent='0.15'
    # thresh=0.98
    # sc='all'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（方法消融：单独radicals）
    # percent='0.15'
    # thresh=0.95
    # sc='all'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（方法消融：单独glyph）
    # percent='0.15'
    # thresh=0.95
    # sc='glyph'
    # search_method='vision'
    # vision_constraint=True

    # # 消融实验（方法消融：单独radicals）
    # percent='0.15'
    # thresh=0.95
    # sc='radicals'
    # search_method='vision'
    # vision_constraint=True

    device = torch.device('cuda:1')

    attacker = Attacker(
        model_src_path, model_tgt_path, device, method=1,
        percent=percent, thresh=thresh, sc=sc,
        search_method=search_method,
        vision_constraint=vision_constraint
    )

    text_src = '他们从未那样做过。'
    text_tgt = 'They really don\'t do that.'

    flag, attack_res = attacker.attack(text_src, text_tgt)
    print(flag, attack_res)

    translation_res = attacker.translate_src('他们从未这样做。')

    print(translation_res)