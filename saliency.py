import torch
import numpy as np
from torch.nn import Softmax
from transformers import BertTokenizer, BertForMaskedLM


class saliency_model:
    def __init__(self, model_path, device):
        # 加载中文BERT模型
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForMaskedLM.from_pretrained(model_path)
        
        self.device = device
        self.model = self.model.to(self.device)

        self.model.eval()
        self.norm = Softmax(dim=1)

    def subword_split(self, words):
        # 对每个词利用当前模型进行进一步的子词划分
        tokenized_text = []
        indexDic = {}
        word_count = 0
        char_count = 0
        for word in words:
            charsList = self.tokenizer.tokenize(word)
            charIdxes = []
            for c in charsList:
                tokenized_text.append(c)
                charIdxes.append(char_count)
                char_count += 1

            indexDic[word_count] = charIdxes
            word_count += 1

        return tokenized_text, indexDic
    
    def mask_and_predict(self, tokenized_text, indexDic, word_idx):
        # mask目标单词
        _tokenized_text = tokenized_text.copy()
        maskCharIdxes = indexDic[word_idx]
        for masked_idx in maskCharIdxes:
            _tokenized_text[masked_idx] = '[MASK]'

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(_tokenized_text)
        segments_ids = [0 for i in range(len(_tokenized_text))]

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        tokens_tensor = tokens_tensor.to(self.device)
        segments_tensors = segments_tensors.to(self.device)

        # 对mask的所有位置进行预测
        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        return maskCharIdxes, predictions
    
    def calc_wordProb(self, tokenized_text, maskCharIdxes, predictions):
        # 若有多个子词组成一个单词，则需要将每个子词的位置概率进行相乘
        wordProb = 1
        for masked_idx in maskCharIdxes:
            confidence_scores = predictions[:, masked_idx, :]
            confidence_scores = self.norm(confidence_scores)

            masked_token = tokenized_text[masked_idx]
            masked_token_id = self.tokenizer.convert_tokens_to_ids([masked_token])[0]
            orig_prob = confidence_scores[0, masked_token_id].item()

            wordProb = wordProb*orig_prob

        return wordProb

    def scores(self, words):
        # 处理头部和尾部

        # 对每个词进行划分，一个词会划分出多个子词
        tokenized_text, indexDic = self.subword_split(words)

        saliency = np.zeros(len(words), dtype=float)

        # 由于存在子词处理，故在mask时，需要将一个词划分出的所有子词同时mask
        for i in range(len(words)):
            # mask目标单词，并对mask的所有位置进行预测
            maskCharIdxes, predictions = self.mask_and_predict(tokenized_text, indexDic, i)

            # 读取预测结果，得到每个被mask的位置里原单词出现的概率
            wordProb = self.calc_wordProb(tokenized_text, maskCharIdxes, predictions)

            # 重要性即为1-wordProb
            saliency[i] = 1 - wordProb

        return saliency


if __name__ == '__main__':
    model_saliency_path = './model/chinese-bert-wwm-ext'

    device = torch.device('cuda:1')

    saliencyModel = saliency_model(model_saliency_path, device)

    words = ['白天', '天气', '很好',]
    saliency = saliencyModel.scores(words)

    print(saliency)
