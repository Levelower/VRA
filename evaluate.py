import re
import time
import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction
from PIL import Image, ImageFont, ImageDraw
from skimage.metrics import structural_similarity as ssim


smooth_func = SmoothingFunction()


def calc_bleu(candidate, reference):
    # 计算一个文本列表的平均BLEU
    if type(candidate) is str:
        candidate = [candidate]
    if type(reference) is str:
        reference = [reference]

    bleu_scores = []

    for cand, ref in zip(candidate, reference):
        try:
            # 将句子转换为字符串列表
            candidate_list = nltk.word_tokenize(cand)
            reference_list = nltk.word_tokenize(ref)

            # 将参考翻译结果转换为一个列表的列表
            references = [reference_list]

            # 计算BLEU指标
            bleu_score = nltk.translate.bleu_score.sentence_bleu(
                references, candidate_list, 
                smoothing_function=smooth_func.method1
            )

            bleu_scores.append(bleu_score)

        except Exception as e:
            bleu_scores.append(0)

    # 计算平均BLEU指标
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu_score


def calc_bleu_asr(source, candidate, reference):
    if type(source) is str:
        source = [source]
    if type(candidate) is str:
        candidate = [candidate]
    if type(reference) is str:
        reference = [reference]

    bleu_scores_atk = []
    bleu_scores_ori = []

    all = 0
    suc = 0

    for ori, cand, ref in zip(source, candidate, reference):
        try:
            source_list = nltk.word_tokenize(ori)
            candidate_list = nltk.word_tokenize(cand)
            reference_list = nltk.word_tokenize(ref)

            references = [reference_list]

            bleu_score_atk = nltk.translate.bleu_score.sentence_bleu(
                references, candidate_list, 
                smoothing_function=smooth_func.method1
            )

            bleu_score_ori = nltk.translate.bleu_score.sentence_bleu(
                references, source_list, 
                smoothing_function=smooth_func.method1
            )

            bleu_scores_atk.append(bleu_score_atk)
            bleu_scores_ori.append(bleu_score_ori)

            if bleu_score_ori > 0 and (bleu_score_ori - bleu_score_atk) / bleu_score_ori > 0.5:
                suc += 1

        except Exception as e:
            bleu_scores_atk.append(0)
            bleu_scores_ori.append(0)

        all += 1

    avg_bleu_score_atk = sum(bleu_scores_atk) / len(bleu_scores_atk)
    avg_bleu_score_ori = sum(bleu_scores_ori) / len(bleu_scores_ori)
    asr = suc / all
    return avg_bleu_score_atk, avg_bleu_score_ori, asr


def calc_success_num(source, candidate, reference):
    if type(source) is str:
        source = [source]
    if type(candidate) is str:
        candidate = [candidate]
    if type(reference) is str:
        reference = [reference]

    bleu_scores_atk = []
    bleu_scores_ori = []

    all = 0
    suc = 0

    for ori, cand, ref in zip(source, candidate, reference):
        try:
            source_list = nltk.word_tokenize(ori)
            candidate_list = nltk.word_tokenize(cand)
            reference_list = nltk.word_tokenize(ref)

            references = [reference_list]

            bleu_score_atk = nltk.translate.bleu_score.sentence_bleu(
                references, candidate_list, 
                smoothing_function=smooth_func.method1
            )

            bleu_score_ori = nltk.translate.bleu_score.sentence_bleu(
                references, source_list, 
                smoothing_function=smooth_func.method1
            )

            bleu_scores_atk.append(bleu_score_atk)
            bleu_scores_ori.append(bleu_score_ori)

            if bleu_score_ori > 0 and (bleu_score_ori - bleu_score_atk) / bleu_score_ori > 0.5:
                suc += 1

        except Exception as e:
            bleu_scores_atk.append(0)
            bleu_scores_ori.append(0)

        all += 1

    avg_bleu_score_atk = sum(bleu_scores_atk) / len(bleu_scores_atk)
    avg_bleu_score_ori = sum(bleu_scores_ori) / len(bleu_scores_ori)
    asr = suc / all
    return all, suc, asr


def get_similarity(text1, text2):
    # 字体
    font = ImageFont.truetype('./utils/chinese_fonts/simsun.ttc', 18)

    # 得到文本的尺寸
    _, _, w1, _ = font.getbbox(text1)
    _, _, w2, _ = font.getbbox(text2)

    # 计算文本尺寸
    margin = 20
    im_w = w1 + w2 + margin

    # 固定高度，按指定宽度创建图像
    im_h = 50
    im1 = Image.new("RGB", (im_w, im_h), (255, 255, 255))
    im2 = Image.new("RGB", (im_w, im_h), (255, 255, 255))
    dr1 = ImageDraw.Draw(im1)
    dr2 = ImageDraw.Draw(im2)

    # 将文本按指定间隔绘制在图像上
    offset = margin // 2
    dr1.text((offset, offset), text1 + " " + text2, font=font, fill="#000000")
    dr2.text((offset, offset), text2 + " " + text1, font=font, fill="#000000")
    arr1 = np.array(im1.convert('L'))
    arr2 = np.array(im2.convert('L'))

    # 计算SSIM
    similarity = ssim(arr1, arr2, multichannel=True)

    return similarity


def calc_ssim(candidate, reference):
    # 计算一个文本列表的平均BLEU
    if type(candidate) is str:
        candidate = [candidate]
    if type(reference) is str:
        reference = [reference]

    ssim_score = 0
    cnt = 0
    for i in range(len(candidate)):
        s_t = get_similarity(
            re.sub(r'[^\w\s]', '', candidate[i].replace(' ','')), 
            re.sub(r'[^\w\s]', '', reference[i].replace(' ',''))
        )
        ssim_score += s_t
        cnt += 1
    return ssim_score / cnt


def get_data(in_path):
    data = pd.read_csv(in_path, sep='\t')
    return data


def print_by_vision(datasets, methods, base='merge_3_0.2_0.95_all_vision_True', language='chinese', src_l='zh', tgt_l='en'):
    for dataset in datasets:
        ori_data = get_data(f'./result/{language}/{dataset}/{dataset}_{base}.csv')
        source = ori_data['ori_translate']

        for method in methods:
            start_time = time.time()

            print(f'\n-------------------------{dataset} {method}---------------------------', flush=True)

            data_eval = get_data(f'./result/{language}/{dataset}/{dataset}_{method}.csv')

            candidate = data_eval['atk_translate']
            reference = data_eval[tgt_l]

            refs = []
            sys = []
            ori = []

            for src, cand, ref in zip(source, candidate, reference):
                references = [ref]
                refs.append(references)
                sys.append(cand)
                ori.append(src)

            ssim_score = calc_ssim(data_eval['atk_content'], data_eval[src_l])

            bleu_atk, bleu_ori, asr = calc_bleu_asr(source, candidate, reference)

            end_time = time.time()

            print('SSIM:\t\t', str(ssim_score), flush=True)
            print('BLEU:\t\t', str(bleu_ori) + '\t->\t' + str(bleu_atk) + '\tdown: ' + str((bleu_ori-bleu_atk)/bleu_ori), flush=True)
            print('ASR:\t\t', str(asr), flush=True)
            print('Time:\t\t', str(end_time - start_time), flush=True)


if __name__ =='__main__':
    print('\n-------------------------主实验 (wmt19、wmt18、ted) ---------------------------', flush=True)
    datasets = ['wmt19', 'wmt18', 'ted']
    methods = [ 
        'merge_3_0.2_0.95_all_vision_True',
    ]
    print_by_vision(datasets, methods)

    print('\n-------------------------主实验 (aspec) ---------------------------', flush=True)
    datasets = ['aspec']
    methods = [ 
        'merge_3_0.2_0.95_glyph_vision_True',
    ]
    print_by_vision(datasets, methods, base='merge_3_0.2_0.95_glyph_vision_True', language='japanese', src_l='ja')

    print('\n-------------------------消融实验 (超参percent) ---------------------------', flush=True)
    datasets = ['wmt19']
    methods = [
        'merge_3_1_0.95_all_vision_True', 
        'merge_3_2_0.95_all_vision_True',
        'merge_3_0.1_0.95_all_vision_True', 
        'merge_3_0.15_0.95_all_vision_True',
        'merge_3_0.2_0.95_all_vision_True', 
        'merge_3_0.25_0.95_all_vision_True', 
    ]
    print_by_vision(datasets, methods)


    print('\n-------------------------消融实验 (超参thresh) ---------------------------', flush=True)
    datasets = ['wmt19']
    methods = [
        'merge_3_0.2_0.93_all_vision_True', 
        'merge_3_0.2_0.94_all_vision_True', 
        'merge_3_0.2_0.95_all_vision_True', 
        'merge_3_0.2_0.96_all_vision_True', 
        'merge_3_0.2_0.97_all_vision_True', 
    ]
    print_by_vision(datasets, methods)


    print('\n-------------------------消融实验 (方法vision similar char) ---------------------------', flush=True)
    datasets = ['wmt19']
    methods = [
        'merge_3_0.2_0.95_all_vision_True', 
        'merge_3_0.2_0.95_glyph_vision_True', 
        'merge_3_0.2_0.95_radicals_vision_True', 
    ]
    print_by_vision(datasets, methods)

    
    print('\n-------------------------消融实验 (组件vision semantics) ---------------------------', flush=True)
    datasets = ['wmt19']
    methods = [
        'merge_3_0.2_0.95_all_vision_True', 
        'merge_3_0.2_0.95_all_vision_False',
        'merge_3_0.2_0.95_all_semantics_True',
        'merge_3_0.2_0.95_all_semantics_False',
    ]
    print_by_vision(datasets, methods)