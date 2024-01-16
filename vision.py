import faiss
import pickle
import torch
import lpips
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
from utils.chinese_stoke.glyph_zh import glyph_similar
from skimage.metrics import structural_similarity as ssim


class vision_model(object):
    def __init__(self):
        # LPIPS计算
        self.loss_fn = lpips.LPIPS(net='alex')
        self.gpu = False


    # 读取中文字符
    def read_characters_from_file(self, filename):
        characters = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                characters.extend(line)
        return characters


    # 字符或字符串转图片
    def simVec(self, text):
        img = Image.new("RGB", (40, 40), (255, 255, 255))
        font = ImageFont.truetype('./utils/chinese_fonts/simsun.ttc', 18)
        draw = ImageDraw.Draw(img)
        draw.text((10, 5), text, font=font, fill="#000000")
        arr = np.array(img.convert('L'))
        return arr


    # 使用MSE计算视觉相似度
    def get_mse_similarity(self, text1, text2):
        arr1 = self.simVec(text1)
        arr2 = self.simVec(text2)
        
        similarity_mse = -np.mean((arr1 - arr2) ** 2)
        
        return similarity_mse


    # 使用SSIM计算视觉相似度
    def get_ssim_similarity(self, text1, text2):
        arr1 = self.simVec(text1)
        arr2 = self.simVec(text2)
        
        similarity_ssim = ssim(arr1, arr2, multichannel=True)
        
        return similarity_ssim


    # 使用LPIPS计算视觉相似度
    def get_lpips_similarity(self, text1, text2):
        arr1 = self.simVec(text1 + ' ' + text2)
        arr2 = self.simVec(text2 + ' ' + text1)

        similarity_lpips = 1 - self.loss_fn(torch.tensor(arr1), torch.tensor(arr2)).item()
        similarity_lpips = 0 if similarity_lpips < 0.9 else (similarity_lpips - 0.9) * 10

        return similarity_lpips


    # 使用glyph计算视觉相似度
    def get_glyph_similarity(self, text1, text2):
        similarity_glyph = glyph_similar(text1 + ' ' + text2, text2 + ' ' + text1)

        return similarity_glyph


    # 自定义视觉相似度计算
    def get_my_similarity(self, text1, text2):
        similarity_lpips = self.get_lpips_similarity(text1, text2)

        similarity_glyph = self.get_glyph_similarity(text1, text2)

        similarity = 0.8 * similarity_lpips + 0.2 * similarity_glyph

        return similarity


    # 计算top10相似字，先用余弦相似度挑选top100，再用自定义的相似度函数挑选top10
    def calculate_top_similar_chars(self, simvecs, similar_function):
        top_similar_chars = {}

        # 将simvecs中的相似度特征存储为一个三维的张量
        simvecs_matrix = np.array(list(simvecs.values()), dtype=np.float32)
        
        # 将三维张量展平为二维张量，即将每一张图片的像素矩阵转化成向量
        simvecs_matrix = simvecs_matrix.reshape(simvecs_matrix.shape[0], -1)
        faiss.normalize_L2(simvecs_matrix)

        # 创建Faiss索引
        index = faiss.IndexFlatIP(simvecs_matrix.shape[1])

        # 使用gpu
        if self.gpu:
            gpu_resource = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)

        # 将特征矩阵添加到Faiss索引中
        index.add(simvecs_matrix)
        
        # 计算相似度和获取top10相似字符
        pbar = tqdm(total=len(simvecs), desc='get top10 similar characters')
        for char, sim_array in simvecs.items():
            # 将查询特征转换为Faiss可接受的格式，即将每一张图片的像素矩阵转化成向量
            query = np.array([sim_array.reshape(1600)], dtype=np.float32)
            faiss.normalize_L2(query)

            # 使用Faiss进行相似度搜索
            # 获取前101个最相似的字符（包括自身）
            _, indices = index.search(query, k=101)
            top_similar = indices[0][1:]  # 去除自身
            top_similar_char_score = [(j, similar_function(list(simvecs.keys())[j], char)) for j in top_similar]

            # 按相似度分数倒序排列，取前十个j 
            top_similar_char_score.sort(key=lambda x: x[1], reverse=True)
            top_similar_chars[char] = [(list(simvecs.keys())[j], score) for j, score in top_similar_char_score[:10]]

            pbar.update(1)

        return top_similar_chars


if __name__ == '__main__':
    # 创建视觉模型
    model = vision_model()

    # 读取中文字符
    filename = './utils/chinese_characters/unicode_characters.txt'
    characters = model.read_characters_from_file(filename)

    # 将字符转为图片，然后转化为一个矩阵
    charVec = {}
    for char in characters:
        charVec[char] = model.simVec(char)

    # 计算字符相似度
    top_similar_chars = model.calculate_top_similar_chars(charVec, model.get_glyph_similarity)

    # 保存结果
    filename = './utils/chinese_characters/vision_similar_char.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(top_similar_chars, file)

