"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import re
import gc

from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from processers import BlipDiffusionInputImageProcessor, BlipDiffusionTargetImageProcessor, text_proceesser
from lavis.models import load_model_and_preprocess
import random
import itertools
from string import ascii_lowercase
import cv2
import numpy as np
import matplotlib.pyplot as plt


class SubjectDrivenTextToImageDataset(Dataset):
    def __init__(
        self,
        image_dir,
        subject_text,
        text_prompt,
        questions,
        repetition=1,
    ):
        self.subject = []
        for subject in subject_text:
            # print(subject)
            self.subject.append(text_proceesser(subject.lower()))
        self.text_prompt = text_prompt
        self.image_dir = image_dir

        self.inp_image_transform = BlipDiffusionInputImageProcessor()
        self.tgt_image_transform = BlipDiffusionTargetImageProcessor()

        
        # # make absolute path
        # self.image_paths = [os.path.abspath(imp) for imp in image_paths]
        
        self.image_pair_paths = get_image_pair_path(image_dir=image_dir, group_count=len(subject_text)) # [ (subject_A, subject_B, ..., mask_subject_A, mask_subject_B, ..., background), ...]
        # self.images=[] 
                
        self.repetition = repetition

        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        
        # Set VQA and get the answer from the questions
        self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=self.device)
        self.questions = questions
        print("2.",torch.cuda.memory_allocated())  # 獲取當前佔用的內存量（以字節為單位）
        print(torch.cuda.memory_reserved())   # 獲取當前保留的內存量

        # Extract VQA features for all images during initialization
        self.vqa_features = self._extract_all_vqa_features()
        # 清除模型和處理器所佔用的 GPU 資源
        del self.model
        del self.vis_processors

        # 強制垃圾回收
        gc.collect()

        # 清空未使用的 GPU 緩存
        torch.cuda.empty_cache()


    def __len__(self):
        return len(self.image_pair_paths) * self.repetition
    
    @property
    def len_without_repeat(self):
        return len(self.image_pair_paths)

    def __getitem__(self, index):
        image_pair_path = self.image_pair_paths[index % len(self.image_pair_paths)]

        modifier_token = re.findall(r'<.*?>', self.text_prompt)

        # Get precomputed VQA features
        vqa = self.vqa_features[index % len(self.image_pair_paths)]

        subjects_image = {}
        for i, subject in enumerate(self.subject):
            subjects_image[subject]= self.inp_image_transform(Image.open(image_pair_path[i]).convert("RGB"))
        subjects_image['background'] = self.inp_image_transform(Image.open(image_pair_path[-1]).convert("RGB"))
            
        target_image, _, _ = composite_images(background_path=image_pair_path[-1], foreground_paths=image_pair_path[0:len(self.subject)], mask_paths=image_pair_path[len(self.subject):-1] )
        target_image = self.tgt_image_transform(target_image)

        # Get the subjects place in the text prompt
        # Ex:
        # if    text_prompt='A dog and a cat on the beach'
        #       subject=['dog', 'cat']
        # then  subjects_position=[1,4]
        subjects_position = []
        for subject in self.subject:
            # 找到所有符合 subject 的位置
            start_idx = 0
            while start_idx < len(self.text_prompt):
                start_idx = self.text_prompt.find(subject, start_idx)  # 查找 subject 在 text_prompt 的位置
                if start_idx == -1:
                    break
                # 計算 subject 後面第一個單詞的索引
                end_idx = start_idx + len(subject)  # 計算 subject 結束的位置
                after_text = self.text_prompt[end_idx:].strip()  # 取 subject 後面的文本
                word_idx = len(self.text_prompt[:end_idx].split())  # 計算 subject 結束後的索引位置
                if after_text:  # 確保有後續文本存在
                    subjects_position.append((subject, word_idx))
                start_idx += len(subject)  # 繼續查找下一個匹配



        # TODO: 多主題的圖片處理
        return {
            "inp_images": subjects_image, # list of all subject in torch.Tensor
            "tgt_image": target_image, # torch.Tensor
            "subject_text": self.subject, # List[str] eg. ['dog', 'cat']
            "caption": self.text_prompt, # str
            "vqa_token": vqa, # dict eg. {}
            "modifier_token": modifier_token,
            "subjects_position": subjects_position, # List[tuple] eg. [('dog', 1), ('cat', 2)]
        }

    def _extract_all_vqa_features(self):
        vqa_features = {}
        for index, images_path in enumerate(self.image_pair_paths):
            ans = {}
            for subject_i in range(len(self.subject)):
                image = Image.open(images_path[subject_i]).convert("RGB")
                ans.update( {'p_'+chr(subject_i+ord('A')): self.extract_vqa_features(image=image, question=self.questions['<p>'])} )
            ans.update( {'b': self.extract_vqa_features(image=image, question=self.questions['<b>'])} )
            
            vqa_features[index] = ans
        return vqa_features

    def extract_vqa_features(self, image, question): #questions: {"<b>": what is the background, ...}
        # we associate a model with its preprocessors to make it easier for inference.
        
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        answer = {}

        answer = self.model.generate({"image": image, "prompt": f"Question: {question}? Answer:"})
        
        return answer


def collate_fn(samples):
    samples = [s for s in samples if s is not None]
    # Check if samples is empty after filtering
    if not samples:
        return {}
    collated_dict = {}
    keys = samples[0].keys() # Use the keys of the first sample as a reference
    for k in keys:
        values = [sample[k] for sample in samples]
        # If the value type for the key is torch.Tensor, stack them else return list
        collated_dict[k] = torch.stack(values, dim=0) if isinstance(values[0], torch.Tensor) else values
    return collated_dict


def get_image_pair_path(image_dir, group_count=2, num_select=5, num_range=(0, 4), valid_exts=None):
    """
    從 image_dir 目錄中讀取符合條件的圖片，這些圖片的檔名格式為
      <groupLetter><兩位數字> (例如 a04.jpg)
    並且對應的遮罩檔案位於同一目錄中，其檔名為 "mask_" 加上原檔名 (例如 mask_a04.jpg)。
    
    此外，image_dir 中必須有一個名為 "bg" 的子資料夾，用於存放背景圖片。
    
    會將每個群組（例如 a、b …）的圖片及對應的遮罩做成 pair，
    利用 Cartesian product 生成所有群組間的組合，再隨機選取 num_select 組。
    
    回傳的 tuple 結構為：
      ( group1_image, group2_image, ..., group1_mask, group2_mask, ..., background_image )
    
    參數:
      image_dir (str): 圖片所在的目錄路徑（此目錄下必須有名為 "bg" 的子資料夾）。
      group_count (int): 要處理的群組數量。例如 group_count=2 表示處理 a 與 b 群組，
                         group_count=3 則處理 a, b, c 群組。
      num_select (int): 從所有可能的組合中隨機選取的組合數量。
      num_range (tuple): 檔名中數字的有效範圍 (min, max)，預設為 (0, 4)。
      valid_exts (set 或 list): 有效的檔案副檔名（不分大小寫），預設為 {"jpg", "jpeg", "png", "webp"}。
    
    回傳:
      final_combinations (list): 每個元素是一個 tuple，依序包含各群組的圖片路徑、
                                  接著各群組的遮罩路徑，最後再加上從 bg 資料夾中隨機選取的背景圖片路徑。
    """
    # 設定預設有效副檔名
    if valid_exts is None:
        valid_exts = {"jpg", "jpeg", "png", "webp"}
    else:
        valid_exts = {ext.lower() for ext in valid_exts}
    
    # 取得前 group_count 個字母 (例如 group_count=2 → ['a', 'b'])
    group_letters = ascii_lowercase[:group_count]
    
    # 建立字典，key 為群組字母，value 為該群組中符合條件的圖片 pair 列表，
    # 每個 pair 格式為 (image_path, mask_path)
    group_images = {letter: [] for letter in group_letters}
    
    # 遍歷 image_dir 下的檔案，排除名為 "bg" 的子資料夾
    for filename in os.listdir(image_dir):
        full_path = os.path.join(image_dir, filename)
        if os.path.isdir(full_path) and filename.lower() == 'bg':
            continue
        if os.path.isfile(full_path):
            base, ext = os.path.splitext(filename)
            ext = ext.lower()[1:]  # 去除 "." 並轉小寫
            if ext in valid_exts and len(base) >= 3:
                first_letter = base[0].lower()
                if first_letter in group_images:
                    try:
                        number = int(base[1:])
                        if num_range[0] <= number <= num_range[1]:
                            # 根據原圖片檔名產生遮罩檔名，例如 a04.jpg → mask_a04.jpg
                            mask_filename = "mask_" + filename
                            mask_path = os.path.join(image_dir, mask_filename)
                            if not os.path.isfile(mask_path):
                                continue
                            group_images[first_letter].append((full_path, mask_path))
                    except ValueError:
                        continue
    
    # 檢查各群組是否至少有一個有效的圖片 pair
    for letter in group_letters:
        if not group_images[letter]:
            raise ValueError(f"群組 '{letter}' 中未找到符合條件的圖片與遮罩 pair。")
    
    # 利用 itertools.product 生成各群組間所有可能的組合
    all_combinations = list(itertools.product(*(group_images[letter] for letter in group_letters)))
    
    # 從所有組合中隨機選取 num_select 組
    if len(all_combinations) < num_select:
        selected_combinations = all_combinations
    else:
        selected_combinations = random.sample(all_combinations, num_select)
    
    # 取得背景圖片：必須存在 image_dir/bg 這個子資料夾中
    bg_dir = os.path.join(image_dir, 'bg')
    if not os.path.isdir(bg_dir):
        raise ValueError("在指定的 image_dir 中找不到名為 'bg' 的資料夾。")
    
    bg_images = []
    for filename in os.listdir(bg_dir):
        full_path = os.path.join(bg_dir, filename)
        if os.path.isfile(full_path):
            base, ext = os.path.splitext(filename)
            ext = ext.lower()[1:]
            if ext in valid_exts:
                bg_images.append(full_path)
    if not bg_images:
        raise ValueError("在 'bg' 資料夾中未找到有效的背景圖片。")
    
    # 重新組合每個選取的組合
    # 希望順序為：
    # ( group1_image, group2_image, ..., group1_mask, group2_mask, ..., background_image )
    final_combinations = []
    for combo in selected_combinations:
        # 取得各群組的圖片路徑列表和遮罩路徑列表
        image_paths = [pair[0] for pair in combo]
        mask_paths  = [pair[1] for pair in combo]
        # 隨機選取一張背景圖片
        random_bg = random.choice(bg_images)
        # 拼接成最終的 tuple
        final_tuple = tuple(image_paths + mask_paths + [random_bg])
        final_combinations.append(final_tuple)
    
    return final_combinations



def composite_images(background_path, foreground_paths, mask_paths, output_size=(512, 512)):
    """
    將背景圖與任意數目的前景圖（及其遮罩）進行合成，前景圖會隨機排列於背景圖上，
    並利用 alpha blending 進行融合（前景圖會依比例縮放、以隨機位置排列且避免過度重疊）。
    
    輸出:
      - combined_image: PIL Image 格式 (RGB)
      - individual_masks: 各前景圖在背景上位置的遮罩 (numpy array, 灰階)
      - mask_background: 整體背景遮罩 (numpy array, 灰階)，背景部分為白色，前景區域為黑色。
      
    參數:
      background_path (str): 背景圖的檔案路徑。
      foreground_paths (list of str): 前景圖檔案路徑的列表。
      mask_paths (list of str): 對應前景遮罩圖（灰階）的檔案路徑列表，順序需與 foreground_paths 一致。
      output_size (tuple): 合成圖像的尺寸 (width, height)，預設 (512, 512)。
    """
    def resize_foreground(foreground, mask, target_width, target_height):
        # 根據背景尺寸計算縮放比例，並乘上一個縮放因子 (0.3) 以避免前景過大
        scale_height = target_height / foreground.shape[0]
        scale_width = target_width / foreground.shape[1]
        scale = min(scale_height, scale_width)
        new_width = int(foreground.shape[1] * scale * 0.3)
        new_height = int(foreground.shape[0] * scale * 0.3)
        resized_foreground = cv2.resize(foreground, (new_width, new_height))
        resized_mask = cv2.resize(mask, (new_width, new_height))
        return resized_foreground, resized_mask

    def get_random_position(bg_width, bg_height, fg_width, fg_height, existing_positions):
        max_attempts = 300
        for _ in range(max_attempts):
            x = random.randint(0, bg_width - fg_width)
            y = random.randint(0, bg_height - fg_height)
            overlap = False
            for (ex_x, ex_y, ex_w, ex_h) in existing_positions:
                overlap_x1 = max(x, ex_x)
                overlap_y1 = max(y, ex_y)
                overlap_x2 = min(x + fg_width, ex_x + ex_w)
                overlap_y2 = min(y + fg_height, ex_y + ex_h)
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    fg_area = fg_width * fg_height
                    if overlap_area > fg_area / 20:
                        overlap = True
                        break
            if not overlap:
                return x, y
        return x, y

    def apply_alpha_blending(background, foreground, mask, x, y):
        fg_height, fg_width = foreground.shape[:2]
        bg_region = background[y:y+fg_height, x:x+fg_width]
        alpha = mask.astype(float) / 255.0
        if len(foreground.shape) == 3:
            alpha = alpha[:, :, None]
        blended = (alpha * foreground + (1 - alpha) * bg_region).astype(np.uint8)
        background[y:y+fg_height, x:x+fg_width] = blended

    # 讀取背景圖並調整至指定尺寸 (注意：OpenCV 讀取結果為 BGR 格式)
    background = cv2.imread(background_path)
    bg_width, bg_height = output_size
    combined_image = cv2.resize(background, (bg_width, bg_height))
    
    existing_positions = []  # 用於記錄前景圖的放置位置
    individual_masks = []    # 每個前景圖在背景上對應的遮罩
    composite_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)  # 整體遮罩，初始皆為 0
    
    # 處理不定數目的前景圖
    for fg_path, mask_path in zip(foreground_paths, mask_paths):
        fg = cv2.imread(fg_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        fg_resized, mask_resized = resize_foreground(fg, mask, bg_width, bg_height)
        fg_h, fg_w = fg_resized.shape[:2]
        x, y = get_random_position(bg_width, bg_height, fg_w, fg_h, existing_positions)
        existing_positions.append((x, y, fg_w, fg_h))
        
        # 建立單一前景放置位置的遮罩
        foreground_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
        foreground_mask[y:y+fg_h, x:x+fg_w] = mask_resized
        individual_masks.append(foreground_mask)
        
        apply_alpha_blending(combined_image, fg_resized, mask_resized, x, y)
        composite_mask = cv2.bitwise_or(composite_mask, foreground_mask)
    
    # 製作背景遮罩：背景部分為白色，前景部分為黑色
    mask_background = cv2.bitwise_not(composite_mask)
    
    # 將合成後的圖像從 OpenCV 的 BGR 轉為 RGB，再轉成 PIL Image 格式
    combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
    pil_combined_image = Image.fromarray(combined_image_rgb).convert("RGB")
    
    return pil_combined_image, individual_masks, mask_background
