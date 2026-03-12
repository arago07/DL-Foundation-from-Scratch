import pandas as pd
import os

# 1. CSV 파일 경로 설정 (네 환경에 맞게 경로를 꼭 수정해줘!)
# 예: csv_path = './data/HAM10000/HAM10000_metadata.csv'
csv_path = './data/HAM10000/HAM10000_metadata.csv' 

# 2. pandas로 CSV 읽어오기
df = pd.read_csv(csv_path)

# 3. 데이터 구조 확인
print(f"전체 데이터 개수: {len(df)}개")
print("\n[데이터 첫 5줄 미리보기]")
print(df.head())

# 4. 정답 라벨(dx 열) 분포 확인 및 숫자로 변환할 딕셔너리 생성
unique_labels = df['dx'].unique() # 중복 없는 고유한 병명들만 추출
print(f"\n고유한 병명(클래스) 목록: {unique_labels}")

# 문자열을 정수 인덱스로 매핑 (예: 'nv': 0, 'mel': 1 ...)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
print(f"\n라벨 매핑 딕셔너리: {label_map}")

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

class HAM10000Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # 1. CSV 파일 읽기
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # 2. 문자로 된 병명을 숫자로 바꾸는 딕셔너리 자동 생성
        unique_labels = self.df['dx'].unique()
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
    def __len__(self):
        # 전체 데이터 개수 반환
        return len(self.df)
    
    def __getitem__(self, idx):
        # 1. 파일 이름 가져와서 경로 완성하기 (.jpg 붙이기)
        img_name = self.df.iloc[idx]['image_id'] + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        
        # 2. 이미지 열기 (컬러 이미지이므로 RGB 확인)
        image = Image.open(img_path).convert('RGB')
        
        # 3. 정답 라벨을 숫자로 변환
        label_str = self.df.iloc[idx]['dx']
        label_idx = self.label_map[label_str]
        
        # 4. 전처리(transform) 적용
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label_idx)

# ==========================================
# 🔍 Sanity Check (내가 짠 클래스가 잘 작동하는지 테스트)
# ==========================================

# 텐서로만 바꿔주는 가장 기본적인 전처리
basic_transform = transforms.Compose([
    transforms.Resize((224, 224)), # ResNet 입력 사이즈로 맞춤
    transforms.ToTensor()
])

# 네 환경에 맞게 경로 수정 필수! (이미지들이 모여있는 폴더 경로)
csv_path = './data/HAM10000/HAM10000_metadata.csv' 
img_path = './data/HAM10000/all_images' # 이미지를 한 곳에 모은 폴더

# 데이터셋 인스턴스 생성
dataset = HAM10000Dataset(csv_file=csv_path, img_dir=img_path, transform=basic_transform)

# 0번째 데이터 하나만 쏙 뽑아보기
sample_img, sample_label = dataset[0]

print("-" * 30)
print("Sanity Check 결과")
print("-" * 30)
print(f"이미지 텐서 모양: {sample_img.shape}")
print(f"정답 라벨 (숫자): {sample_label.item()}")
