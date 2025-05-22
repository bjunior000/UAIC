import torch
import torch.nn as nn
from torch.nn import functional as F
import clip
from PIL import Image
import json
import os
import tqdm
import h5py

class CLIPUncertaintyEstimator(nn.Module):
    def __init__(self):
        super(CLIPUncertaintyEstimator, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        # 불확실성 예측을 위한 추가 레이어
        self.uncertainty_projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)
        
    def forward(self, image, text):
        # 이미지와 텍스트 인코딩
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
        
        # 정규화
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 유사도 계산
        similarity = (100.0 * image_features @ text_features.T)
        
        # 불확실성 계산 (1 - 유사도의 소프트맥스)
        confidence = F.softmax(similarity, dim=-1)
        uncertainty = 1 - confidence
        
        return uncertainty, similarity

def check_paths():
    """
    데이터 경로를 확인하고 필요한 파일/폴더의 존재 여부를 출력합니다.
    """
    data_path = 'data'
    img_features_path = os.path.join(data_path, 'coco_detections.hdf5')
    image_dir = os.path.join(data_path, 'train2014')
    caption_path = os.path.join(data_path, 'annotations', 'captions_train2014.json')
    
    print(f"특징 파일 존재 여부: {os.path.exists(img_features_path)}")
    print(f"이미지 폴더 존재 여부: {os.path.exists(image_dir)}")
    print(f"캡션 파일 존재 여부: {os.path.exists(caption_path)}")
    
    # 이미지 폴더가 없는 경우 안내
    if not os.path.exists(image_dir):
        print("원본 이미지 폴더가 없습니다. COCO 데이터셋을 다운로드하거나 특징 벡터 방식을 사용하세요.")
    
    return {
        'features_exist': os.path.exists(img_features_path),
        'images_exist': os.path.exists(image_dir),
        'captions_exist': os.path.exists(caption_path)
    }

def estimate_clip_uncertainty():
    """
    원본 이미지를 사용하여 CLIP 기반 불확실성 예측을 수행합니다.
    """
    data_path = 'data'
    output_path = os.path.join(data_path, 'clip_uncertainty_captions.json')
    
    # CLIP 모델 초기화
    uncertainty_model = CLIPUncertaintyEstimator()
    device = uncertainty_model.device
    
    # 캡션 데이터 로드
    caption_path = os.path.join(data_path, 'annotations')
    train_caption_path = os.path.join(caption_path, 'captions_train2014.json')
    
    with open(train_caption_path, 'r', encoding='utf-8') as j:
        captions = json.load(j)
    
    # 이미지 경로 세팅
    image_dir = os.path.join(data_path, 'train2014')
    
    annotation_list = []
    
    for i, sample in enumerate(tqdm.tqdm(captions['annotations'][:1000])):  # 예시로 1000개만 처리
        img_id = sample['image_id']
        caption = sample['caption']
        
        # 이미지 로드 및 전처리
        image_path = os.path.join(image_dir, f'COCO_train2014_{img_id:012d}.jpg')
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = uncertainty_model.preprocess(image).unsqueeze(0).to(device)
            
            # 텍스트 토큰화
            text_input = clip.tokenize([caption]).to(device)
            
            # 불확실성 계산
            uncertainty, similarity = uncertainty_model(image_input, text_input)
            
            # 결과 저장
            sample['clip_uncertainty'] = uncertainty.item()
            sample['clip_similarity'] = similarity.item()
            annotation_list.append(sample)
            
            if i % 100 == 0:
                print(f"Processed {i} samples")
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
    
    # 결과 저장
    new_captions = {}
    new_captions['annotations'] = annotation_list
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_captions, f, indent=4)
    
    print(f"CLIP 기반 불확실성 예측 완료. 결과 저장 경로: {output_path}")

def estimate_clip_uncertainty_with_features():
    """
    기존 Bag-of-Words가 사용하는 특징 벡터를 활용하여 CLIP 기반 불확실성 예측을 수행합니다.
    """
    data_path = 'data'
    output_path = os.path.join(data_path, 'clip_uncertainty_features_captions.json')
    
    # CLIP 모델 초기화
    uncertainty_model = CLIPUncertaintyEstimator()
    device = uncertainty_model.device
    
    # 캡션 데이터 로드
    caption_path = os.path.join(data_path, 'annotations')
    train_caption_path = os.path.join(caption_path, 'captions_train2014.json')
    
    with open(train_caption_path, 'r', encoding='utf-8') as j:
        captions = json.load(j)
    
    # Bag-of-Words와 동일한 특징 파일 사용
    img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'))
    
    annotation_list = []
    
    for i, sample in enumerate(tqdm.tqdm(captions['annotations'][:1000])):
        img_id = sample['image_id']
        caption = sample['caption']
        
        feature_id = str(img_id) + '_features'
        
        # 특징 벡터 직접 사용 (CLIP 인코더 우회)
        try:
            # 기존 특징 로드
            image_feature = torch.FloatTensor(img_features[feature_id]).to(device)
            
            # 텍스트 토큰화
            text_input = clip.tokenize([caption]).to(device)
            text_features = uncertainty_model.clip_model.encode_text(text_input)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 특징 벡터 형태 조정 및 정규화 (2048 -> 512)
            if image_feature.dim() > 2:
                image_feature = image_feature.mean(dim=1)  # 공간 차원 평균화
            
            # 차원 맞추기 위한 투영
            feature_dim = image_feature.size(-1)
            if not hasattr(uncertainty_model, 'feature_adapter'):
                uncertainty_model.feature_adapter = nn.Linear(feature_dim, 512).to(device)
            
            adapted_features = uncertainty_model.feature_adapter(image_feature)
            adapted_features = adapted_features / adapted_features.norm(dim=-1, keepdim=True)
            
            # 유사도 계산
            similarity = 100.0 * torch.matmul(adapted_features, text_features.T)
            
            # 불확실성 계산
            confidence = F.softmax(similarity, dim=-1)
            uncertainty = 1 - confidence
            
            # 결과 저장
            sample['clip_uncertainty'] = uncertainty.item()
            sample['clip_similarity'] = similarity.item()
            annotation_list.append(sample)
            
            if i % 100 == 0:
                print(f"Processed {i} samples")
        except Exception as e:
            print(f"Error processing image {img_id}: {e}")
            
    # 결과 저장
    new_captions = {}
    new_captions['annotations'] = annotation_list
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_captions, f, indent=4)
    
    print(f"특징 벡터 기반 CLIP 불확실성 예측 완료. 결과 저장 경로: {output_path}")

if __name__ == "__main__":
    # CLIP 모듈 설치 확인
    try:
        import clip
        print("CLIP 모듈이 이미 설치되어 있습니다.")
    except ImportError:
        print("CLIP 모듈이 설치되어 있지 않습니다.")
        print("다음 명령어로 설치하세요:")
        print("pip install git+https://github.com/openai/CLIP.git")
        print("pip install ftfy regex tqdm pillow")
        exit(1)
    
    # 데이터 경로 확인
    path_status = check_paths()
    
    # 실행 방법 선택
    if path_status['images_exist']:
        print("원본 이미지를 사용하여 불확실성을 측정합니다.")
        estimate_clip_uncertainty()
    elif path_status['features_exist']:
        print("특징 벡터를 사용하여 불확실성을 측정합니다.")
        estimate_clip_uncertainty_with_features()
    else:
        print("필요한 데이터 파일이 없습니다. 데이터 경로를 확인하세요.")
