# CLIP 없이 사용할 수 있는 BOW 구현
from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel 
# 더 이상 transformers에서 직접 import할 수 없음
from torch.optim import AdamW  
import torch.nn as nn
import torch 
from dataset import BagWordsDataset 
import torch.optim 
from utilis import AverageMeter
import os 
import json 
import h5py 
from torch.nn.functional import softmax 


# Uncertainty-aware image-conditioned bag-of-words
class BagofWords(BertPreTrainedModel):
    def __init__(self, config):
        super(BagofWords, self).__init__(config)
        self.feature_embd = nn.Linear(2048, config.hidden_size)
        self.transformer = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        self.init_weights()

    def acc_compute(self, output, labels):
        output = output * labels
        thres = torch.tensor([[0.3]]).expand_as(output)
        summ = torch.ge(output, thres).sum().item()
        total = labels.sum().item()
        return summ/total

    
    def forward(self, img_embs, labels=None):
        img_embs = self.feature_embd(img_embs)
        transformer_outputs = self.transformer(inputs_embeds=img_embs)
        hidden_states = transformer_outputs[1]
        pool_outputs = self.dropout(hidden_states)
        pool_outputs = self.classifier(pool_outputs)

        if labels is None:
            return pool_outputs

        criterion = nn.MultiLabelSoftMarginLoss()
        loss = criterion(pool_outputs, labels)
        acc = self.acc_compute(pool_outputs, labels)

        return loss, acc


# train the image conditioned bag-of-words 
def train():
    print("CLIP 없이 BOW 모델 학습을 시작합니다...")
    epochs = 25 
    model_path = 'model'
    gradient_accumlation_steps = 5 
    
    # 디렉토리가 없으면 생성
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print(f"Created directory: {model_path}")

    # vocab.txt 파일 확인
    vocab_path = 'data/vocab.txt'
    if not os.path.exists(vocab_path):
        # ckpt 폴더에서 복사
        import shutil
        if os.path.exists('ckpt/vocab.txt'):
            if not os.path.exists('data'):
                os.makedirs('data')
            shutil.copy('ckpt/vocab.txt', vocab_path)
            print(f"Copied vocab.txt from ckpt to data folder")
        else:
            print(f"Error: {vocab_path} not found!")
            return

    tokenizer = BertTokenizer(vocab_path) 
    
    # [UNK] 토큰 처리
    unk_id = tokenizer.vocab.get('[UNK]')
    if unk_id is None:
        print("Warning: [UNK] token not found in vocabulary, using default value.")
        unk_id = 10876  # README에서 확인된 값
        
    configuration = BertConfig(vocab_size=unk_id + 1, 
                                num_hidden_layers=3, 
                                intermediate_size=2048)
    model = BagofWords(configuration)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    dataset = BagWordsDataset('data', tokenizer)

    model.train()
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        avg_loss = AverageMeter()
        avg_acc = AverageMeter()
        iteration = 1 
        for img, label in dataset:
            img = img.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            loss, acc = model(img, label)
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if iteration % gradient_accumlation_steps == 0: 
                optimizer.zero_grad()
                optimizer.step()
                avg_loss.update(loss.item() / gradient_accumlation_steps)
                break

            avg_acc.update(acc)
            print('acc: ', acc)
            break
            iteration += 1
            
        # 모델 저장
        checkpoint_path = os.path.join(model_path, f'epoch{epoch}_acc_{avg_acc.avg:.3f}')
        torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict()}, checkpoint_path)
        
        # config.json 저장
        model.config.to_json_file(os.path.join(model_path, 'config.json'))
        
        # checkpoint 파일을 pytorch_model.bin으로 복사 (불확실성 계산용)
        import shutil
        shutil.copy(checkpoint_path, os.path.join(model_path, 'pytorch_model.bin'))
        
        # 토크나이저 저장
        tokenizer.save_vocabulary(model_path)
        
        break
        loss_list.append(avg_loss.avg)
        acc_list.append(avg_acc.avg)

    print(loss_list)
    print(acc_list)


# 불확실성 계산 함수
def uncertainty_estimation(max_samples=None, batch_size=16):
    print("CLIP 없이 불확실성을 계산합니다...")
    path = 'model'
    ckpt_path = 'model/pytorch_model.bin'
    data_path = 'data'
    
    # 디렉토리가 없으면 생성
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created directory: {data_path}")

    # 필요한 토크나이저 파일이 있는지 확인
    if not os.path.exists(os.path.join(path, 'vocab.txt')):
        print(f"Error: vocab.txt not found in {path}!")
        if os.path.exists('data/vocab.txt'):
            # data 폴더에서 복사
            import shutil
            shutil.copy('data/vocab.txt', os.path.join(path, 'vocab.txt'))
            print(f"Copied vocab.txt from data to model folder")
        elif os.path.exists('ckpt/vocab.txt'):
            # ckpt 폴더에서 복사
            import shutil
            shutil.copy('ckpt/vocab.txt', os.path.join(path, 'vocab.txt'))
            print(f"Copied vocab.txt from ckpt to model folder")
        else:
            print("Cannot find vocab.txt in any folder. Please provide it.")
            return
            
    # 토크나이저 로드 시 직접 path를 사용하는 대신 이미 복사된 파일을 사용
    tokenizer = BertTokenizer(os.path.join(path, 'vocab.txt'), do_lower_case=True)
    
    # 모델 설정 로드
    if not os.path.exists(os.path.join(path, 'config.json')):
        print(f"Error: config.json not found in {path}!")
        # 기본 설정 만들기
        # README에서 확인된 값으로 vocab_size 설정
        configuration = BertConfig(vocab_size=10877, 
                                  num_hidden_layers=3,
                                  intermediate_size=2048)
        configuration.save_pretrained(path)
        print(f"Created default config.json in {path}")
    else:
        model_config = BertConfig.from_pretrained(path)

    model = BagofWords(model_config)

    # 체크포인트 파일 확인
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found! Cannot proceed with uncertainty estimation.")
        return
        
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    
    # GPU 사용 가능하면 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval() 

    unk = tokenizer._convert_token_to_id('[UNK]')
    
    # 이미지 특징 파일 확인
    h5py_path = os.path.join(data_path, 'coco_detections.hdf5')
    if not os.path.exists(h5py_path):
        print(f"Error: {h5py_path} not found! Please download the file first.")
        return
    
    img_features = h5py.File(h5py_path)
    
    # 주석 파일 확인
    caption_path = os.path.join(data_path, 'annotations')
    train_caption_path = os.path.join(caption_path, 'captions_train2014.json')
    if not os.path.exists(train_caption_path):
        print(f"Error: {train_caption_path} not found! Please download annotations first.")
        return
        
    with open(train_caption_path, 'r', encoding='utf-8') as j:
        captions = json.load(j)
    
    # 총 샘플 수 계산
    total_samples = len(captions['annotations'])
    if max_samples and max_samples < total_samples:
        print(f"처리할 샘플 수를 {max_samples}개로 제한합니다 (전체: {total_samples}개)")
        samples = captions['annotations'][:max_samples]
    else:
        samples = captions['annotations']
        print(f"총 {total_samples}개 샘플을 처리합니다")
    
    # tqdm을 사용하여 진행 상황 표시 개선
    try:
        from tqdm import tqdm
    except ImportError:
        print("tqdm 패키지가 설치되어 있지 않습니다. 'pip install tqdm'으로 설치하세요.")
        # tqdm이 없으면 간단한 진행률 표시 함수 정의
        def tqdm(iterable, **kwargs):
            total = len(iterable)
            for i, item in enumerate(iterable):
                if i % 10 == 0 or i == total - 1:
                    print(f"Progress: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
                yield item
    
    # 결과 저장용
    annotation_list = []
    
    # 배치 처리를 위한 준비
    num_batches = (len(samples) + batch_size - 1) // batch_size
    
    print(f"배치 크기: {batch_size}, 총 배치 수: {num_batches}")
    
    # 배치 단위로 처리
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(samples))
        current_batch = samples[batch_start:batch_end]
        
        for sample in current_batch:
            try:
                img_id = str(sample['image_id']) + '_features'
                
                # 이미지 ID가 존재하는지 확인
                if img_id not in img_features:
                    print(f"Warning: {img_id} not found in features file. Skipping.")
                    continue
                
                # 토큰화
                tokens = tokenizer.tokenize(sample['caption'])
                caption_ids = tokenizer.convert_tokens_to_ids(tokens)
                
                # 이미지 특징 로드 및 모델 실행
                input_f = torch.FloatTensor(img_features[img_id]).view(1, -1, 2048).to(device)
                with torch.no_grad():  # 메모리 효율성 향상
                    pro_vocab = model(img_embs=input_f).view(-1).cpu()  # GPU 메모리 절약을 위해 CPU로 이동
                
                # 불확실성 계산
                uncertainty = []
                for word_idx in caption_ids:
                    if word_idx == unk:
                        uncertainty.append(0)
                    else:
                        uncertainty.append(pro_vocab[word_idx].item())
                
                sample['uncertainty'] = uncertainty
                annotation_list.append(sample)
                
            except Exception as e:
                print(f"Error processing sample {sample['image_id']}: {e}")
    
    # 결과 저장
    new_captions = {'annotations': annotation_list}
    output_path = os.path.join(data_path, 'uncertainty_captions.json')
    
    print(f"총 {len(annotation_list)}개 샘플 처리 완료. 결과를 {output_path}에 저장합니다...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_captions, f, indent=4)
    
    print("불확실성 계산이 완료되었습니다.")


if __name__ == "__main__":
    # 사용자 선택
    print("1. BOW 모델 학습")
    print("2. 불확실성 계산")
    print("3. 모두 실행")
    choice = input("선택하세요 (1-3): ")
    
    if choice == '1':
        train()
    elif choice == '2':
        uncertainty_estimation()
    elif choice == '3':
        train()
        uncertainty_estimation()
    else:
        print("잘못된 선택입니다.") 