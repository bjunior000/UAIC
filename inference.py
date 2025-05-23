from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  
import torch 
from model import UAIC 
from transformers import BertTokenizer
import os 
import json 
import h5py 
from train import SPECIAL_TOKENS 

#  ["[BOS]", "[EOS]", "[NONE]" "[IMG]", "[TXT]", "[PAD]"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# generate candidates using inout image 
def generate_cap(img_feature, model, tokenizer): 
    bos, eos, none, img, txt, pad = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS) 
    print(tokenizer.convert_tokens_to_ids('[bos]'))  # [UNK]로 나오면 문제 있음 -> 10869
    print(tokenizer.convert_tokens_to_ids('[BOS]'))  # 올바른 토큰 ID 확인 -> 10874
    # why use small-case? it should be distinguished
    # input_txt = ['[bos]'] 
    input_txt = ['[BOS]']
    # for test
    # input_txt = ['[BOS]', 'a'] 
    output_ids = []
    max_length = 30  # 최대 생성 길이 제한
    iteration = 0
    
    print(f"시작 토큰: {input_txt}")
    
    while True: 
        iteration += 1
        if iteration > max_length:
            print("최대 반복 횟수 도달")
            break
            
        if len(output_ids) > 0 and all(id == none for id in output_ids):
            print("NONE 토큰만 생성되어 종료")
            break 
            
        input_txt_ids = tokenizer.convert_tokens_to_ids(input_txt)
        print(f"입력 토큰 IDs: {input_txt_ids}")
        
        input_txt_tensor = torch.Tensor(input_txt_ids).long().to(device)
        input_embs = model.transformer.embeddings.word_embeddings(input_txt_tensor)
        
        img_embs = model.image_ff(img_feature)
        input_embs = torch.cat([img_embs, input_embs], dim=0)
        
        print(f"입력 임베딩 크기: {input_embs.size()}")
        
        token_type_ids = [img] * img_embs.size(0) + [txt] * input_txt_tensor.size(0)
        token_type_ids = torch.Tensor(token_type_ids).long().to(device)
        
        token_type_embs = model.transformer.embeddings.word_embeddings(token_type_ids)
        input_embs = input_embs + token_type_embs 
        
        # 차원 확인 및 조정
        seq_len = input_embs.size(0)
        if seq_len > 512:
            print(f"경고: 시퀀스 길이({seq_len}) > 512, 잘라냅니다.")
            input_embs = input_embs[:512]
        
        # 배치 차원 추가
        input_embs = input_embs.unsqueeze(0)
        
        # 모델 실행
        with torch.no_grad():
            out = model(input_embs)
        
        logits = out[0].squeeze(0)  # [seq_len, vocab_size]
        
        # 마지막 토큰의 로짓만 사용
        last_token_logits = logits[-input_txt_tensor.size(0):, :]
        
        # 확률 계산
        probs = torch.softmax(last_token_logits, dim=1)
        
        # 가장 확률 높은 토큰 선택
        output = torch.argmax(probs, dim=1)
        output_ids = output.cpu().numpy().tolist()
        
        # 각 토큰의 확률 출력
        top_probs, top_indices = torch.topk(probs, 5, dim=1)
        print(f"Top 5 토큰 확률:")
        for i, (probs_i, indices_i) in enumerate(zip(top_probs, top_indices)):
            top_tokens = tokenizer.convert_ids_to_tokens(indices_i.tolist())
            probs_list = probs_i.tolist()
            print(f"  위치 {i}: {list(zip(top_tokens, probs_list))}")
        
        # 출력 토큰 확인
        output_txt = tokenizer.convert_ids_to_tokens(output_ids)
        print(f"생성된 토큰: {output_txt}")
        
        # 입력 업데이트
        input_txt = sequence_stage_combine(input_txt, output_txt)
        print(f"업데이트된 입력: {input_txt}")
        print("="*50)

    print(f"최종 생성 결과: {input_txt}")
    return input_txt

# concatenate the substage to total sentence 
def sequence_stage_combine(input_txt, output_txt):
    new_sequence = []
    idx = 0
    none_count = 0
    
    # 길이 안전 검사
    min_len = min(len(input_txt), len(output_txt))
    
    # 입력과 출력의 길이 로깅
    print(f"sequence_combine - 입력 길이: {len(input_txt)}, 출력 길이: {len(output_txt)}")
    
    while idx < min_len:
        new_sequence.append(input_txt[idx])
        if output_txt[idx] != '[NONE]':
            print(f"토큰 추가: {output_txt[idx]}")
            new_sequence.append(output_txt[idx])
        else:
            none_count += 1
        idx += 1
    
    # 남은 입력 추가
    while idx < len(input_txt):
        new_sequence.append(input_txt[idx])
        idx += 1
    
    print(f"NONE 토큰 수: {none_count}/{len(output_txt)}")
    return new_sequence


# evaluate the ckpt 
def eval():
    # for test
    
    ckpt_path = 'ckpt'
    data_path = 'data'

    tokenizer = BertTokenizer.from_pretrained(ckpt_path)
    model = UAIC.from_pretrained(ckpt_path)
    model = model.to(device)
    model.eval()
    smooth = SmoothingFunction() 

    annotation_path = os.path.join(data_path, 'annotations')
    val_path = os.path.join(annotation_path, 'captions_val2014.json')
    val_data = json.load(open(val_path, 'r'))
    val_data = val_data['annotations']
    val_data = val_data[:20]
    img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'))
    
    with torch.no_grad(): 
        results = []
        for instance in val_data:
            print(instance)
            img_id = str(instance['image_id']) + '_features'
            img_feature = torch.FloatTensor(img_features[img_id]).to(device)
            candidates = generate_cap(img_feature, model, tokenizer)
            # 모델의 캡션 생성 결과물 출력(디버그용)
            print(candidates)
            reference = instance['caption']
            results.append(corpus_bleu([[reference]], [candidates], smoothing_function=smooth.method1))


if __name__ == "__main__": 
    eval()