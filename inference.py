from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  
import torch 
from model import UAIC 
from transformers import BertTokenizer
import os 
import json 
import h5py 
from train import SPECIAL_TOKENS 
from evaluation import compute_scores, PTBTokenizer

#  ["[BOS]", "[EOS]", "[NONE]" "[IMG]", "[TXT]", "[PAD]"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# generate candidates using inout image 
def generate_cap(img_feature, model, tokenizer):
    print("\n--- 캡션 생성 시작 ---")
    bos, eos, none, img, txt = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    
    # 특수 토큰 정보 출력
    print(f"Special Tokens IDs: BOS={bos}, EOS={eos}, NONE={none}, IMG={img}, TXT={txt}")
    print(f"Special Tokens: BOS={tokenizer.convert_ids_to_tokens(bos)}, EOS={tokenizer.convert_ids_to_tokens(eos)}, NONE={tokenizer.convert_ids_to_tokens(none)}")
    
    input_txt = ['[bos]']
    output_ids = []

    step = 0
    MAX_LEN = 30  # 최대 길이를 512에서 30으로 줄임 (보통 캡션은 짧음)
    while True:
        print(f"\n현재 단계: {step}, 현재 입력: {input_txt}")
        
        if len(output_ids) > 0:
            print(f"출력 IDs: {output_ids}")
            if output_ids == [none] * len(output_ids):
                print("모든 출력이 NONE입니다. 생성 중단.")
                break
        
        if step >= MAX_LEN:
            print("Warning: Reached max steps in generate_cap. Forcing exit.")
            break
            
        # 토큰 ID 변환 및 출력
        input_txt_ids = tokenizer.convert_tokens_to_ids(input_txt)
        print(f"입력 토큰 IDs: {input_txt_ids}")
        
        input_txt_tensor = torch.Tensor(input_txt_ids).long().to(device)
        input_embs = model.transformer.embeddings.word_embeddings(input_txt_tensor)

        img_embs = model.image_ff(img_feature)
        print(f"이미지 임베딩 형태: {img_embs.shape}, 텍스트 임베딩 형태: {input_embs.shape}")

        # (중요!) 512 초과 시 input_embs 자르기
        total_len = img_embs.size(0) + input_embs.size(0)
        if total_len > 512:  # 전체 MAX_LEN은 512로 유지
            excess = total_len - 512
            input_embs = input_embs[excess:]  # 뒤에서 자르기
            input_txt_tensor = input_txt_tensor[excess:]  # 대응되는 토큰 ID도 잘라야 함

        input_embs = torch.cat([img_embs, input_embs], dim=0)

        token_type_ids = [img] * img_embs.size(0) + [txt] * input_txt_tensor.size(0)
        token_type_ids = torch.Tensor(token_type_ids).long()

        token_type_embs = model.transformer.embeddings.word_embeddings(token_type_ids.to(device))

        input_embs = input_embs + token_type_embs

        # 모델에 입력
        out = model(input_embs.view(1, -1, 768))
        out = out[0].squeeze(0)[-input_txt_tensor.size(0):, :]

        # 출력 확률 확인 (상위 3개)
        probs = torch.softmax(out, dim=1)
        top_probs, top_indices = torch.topk(probs, 3, dim=1)
        
        for i in range(len(top_indices)):
            top_tokens = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top_indices[i]]
            top_probs_val = [f"{p:.4f}" for p in top_probs[i].tolist()]
            print(f"위치 {i}의 상위 토큰: {list(zip(top_tokens, top_probs_val))}")
            
        # 출력 생성
        output = torch.argmax(out, dim=1)
        output_ids = output.cpu().numpy().tolist()
        output_txt = tokenizer.convert_ids_to_tokens(output_ids)
        print(f"생성된 출력 토큰: {output_txt}")
        
        # 다음 단계 입력 생성
        new_input_txt = sequence_stage_combine(input_txt, output_txt)
        
        if new_input_txt == input_txt:  # 입력이 변하지 않으면 중단
            print("입력이 변하지 않았습니다. 생성 중단.")
            break
            
        input_txt = new_input_txt
        step += 1

    print(f"최종 생성 결과: {input_txt}")
    # 특수 토큰 제거
    result = [token for token in input_txt if token not in ['[bos]', '[eos]', '[NONE]', '[none]', '[pad]']]
    print(f"특수 토큰 제거 후: {result}")
    return result


# concatenate the substage to total sentence 
def sequence_stage_combine(input_txt, output_txt):
    print(f"Combining: Input={input_txt}, Output={output_txt}")
    new_sequence = []
    min_len = min(len(input_txt), len(output_txt))  # 길이 보정

    for idx in range(min_len):
        new_sequence.append(input_txt[idx])
        if output_txt[idx] != '[NONE]':
            new_sequence.append(output_txt[idx])

    # 남은 input_txt가 있다면 그냥 추가
    if len(input_txt) > min_len:
        new_sequence.extend(input_txt[min_len:])

    print(f"Combined result: {new_sequence}")
    return new_sequence


# evaluate the ckpt 
def eval():
    ckpt_path = 'ckpt'
    data_path = 'data'

    print("\n--- 모델 및 토크나이저 로딩 ---")
    tokenizer = BertTokenizer.from_pretrained(ckpt_path)
    print(f"토크나이저 어휘 크기: {len(tokenizer.vocab)}")
    print(f"토크나이저 샘플 토큰: {list(tokenizer.vocab.keys())[:10]}")
    
    model = UAIC.from_pretrained(ckpt_path)
    model = model.to(device)
    model.eval()
    
    # 토크나이저를 위한 설정
    ptb_tokenizer = PTBTokenizer()

    annotation_path = os.path.join(data_path, 'annotations')
    val_path = os.path.join(annotation_path, 'captions_val2014.json')
    val_data = json.load(open(val_path, 'r'))
    val_data = val_data['annotations']
    val_data = val_data[:1]  # 테스트를 위해 1개로 설정
    img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'))
    
    with torch.no_grad(): 
        gts = {}  # 정답 캡션 저장
        gen = {}  # 생성된 캡션 저장
        
        for idx, instance in enumerate(val_data):
            img_id = str(instance['image_id'])
            print(f"\n처리 중인 이미지 ID: {img_id}")
            print(f"정답 캡션: {instance['caption']}")
            
            img_feature = torch.FloatTensor(img_features[img_id + '_features']).to(device)
            print(f"이미지 특성 형태: {img_feature.shape}")
            
            candidates = generate_cap(img_feature, model, tokenizer)
            
            # 특수 토큰 제거 및 문장 생성
            caption = ' '.join([token for token in candidates if token not in ['[bos]', '[eos]', '[NONE]', '[none]']])
            print(f"생성된 캡션: {caption}")
            
            # 결과 저장 (문자열 형식으로)
            if img_id not in gts:
                gts[img_id] = []
            gts[img_id].append(instance['caption'])
            
            if img_id not in gen:
                gen[img_id] = []
            gen[img_id].append(caption)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(val_data)} images")
        
        print("\n토크나이징 진행 중...")
        # 토크나이징
        gts = ptb_tokenizer.tokenize(gts)
        gen = ptb_tokenizer.tokenize(gen)
        
        # 모든 평가 지표 계산
        print("평가 지표 계산 중...")
        scores, _ = compute_scores(gts, gen)
        
        # 결과 출력
        print("\n===== Evaluation Results =====")
        for metric, score in scores.items():
            if metric == "Bleu":
                for i, bleu_score in enumerate(score):
                    print(f"BLEU-{i+1}: {bleu_score:.4f}")
            elif isinstance(score, list):
                print(f"{metric}: {', '.join([f'{s:.4f}' for s in score])}")
            else:
                print(f"{metric}: {score:.4f}")


if __name__ == "__main__": 
    eval()