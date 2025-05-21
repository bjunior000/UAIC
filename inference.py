import json 
import torch 
import os 
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction  
from model import UAIC 
from transformers import BertTokenizer
import os 
import json 
import h5py 
from train import SPECIAL_TOKENS_DICT
from evaluation import compute_scores, PTBTokenizer
import time

# 수정된 SPECIAL_TOKENS 정의
SPECIAL_TOKENS = ["[BOS]", "[EOS]", "[NONE]", "[IMG]", "[TXT]", "[PAD]"]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU 관련 설정
if torch.cuda.is_available():
    # GPU 메모리 캐싱 활성화
    torch.backends.cudnn.benchmark = True
    # 결정론적 알고리즘 비활성화 (속도 향상)
    torch.backends.cudnn.deterministic = False


# generate candidates using inout image 
def generate_cap(img_feature, model, tokenizer, debug=False):
    bos, eos, none, img, txt, pad = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    input_txt = ['[BOS]']
    output_ids = []

    # 디버깅 정보 출력
    if debug:
        print(f"특수 토큰 ID: BOS={bos}, EOS={eos}, NONE={none}, IMG={img}, TXT={txt}, PAD={pad}")
        print(f"UNK 토큰 ID: {tokenizer.unk_token_id}, PAD 토큰 ID: {tokenizer.pad_token_id}")

    step = 0
    MAX_LEN = 512
    all_generated_tokens = []  # 모든 생성된 토큰을 저장
    
    while True:
        if len(output_ids) > 0 and output_ids == [none] * len(output_ids):
            break
        if step >= MAX_LEN:
            break
        input_txt_ids = tokenizer.convert_tokens_to_ids(input_txt)
        
        # 디버깅
        if debug and step < 3:
            print(f"Step {step} - input_txt: {input_txt}")
            print(f"input_txt_ids: {input_txt_ids}")
        
        input_txt_tensor = torch.tensor(input_txt_ids, dtype=torch.long, device=device)
        input_embs = model.transformer.embeddings.word_embeddings(input_txt_tensor)

        img_embs = model.image_ff(img_feature)

        # 512 초과 시 input_embs 자르기
        total_len = img_embs.size(0) + input_embs.size(0)
        if total_len > MAX_LEN:
            excess = total_len - MAX_LEN
            input_embs = input_embs[excess:]
            input_txt_tensor = input_txt_tensor[excess:]

        input_embs = torch.cat([img_embs, input_embs], dim=0)

        token_type_ids = [img] * img_embs.size(0) + [txt] * input_txt_tensor.size(0)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)

        token_type_embs = model.transformer.embeddings.word_embeddings(token_type_ids)

        input_embs = input_embs + token_type_embs

        out = model(input_embs.view(1, -1, 768))
        out = out[0].squeeze(0)[-input_txt_tensor.size(0):, :]

        # 모델 출력의 상위 토큰 확인 (디버깅)
        if debug and step < 3:
            top_probs, top_indices = torch.topk(out, 5, dim=1)
            print(f"Top 5 토큰 (첫 번째 위치): {[tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in top_indices[0]]}")
            print(f"Top 5 확률 (첫 번째 위치): {top_probs[0].tolist()}")

        output = torch.argmax(out, dim=1)
        output_ids = output.cpu().numpy().tolist()
        output_txt = tokenizer.convert_ids_to_tokens(output_ids)
        
        # 모든 생성된 토큰 저장
        all_generated_tokens.extend(output_txt)
        
        # 디버깅
        if debug and step < 3:
            print(f"output_ids: {output_ids}")
            print(f"output_txt: {output_txt}")
            print("-" * 30)
        
        input_txt = sequence_stage_combine(input_txt, output_txt)
        step += 1

    # 모든 생성된 토큰 분석 (수정)
    if debug:
        print(f"생성된 모든 토큰 ({len(all_generated_tokens)}개): {all_generated_tokens[:20]}...")
        
        token_counts = {}
        for token in all_generated_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        print("가장 많이 생성된 토큰:")
        for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {token}: {count}회 ({count/len(all_generated_tokens)*100:.1f}%)")
        
        # UNK 토큰 분석
        unk_count = token_counts.get('[UNK]', 0)
        if unk_count > 0:
            print(f"UNK 토큰: {unk_count}회 ({unk_count/len(all_generated_tokens)*100:.1f}%)")
        
        # NONE 토큰 분석
        none_count = token_counts.get('[NONE]', 0)
        if none_count > 0:
            print(f"NONE 토큰: {none_count}회 ({none_count/len(all_generated_tokens)*100:.1f}%)")

    # 빈 캡션 방지를 위해 토큰 목록 그대로 반환 (후처리는 호출자에서 수행)
    return input_txt, all_generated_tokens


# concatenate the substage to total sentence 
def sequence_stage_combine(input_txt, output_txt):
    new_sequence = []
    min_len = min(len(input_txt), len(output_txt))  # 길이 보정

    for idx in range(min_len):
        new_sequence.append(input_txt[idx])
        if output_txt[idx] != '[NONE]':
            new_sequence.append(output_txt[idx])

    # 남은 input_txt가 있다면 그냥 추가
    if len(input_txt) > min_len:
        new_sequence.extend(input_txt[min_len:])

    return new_sequence


# 배치 처리를 위한 함수
def process_batch(img_features_batch, model, tokenizer, debug=False):
    results = []
    raw_tokens = []  # 모든 생성된 토큰 저장
    
    # 배치 처리 (CUDA 스트림 사용 방식 수정)
    for i, img_feature in enumerate(img_features_batch):
        # 첫 번째 배치의 첫 번째 이미지만 디버그
        current_debug = debug and i == 0
        input_txt, all_tokens = generate_cap(img_feature, model, tokenizer, debug=current_debug)
        results.append(input_txt)
        raw_tokens.append(all_tokens)
    
    return results, raw_tokens


# evaluate the ckpt 
def eval():
    ckpt_path = 'ckpt'
    data_path = 'data'

    tokenizer = BertTokenizer.from_pretrained(ckpt_path)
    # 특수 토큰 추가
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    
    print("평가 시작...")
    
    # 토크나이저 정보 확인
    print(f"어휘 크기: {len(tokenizer)}")
    print(f"특수 토큰: {SPECIAL_TOKENS}")
    print(f"PAD 토큰 ID: {tokenizer.pad_token_id}")
    print(f"UNK 토큰 ID: {tokenizer.unk_token_id}")
    print(f"특수 토큰 인덱스: {[tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS]}")
    
    # 모델을 GPU로 명시적 이동
    model = UAIC.from_pretrained(ckpt_path)
    model = model.to(device)
    model.eval()
    
    # GPU 메모리 사용량 확인
    if torch.cuda.is_available():
        print(f"초기 GPU 메모리 사용량: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    annotation_path = os.path.join(data_path, 'annotations')
    val_path = os.path.join(annotation_path, 'captions_val2014.json')
    val_data = json.load(open(val_path, 'r'))
    val_data = val_data['annotations']
    # 전체 검증 데이터셋을 사용하려면 다음 라인을 주석 처리
    val_data = val_data[:20]  # 빠른 테스트를 위해 20개만 사용
    img_features = h5py.File(os.path.join(data_path, 'coco_detections.hdf5'))
    
    print(f"총 {len(val_data)}개 샘플 평가 중...")
    
    # 배치 크기 설정
    BATCH_SIZE = 4  # GPU 메모리에 맞게 조절 가능
    
    with torch.no_grad(): 
        gts = {}  # ground truth 저장
        gen = {}  # 생성된 캡션 저장
        all_captions = []  # 모든 생성된 캡션 저장
        
        # 배치 처리를 위해 데이터 준비
        batches = [val_data[i:i+BATCH_SIZE] for i in range(0, len(val_data), BATCH_SIZE)]
        
        start_time = time.time()
        
        # 모델이 생성하는 토큰 통계
        total_generated_tokens = []
        
        for batch_idx, batch in enumerate(batches):
            # 진행률 표시
            print(f"배치 처리 중: {batch_idx+1}/{len(batches)} ({(batch_idx+1)/len(batches)*100:.0f}%)")
            
            # 배치 이미지 특성 로드
            batch_features = []
            batch_image_ids = []
            batch_references = []  # 참조 캡션 저장
            
            # GPU 메모리로 직접 로드
            for instance in batch:
                img_id = str(instance['image_id']) + '_features'
                # numpy 배열로 변환 후 GPU로 바로 전송
                img_feature = torch.tensor(np.array(img_features[img_id]), dtype=torch.float, device=device)
                batch_features.append(img_feature)
                batch_image_ids.append(instance['image_id'])
                batch_references.append(instance['caption'])
                
                # ground truth 저장
                image_id = instance['image_id']
                if image_id not in gts:
                    gts[image_id] = []
                gts[image_id].append(instance['caption'])
            
            # 배치 처리 (첫 번째 배치만 디버그)
            debug_this_batch = (batch_idx == 0)
            batch_candidates, batch_raw_tokens = process_batch(batch_features, model, tokenizer, debug=debug_this_batch)
            
            # 모든 생성된 토큰 수집
            for tokens in batch_raw_tokens:
                total_generated_tokens.extend(tokens)
            
            # 결과 처리
            for idx, (candidates, image_id, reference, raw_tokens) in enumerate(zip(batch_candidates, batch_image_ids, batch_references, batch_raw_tokens)):
                # 특수 토큰 제거 및 문자열로 변환
                removed_tokens = ['[BOS]', '[EOS]', '[NONE]', '[IMG]', '[TXT]', '[PAD]']
                
                # 수정된 방식: 특수 토큰 제거 시 로그 추가
                filtered_tokens = []
                for token in candidates:
                    if token not in removed_tokens:
                        filtered_tokens.append(token)
                
                # 캡션을 문자열로 변환
                caption = ' '.join(filtered_tokens)
                
                # UNK 토큰 비율 확인 - 문제 진단용
                unk_count = caption.count('[UNK]')
                total_tokens = len(filtered_tokens)
                unk_ratio = 0 if total_tokens == 0 else unk_count / total_tokens
                
                # 빈 캡션 방지 - UNK만 있더라도 뭐라도 출력
                if caption.strip() == '':
                    # 특수 토큰 제외한 원시 토큰 사용
                    raw_caption = ' '.join([t for t in raw_tokens if t not in removed_tokens])
                    # 여전히 비어있으면 [UNK] 하나라도 넣기
                    if raw_caption.strip() == '':
                        caption = '[UNK]'
                    else:
                        caption = raw_caption
                
                # 생성된 모든 캡션 저장
                all_captions.append({
                    'image_id': image_id,
                    'reference': reference,
                    'generated': caption,
                    'unk_ratio': unk_ratio,
                    'total_tokens': total_tokens
                })
                
                # 첫 번째 배치의 결과 출력
                if batch_idx == 0 and idx < 2:
                    print(f"\n이미지 ID: {image_id}")
                    print(f"참조 캡션: {reference}")
                    print(f"생성 캡션: {caption}")
                    print(f"UNK 비율: {unk_ratio:.2f} ({unk_count}/{total_tokens})")
                    print(f"원시 토큰: {candidates[:20]}..." if len(candidates) > 20 else candidates)
                    print(f"모든 생성된 토큰: {raw_tokens[:20]}..." if len(raw_tokens) > 20 else raw_tokens)
                    print("-" * 50)
                
                # 메트릭 라이브러리 요구사항: 각 이미지 ID에 대해 하나의 생성된 캡션만 있어야 함
                gen[image_id] = [caption]  # 리스트로 감싸기
            
            # 배치마다 GPU 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        elapsed_time = time.time() - start_time
        print(f"추론 완료: {elapsed_time:.2f}초 소요 (평균: {elapsed_time/len(val_data):.2f}초/샘플)")
        
        if torch.cuda.is_available():
            print(f"최종 GPU 메모리 사용량: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        # 전체 생성된 토큰 분석
        if total_generated_tokens:
            token_counts = {}
            for token in total_generated_tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            
            print("\n전체 생성된 토큰 통계:")
            total_count = len(total_generated_tokens)
            print(f"총 생성된 토큰 수: {total_count}")
            
            print("가장 많이 생성된 토큰:")
            for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  - {token}: {count}회 ({count/total_count*100:.1f}%)")
        
        # 캡션 통계 분석
        print("\n생성된 캡션 통계:")
        empty_captions = sum(1 for item in all_captions if item['generated'].strip() == '')
        print(f"빈 캡션 수: {empty_captions}/{len(all_captions)} ({empty_captions/len(all_captions)*100:.1f}%)")
        
        # UNK 비율 통계
        unk_ratios = [item['unk_ratio'] for item in all_captions]
        avg_unk_ratio = sum(unk_ratios) / len(unk_ratios) if unk_ratios else 0
        print(f"평균 UNK 비율: {avg_unk_ratio:.2f}")
        
        high_unk = sum(1 for r in unk_ratios if r > 0.5)
        print(f"UNK 비율 50% 초과 캡션: {high_unk}/{len(all_captions)} ({high_unk/len(all_captions)*100:.1f}%)")
        
        # 생성된 단어 통계
        all_words = []
        for item in all_captions:
            all_words.extend([w for w in item['generated'].split() if w != '[UNK]'])
        
        if all_words:
            word_counts = {}
            for word in all_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            print("\n[UNK] 제외 가장 자주 나타나는 단어:")
            for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {word}: {count}회")
        else:
            print("\n[UNK] 제외 생성된 텍스트에 단어가 없습니다.")
        
        print("\n토큰화 및 평가 중...")
        # 토크나이저를 사용하여 전처리
        tokenizer_ptb = PTBTokenizer()
        gts = tokenizer_ptb.tokenize(gts)
        gen = tokenizer_ptb.tokenize(gen)
        
        # 토큰화된 결과 확인
        print("\n토큰화 결과 샘플:")
        for image_id in list(gen.keys())[:2]:
            print(f"이미지 ID: {image_id}")
            print(f"참조: {gts.get(image_id, ['없음'])}")
            print(f"생성: {gen.get(image_id, ['없음'])}")
            print("-" * 30)
        
        # 모델 체크포인트 정보 확인
        try:
            print("\n모델 정보:")
            print(f"체크포인트 경로: {ckpt_path}")
            config_path = os.path.join(ckpt_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"모델 타입: {config.get('model_type', '알 수 없음')}")
                print(f"어휘 크기: {config.get('vocab_size', '알 수 없음')}")
        except Exception as e:
            print(f"모델 정보 확인 중 오류: {e}")
        
        # 모든 메트릭으로 평가
        scores, _ = compute_scores(gts, gen)
        
        # 결과 출력
        print("\n===== 평가 결과 =====")
        for metric, score in scores.items():
            if metric == "Bleu":
                # BLEU는 1-4까지 점수가 배열로 제공됨
                for i, s in enumerate(score):
                    print(f"BLEU-{i+1}: {s:.3f}")
            elif isinstance(score, (list, tuple)):
                # 다른 메트릭도 배열인 경우
                for i, s in enumerate(score):
                    print(f"{metric}-{i+1}: {s:.3f}")
            else:
                # 단일 값인 경우
                print(f"{metric}: {score:.3f}")

        # 문제 해결 제안
        print("\n===== 문제 해결 제안 =====")
        if empty_captions > 0 or avg_unk_ratio > 0.3:
            print("1. 모델이 의미 있는 토큰을 생성하지 못하는 문제가 있습니다.")
            print("2. 다음 방법을 시도해 보세요:")
            print("   - 모델 재훈련 또는 다른 체크포인트 사용")
            print("   - 토크나이저와 모델의 어휘 일치 여부 확인")
            print("   - 특수 토큰 정의가 훈련 시와 일치하는지 확인")
            print("   - 생성 로직에서 온도(temperature) 파라미터 추가하여 다양성 증가")


if __name__ == "__main__": 
    print(torch.__version__)              # CUDA 버전 확인을 위한 방법 수정
    print(torch.cuda.is_available())      # True
    print(torch.cuda.get_device_name(0))  # 'NVIDIA GeForce RTX 2060'
    eval()