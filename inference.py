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
    bos, eos, none, img, txt = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    input_txt = ['[bos]']
    output_ids = []

    step = 0
    MAX_LEN = 512
    while True:
        if len(output_ids) > 0 and output_ids == [none] * len(output_ids):
            break
        if step >= MAX_LEN:
            print("Warning: Reached max steps in generate_cap. Forcing exit.")
            break
        input_txt_ids = tokenizer.convert_tokens_to_ids(input_txt)
        input_txt_tensor = torch.Tensor(input_txt_ids).long().to(device)
        input_embs = model.transformer.embeddings.word_embeddings(input_txt_tensor)

        img_embs = model.image_ff(img_feature)

        # (중요!) 512 초과 시 input_embs 자르기
        total_len = img_embs.size(0) + input_embs.size(0)
        if total_len > MAX_LEN:
            excess = total_len - MAX_LEN
            input_embs = input_embs[excess:]  # 뒤에서 자르기
            input_txt_tensor = input_txt_tensor[excess:]  # 대응되는 토큰 ID도 잘라야 함

        input_embs = torch.cat([img_embs, input_embs], dim=0)

        token_type_ids = [img] * img_embs.size(0) + [txt] * input_txt_tensor.size(0)
        token_type_ids = torch.Tensor(token_type_ids).long()

        token_type_embs = model.transformer.embeddings.word_embeddings(token_type_ids.to(device))

        input_embs = input_embs + token_type_embs

        out = model(input_embs.view(1, -1, 768))
        out = out[0].squeeze(0)[-input_txt_tensor.size(0):, :]

        output = torch.argmax(out, dim=1)
        output_ids = output.cpu().numpy().tolist()
        output_txt = tokenizer.convert_ids_to_tokens(output_ids)
        input_txt = sequence_stage_combine(input_txt, output_txt)

        step += 1

    return input_txt


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


# evaluate the ckpt 
def eval():
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
            reference = instance['caption']
            results.append(corpus_bleu([[reference]], [candidates], smoothing_function=smooth.method1))


if __name__ == "__main__": 
    eval()