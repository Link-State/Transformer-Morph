# transformer 를 이용한 한국어 형태소 분석기 개발
# hugging face EncoderDecoderModel 을 이용함.

# Program:  korean mophological analyzer using transformer that employs two BERTs as encoder and decoder.
#     encoder: etri eojeol BERT
#     decoder: etri morph BERT
#

# 수행 모드:  다음 2 가지 중 하나를 선택해야 한다.

import os
import random
import sys
import time
from datetime import datetime 
import numpy as np
import torch
from tqdm import tqdm, trange
import transformers
from transformers import GenerationConfig  # model.generate 메소드 사용에 적용됨.
import eojeol_etri_tokenizer.file_utils
from eojeol_etri_tokenizer.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from eojeol_etri_tokenizer.eojeol_tokenization import eojeol_BertTokenizer
from morph_etri_tokenizer.morph_tokenization import morph_BertTokenizer
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, EncoderDecoderConfig       # 이것들은 아래서 어떻게 사용되는가?
from transformers import EncoderDecoderModel
from torch.optim import AdamW

def log(msg="", output=True) :
    f_log = open("./log.txt", 'a', encoding="utf-8")
    f_log.write(str(msg) + "\n")
    f_log.close()
    if output :
        print(msg)
    return

def timeformat(elapse:float) :
    second = ""
    minute = ""
    hour = ""
    day = ""

    if elapse >= 86400 :
        day = elapse // 86400
        day = f"{day}d "
        hour = (elapse % 86400) // 3600
        hour = f"{hour}h "
        minute = ((elapse % 86400) % 3600) // 60
        minute = f"{minute}m "
        second = ((elapse % 86400) % 3600) % 60
        second = f"{second:.3f}s"
    elif elapse >= 3600 :
        hour = elapse // 3600
        hour = f"{hour}d "
        minute = (elapse % 3600) // 60
        minute = f"{minute}m "
        second = (elapse % 3600) % 60
        second = f"{second:.3f}s"
    elif elapse >= 60 :
        minute = elapse // 60
        minute = f"{minute}m "
        second = elapse % 60
        second = f"{second:.3f}s"
    else :
        second = elapse
        second = f"{second:.3f} sec"
    
    result = f"{day}{hour}{minute}{second}"

    return result

log(msg=f"\n------------------------{str(datetime.now())}------------------------\n", output=False)

# 작업에 사용할 데이타 구간을 설정해야 한다: 시작 파일 번호와 구간 마지막 파일 번호를 준다:
# 디렉토리 clean 내의 324개의 파일들에서 추천 설정구간:
start_idx_file_to_read  =   309  # clean 내의 파일 중 읽을 구간의 첫 파일의 index
end_idx_file_to_read    =   323  # clean 내의 파일 중 읽을 구간의 마지막 파일의 index

file_for_saving_parameters = datetime.now().strftime("./saved_model/model_%Y%m%d%H%M%S")                        # 훈련을 한 후 결과를 저장할 파일
file_of_previous_parameters_for_further_train = "./saved_model/nnew_bert2bert_3"      # 훈련시작시에 loading 할 이전결과를 담은 파일
file_of_parameters_for_testing = ""                     # 테스트 모드 시에 이 파일을 이용함.
fp_log = open("./Log_1.txt", "w", encoding='utf-8')                    # log 파일 (성능평가 정보등 주요 정보를 저장함)

# 배치 준비 (배치를 만드는데 torch 의 도구 TensorDataSet, DataLoader 이용)
BATCH_SIZE = 16
Max_enc_seq = 128    # encoder 에 입력할 token id seq 의 최대 길이(어절 문장에 대응).
Max_dec_seq = 150    # decoder 에 입력할 token id seq 의 최대 길이(형태소 문장에 대응). 형태소수가 어절수 보다 많으므로 더 크게.
num_EPOCHS = 1       # 총 epoch 수

## special token ids that are defined in eojeol and morph tokenizers.

PAD_token_id = 0    # this will be used for padding.
CLS_token_id = 2    # this will be used as bos token
SEP_token_id = 3    # this will be used as eos token
decoding_vocab_size = 30349   # 형태소분석 문장에 대한 token 사전크기(001_bert_morp_pytorch 의 config 파일에 나옴)

# if CUDA is available, set device to GPU. otherwise CPU is used.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f'Using device:{device}')

# set up tokenizers for encoder and decoder.
eojeol_tokenizer = eojeol_BertTokenizer("./003_bert_eojeol_pytorch/vocab.korean.rawtext.list", do_lower_case=False)
morph_tokenizer = morph_BertTokenizer("./001_bert_morp_pytorch/vocab.korean_morp.list", do_lower_case=False)

clean_dir_path = './clean'  # 학습 원본데이터(cleaned files)를 가진 디렉토리.
clean_file_list = os.listdir(clean_dir_path)    # cleaned 파일명(경로)들을 준비함.

# print(clean_file_list)

# 한 파일의 문장 읽기 함수:  한 파일의 경로를 입력받아 그 안의 문장들을 읽는다.
#  output: 두개의 list.
#    - 첫 리스트는 파일에서 읽은 어절 문장들을 가짐(각 문장은 어절문장(str) 하나).
#    - 둘째 리스트는 파일에서 형태소분석 문장들을 가짐(각 문장은 형태소분석 문장(str) 하나).

if not os.path.exists("./saved_model/") :
    os.makedirs("./saved_model/")

file_list = os.listdir("./saved_model/")
model_list = list()
for file in file_list :
    if os.path.isdir("./saved_model/" + file) :
        model_list.append(file)

override_model = None
if len(model_list) > 0 :
    log("------------------------")
    for i in range(len(model_list)) :
        log(f"[{i+1}]   {model_list[i]}")
    log("[N]   불러오지 않음")
    log("------------------------")
    cmd = input("불러올 모델파일 번호 입력  >> ")
    log(f"불러올 모델파일 번호 입력  >> {cmd}", output=False)
    log("")

    if cmd.isdigit() and int(cmd) >= 1 and int(cmd) <= len(model_list) :
        override_model = "./saved_model/" + model_list[int(cmd)-1]
        file_for_saving_parameters = override_model

# 프로그램 수행 모드 설정: 1: 훈련모드  , 2: 테스트모드
do_training = None
while do_training == None :
    do_training = input("훈련모드(1) 검증모드(2) 테스트모드(3) >> ")
    log(f"훈련모드(1) 검증모드(2) 테스트모드(3) >> {do_training}", output=False)

    if do_training == "1" :
        do_training = 1
    elif do_training == "2" and override_model != None :
        do_training = 2
    elif do_training == "3" and override_model != None :
        do_training = 3
    else :
        do_training = None


def read_sentences_cleaned_from_a_file(fpath):
    eojeol_sentences = []
    morph_sentences = []
    lcnt = 0
    with open(fpath, "r", encoding='euc-kr') as fp:
        for line in fp.readlines():
            if lcnt % 2 == 0:
                line = line[:-1]    # This line has eojeols. remove new line char at last loc.
                eojeol_sentences.append(line)    # line is a string of a sentence.
            else:
                # this line has tagged morphemes of eojeols. Morphemes in the same eojeol are joined by '+".
                splited = line.split(' ')   # each element of splited is a morph analysis result for an eojeol.
                morph_list_of_sentence = []
                for morph_list_of_eojeol in splited:
                    morph_list = morph_list_of_eojeol.split('+')
                    morph_list_of_sentence.extend(morph_list)
                # create one strings of all morps in a sentence.
                morph_string_of_sentence = ''
                for str in morph_list_of_sentence:
                    morph_string_of_sentence = morph_string_of_sentence + ' ' + str
                # remove blank at loc 0 and new line char at last loc.
                morph_string_of_sentence = morph_string_of_sentence[1:-1]
                morph_sentences.append(morph_string_of_sentence)
            lcnt = lcnt + 1
    #print("number of lines in file:", fpath, " = ", int(lcnt/2))
    return eojeol_sentences, morph_sentences

#  Tokenization 단계:  모든 문장을 읽어서 문장마다 token id sequence 를 만들어 저장한다.
#   (1) e_sentences_id list: 어절문장 마다 eojeol token id sequence 를 준비하여 원소로 넣음.
#   (2) m_sentences_id list: 형태소문장 마다 morph token id sequence 를 준비하여 원소로 넣음.
#
#  그리고 e_sentences_str, m_sentences_toks 를 준비한다:
#    e_sentences_str: 어절문장 스트링을 넣은 리스트(각 원소는 문장스트링)
#    m_sentences_toks: 형태소문장을 tokenize 하여 얻은 리스트를 넣은 리스트(각 원소는 한 형태소문장의 토큰(str) sequence).
#
#  주의: 위 과정에서 truncation 을 미리 수행한다. 즉 토큰 개수가 최대수 보다 많은 문장은 버린다.
#       그리고 문장마다 처음과 마지막에 특수토큰 [CLS], [SEP] 를 추가한다.

# change sentence strings (of eojeols and morpheme/pos pairs) into lists of token id's.
e_sentences_id, m_sentences_id = [], []
e_sentences_str = []    # 각 원소는 문장 하나의 단어들로 구성된 스트링
m_sentences_toks = []   # 각 원소는 문장 하나의 토큰(토큰스트링)들로 구성되는 스트링.

num_files_read = 0

for j, afile in enumerate(clean_file_list):
    if j < start_idx_file_to_read:
        continue   # 이 파일은 무시하고 다음 파일로 간다.
    elif j > end_idx_file_to_read:
        break  # 읽기 종료.
    else:
        i = 0   # 아무 의미 없고 그냥 아래로 진행하기 위함.

    filepath = './clean/' + afile
    num_files_read += 1

    # read all sentences in a file
    e_sents, m_sents = read_sentences_cleaned_from_a_file(filepath)
    for i in range(len(e_sents)):
        e_sent = e_sents[i]     # get a sentence string consisting of eojeols.
        m_sent = m_sents[i]     # get a sentence string consisting of morph/pos
        e_sent_tk = eojeol_tokenizer.tokenize(e_sent)   # 각 원소는 어절토큰(스트링)
        m_sent_tk = morph_tokenizer.tokenize(m_sent)    # 각 원소는 형태소토큰(스트링)
        eleng = len(e_sent_tk)
        mleng = len(m_sent_tk)
        if eleng > Max_enc_seq - 5 or mleng > Max_dec_seq - 5:
                    #  -5 를 한 이유는 [CLS],[SEP] 와 추가적인 3 칸의 여유를 위함.
            continue    # ignore this sentence. 너무 긴 문장이다.

        if eleng < 1 or mleng < 1:
            continue	# 토큰이 하나도 없는 문장으므로 버린다.

        e_sent_tkid = eojeol_tokenizer.convert_tokens_to_ids(e_sent_tk)     # token을 index 로 변경.
        m_sent_tkid = morph_tokenizer.convert_tokens_to_ids(m_sent_tk)	    #   "       "        "

		# 문장 시작에 [CLS] 토큰을,  끝에   [SEP] 토큰을 추가한다.

        e_sent_tkid_bos_eos_added = [CLS_token_id] + e_sent_tkid + [SEP_token_id]
        m_sent_tkid_bos_eos_added = [CLS_token_id] + m_sent_tkid + [SEP_token_id]

        e_sentences_id.append(e_sent_tkid_bos_eos_added)
        m_sentences_id.append(m_sent_tkid_bos_eos_added)

        e_sentences_str.append(e_sent)  # 어절 문장의 단어스트링으로 된 문장들을 저장.
        m_sentences_toks.append(m_sent_tk)  # 형태소 문장의 토큰스트링으로 된 문장들을 저장.

    if num_files_read % 20 == 0:
      log(f"num files read so far : {num_files_read}")

# prepare data for model input and target: input_ids, decoder_input_ids.

total_num_sents = len(e_sentences_id)

# 학습 data 만들기 단계 (결국 torch tensor 형태로 만듬)
#    1) input_ids : encoder 의 입력으로 넣을 input token sequences
#    2) decoder_input_ids: decoder 의 입력으로 넣을 input token sequences <-- 더 이상 사용 안함.
#    3) attention_mask: input_ids 에 대한 masking 정보(0/1 로 표시)
#    4) attention_masks_decode: decoder_input_ids 에 대한 masking 정보(0/1 로 표시) <--- 더 이상 사용 안함.
#    5) labels: decoder 의 출력에 대한 target 정보 (이는 결국 decoder_input_ids 와 동일함.
#                 단, pad 토큰에 대해서는 -100 로 변경해야 함.)

input_ids = []  # training data for encoder input
decoder_input_ids = []  # training data for decoder input

# Padding 작업 수행:  길이가 최대길이보다 적으면 패딩토큰 [PAD]= 0 으로 재워 길이를 최대 길이로 맞춘다.
# (truncation is not needed since longer sentences were filtered before. )
# 패딩에 사용되는 토큰으로는 [PAD] 로 id 는 0 이다.
# 그 결과로 input_ids의 원소들은 동일 길이(Max_enc_seq)의 token id seq,
#   decoder_input_ids의 원소들도 동일 길이(Max_dec_seq)의 token id seq 이다.

for i in range(total_num_sents):
    e_sent_id = e_sentences_id[i]
    eleng = len(e_sent_id)
    d_sent_id = m_sentences_id[i]
    dleng = len(d_sent_id)

    if eleng < Max_enc_seq:
        e_sent_id = e_sent_id + (Max_enc_seq - eleng)*[PAD_token_id]      # padding 수행
    if dleng < Max_dec_seq:
        d_sent_id = d_sent_id + (Max_dec_seq - dleng)*[PAD_token_id]      # padding 수행

    input_ids.append(e_sent_id)
    decoder_input_ids.append(d_sent_id)

# 마스킹정보 및 레이블정보 만들기:  attention_masks and decoder_attention_masks and labels.
# As a mask value, 0.0 is given for pads and 1.0 is given for non-pad tokens.

attention_masks = [[float(i > 0) for i in seq] for seq in input_ids]        # i 는 token id. [PAD] 토큰은 0, 아니면 1.
#attention_masks_decode = [[float(i > 0) for i in seq] for seq in decoder_input_ids]     # [PAD] 토큰은 0, 아니면 1.
tlabels = decoder_input_ids
tlabels = [[-100 if tk == PAD_token_id else tk for tk in tklist] for tklist in tlabels]  # [PAD] 토큰은 -100 으로 변경.

# 학습데이타의 tensor 화 (주: 아직은 main memory 에 준비; 사용 직전에 GPU 로 보내서 model에 입력한다.)

input_ids = torch.tensor(input_ids)		# 각 행은 각 eojeol 문장의 token id 들로 구성. 모든 행의 길이가 동일.
attention_masks = torch.tensor(attention_masks)
labels = torch.tensor(tlabels)

log(f"total number of sentences : {total_num_sents}")

#  배치 준비
data = TensorDataset(input_ids, attention_masks, labels)
dataloader = DataLoader(data, sampler=RandomSampler(data), batch_size=BATCH_SIZE, drop_last=True)
num_batches = len(dataloader)

log(f"total number of batches = {num_batches}")

# 다음 함수는 한 배치에 대한 모델의 출력에 대해 decoder output token id sequences 로부터 morpheme sequences 를 만든다.
# 그리고 이 배치에 대하여 target 이 되는 morpheme sequences 를 만든다.

def get_morpheme_sequences_for_model_output(pred_decoder_id_seqs, target_dec_id_seqs, num_sentences):

    morphs_sentences_model = [] # morpheme-sentences (for a batch) generated by the model.
    morphs_sentences_tgt = []   #    "         "          "        of the target.

    for i in range(num_sentences):

        toks_model = morph_tokenizer.convert_ids_to_tokens(pred_decoder_id_seqs[i])
        # 디코더의 target (즉 labels)을 준비할 때 0 을 -100으로 변경했었다. 이를 복원해야 한다.
        for k in range(Max_dec_seq):
            if target_dec_id_seqs[i][k] == -100:        # -100이 label 이면 다시 pad token 으로 변경.
                target_dec_id_seqs[i][k] = 0
        toks_tgt = morph_tokenizer.convert_ids_to_tokens(target_dec_id_seqs[i])

        # prepare morpheme seq for a sentence predicted by model and store it in morphs_sentences_model.
        morphs_one_sentence_model = []
        morph_model = ''
        for tok in toks_model:
            if tok in ['[PAD]', '[CLS]', '[SEP]']:  # ignore special tokens.
                continue
            morph_model += tok  # concatenate a token to a morph.
            if tok[-1] == '_':  # this is the last tok of a morph.
                morphs_one_sentence_model.append(morph_model)  # finish a morph and store it.
                # if "?/SF_" morpheme has come, we finish creating this sentence.
                if (len(morph_model) >= 3) and (morph_model[-1] == '_' and morph_model[-2] == 'F' and morph_model[-3] == 'S'):
                    break  # stop making a sentence at the first /SF_ token.
                else:
                    morph_model = ''    # 새로운 morpheme(형태소)를 만들기 위해 초기화 한다.

        morphs_sentences_model.append(morphs_one_sentence_model)

        # prepare  target morpheme sequence of a sentence and store in a list morphs_sentences_tgt
        morphs_one_sentence_tgt = []
        morph_tgt = ''
        for tok in toks_tgt:
            if tok in ['[PAD]', '[CLS]', '[SEP]']:  # ignore special tokens.
                continue
            morph_tgt += tok  # concatenate a token to a morph.
            if tok[-1] == '_':  # this is the last tok of a morph.
                morphs_one_sentence_tgt.append(morph_tgt)  # finish a morph and store it.
                # if "?/SF_" morpheme has come, we finish creating this sentence.
                if morph_tgt[-1] == '_' and morph_tgt[-2] == 'F' and morph_tgt[-3] == 'S':
                    break  # stop at the first /SF_ token.
                else:
                    morph_tgt = ''  # initialize to construct another morpheme.

        morphs_sentences_tgt.append(morphs_one_sentence_tgt)

    return morphs_sentences_model, morphs_sentences_tgt

# 모델이 생성한 한 문장의 형태소 리스트와 이에 대응하는 target 형태소 리스트를 비교하여
# 모델이 한 문장에서 잘 알아낸 형태소 수를 구하여 반환한다.
# matching succeeds if an element in target seq is found in an unused lococation of model seq.
# count the number of elements in target seq that succeeds in matching.
def match_simple(seqt, seqm):
    tleng = len(seqt)
    mleng = len(seqm)
    used = np.zeros((mleng), dtype=np.intc) #used flag
    match_cnt = 0

    # file-write target morpheme seq of a sentence.
    fp_log.write("t>  ")
    fp_log.flush()
    for t in range(len(seqt)):
        fp_log.write(seqt[t]+" ")
        fp_log.flush()
    fp_log.write("\n")
    fp_log.flush()

    # file-write model-generated morpheme seq of a sentence.
    fp_log.write("m>  ")
    fp_log.flush()
    for m in range(len(seqm)):
        fp_log.write(seqm[m] + " ")
        fp_log.flush()
    fp_log.write("\n")
    fp_log.flush()

    for i in range(tleng):
        # find seqt[i] in seqm without using used elements of seqm.

        for k in range(mleng):
            if used[k] == 0 and seqm[k] == seqt[i]:
                used[k] = 1
                match_cnt += 1
                break
    fp_log.write("Match count = "+str(match_cnt)+" out of "+str(tleng)+"\n\n")
    fp_log.flush()

    return match_cnt

###### 모델 설정 ###################################################################################

# 아래 3 줄 중 하나만 선택하고 다른 2 줄은 첫 글자로 # 를 넣어 코멘트로 만들어서 모델을 설정한다.
if do_training == 1 :
    if override_model == None :
        model = EncoderDecoderModel.from_encoder_decoder_pretrained("./003_bert_eojeol_pytorch", "./001_bert_morp_pytorch") # 원점에서 시작하는 훈련모드 수행
    else :
        model = EncoderDecoderModel.from_pretrained(override_model)    # 이전훈련결과에서 이어서 시작하는 훈련모드 수행
elif do_training == 2 or do_training == 3 :
    model = EncoderDecoderModel.from_pretrained(override_model)    # 검증/테스트 모드 수행 (사용할 훈련결과 파일명을 주어야 함).

# 그래도 혹시 training 시에 이용할지 몰라서 model.config 에도 다음처럼 설정하자. ** 실제는 여기를 이용 안한다함!!
model.config.decoder_start_token_id = CLS_token_id
model.config.eos_token_id = SEP_token_id
model.config.pad_token_id = PAD_token_id
model.config.bos_token_id = CLS_token_id        # bos token id 도 설정해야 되나?
model.config.vocab_size = decoding_vocab_size   # 디코더 측의 토큰사전의 크기를 주어야 한다고 함.
#model.config.num_beams=4
#model.config.early_stopping=True

optimizer = AdamW(model.parameters(), lr=0.5e-5)
model = model.to(device)    # send model to cuda

if do_training == 1:
    # 훈련 (Training) 모드 수행

    min_avg_loss = 0     # 지금까지 수행한 epoch 들에서 달성한 배치당 평균 loss 중 최소값(초기값은 아무래도 좋음).
    log("Training starts.")
    start_time = time.perf_counter()
    for i in range(num_EPOCHS):
        total_loss = 0.0
        model.train()  # set to training mode.

        # 훈련 (한 에폭에 대하여).
        epoch_start_time = time.perf_counter()
        for j, batch in enumerate(dataloader):
            batch = tuple(t.to(device) for t in batch)  # send training batch to GPU.
            # b_input_ids, b_dec_input_ids, b_att_masks, b_dec_att_masks, b_labels = batch
            b_input_ids, b_att_masks, b_labels = batch
            optimizer.zero_grad()
            model.zero_grad()

            loss = model(input_ids=b_input_ids, attention_mask=b_att_masks, labels=b_labels).loss

            total_loss = total_loss + loss.item()

            loss.backward()
            optimizer.step()

            if j != 0 and j % 20 == 0:
                avg_loss = total_loss / (j+1)
                log(f"  Number of batches finished in an epoch of training = {j+1}, Avg loss = {avg_loss}")

        avg_loss_at_epoch = total_loss / num_batches
        epoch_end_time = time.perf_counter()
        epoch_elapse = epoch_end_time - epoch_start_time
        log(f" Finished Epoch : {i}.  its loss: {avg_loss_at_epoch}, ({timeformat(epoch_elapse)})")

        # Save model parameters into disk if there is improvement in average loss.
        # The method save will create a directory of given name by itself if it does not exist.
        if i == 0 or (i > 0 and avg_loss_at_epoch < min_avg_loss):      # 첫 epoch이거나, loss 의 향상이 있다면,
            log(f" saving the checkpoints in training at epoch={i}")
            model.save_pretrained(file_for_saving_parameters)
            min_avg_loss = avg_loss_at_epoch

    end_time = time.perf_counter()
    elapse = end_time - start_time
    log(f"훈련모드를 종료한다. ({timeformat(elapse)})")

elif do_training == 2:

    # 검증 (Validation) 단계 수행:
    model.eval()
    hit_cnt = 0
    total_cnt = 0
    print(f"{start_idx_file_to_read} ~ {end_idx_file_to_read}")
    log("검증 모드 수행")
    start_time = time.perf_counter()

    # 최근의 transformers 의 model.generate 메소드는 model.config 대신에 다음을 이용하여 동작한다고 함.
    generation_config = GenerationConfig (num_beams=4,do_sample=True,no_repeat_ngram_size=3,early_stopping=True,\
                            decoder_start_token=CLS_token_id,eos_token_id=SEP_token_id,\
                            pad_token_id=PAD_token_id,bos_token_id=CLS_token_id,\
                            max_new_tokens=Max_dec_seq,max_length=Max_dec_seq,min_length=1,\
                            num_return_sequences=1,length_penalty=1.0)

    for h, vbatch in enumerate(dataloader):
        vbatch = tuple(t.to(device) for t in vbatch)  # send validation batch to GPU.
        vb_input_ids, vb_att_masks, vb_labels = vbatch

        # 입력(어절문장의 token seq)에 대한 형태소문장의 token seq 를 생성한다.
        #generated = model.generate(input_ids=vb_input_ids, attention_mask=vb_att_masks, generation_config=generation_config)  # 잘 동작함.
        generated = model.generate(input_ids=vb_input_ids, generation_config=generation_config) # 이것도 잘 동작함.


        pred_decoder_id_seqs = generated.tolist()     # token id lists generated by the model

        # target_dec_id_seqs = vb_dec_input_ids.tolist()     # the corresponding token id lists of the target
        target_dec_id_seqs = vb_labels.tolist()

        num_sentences_generated_model = len(pred_decoder_id_seqs)
        num_sentences_in_target = len(target_dec_id_seqs)

        if num_sentences_generated_model != num_sentences_in_target:
            log(" Logic error: number of generations by model is wrong.")
            time.sleep(800)

        # 모델 출력과 target 모두 단위를 token id 에서 형태소로 변환한다.
        morpheme_seqs_model, morpheme_seqs_tgt = \
            get_morpheme_sequences_for_model_output(pred_decoder_id_seqs, target_dec_id_seqs, num_sentences_generated_model)

        # 모델의 형태소열과 target 의 형태소열을 문장마다 비교하여 맞춘 형태소 수를 구하여 축적한다.

        for k in range(num_sentences_generated_model):
            num_match = match_simple(morpheme_seqs_tgt[k], morpheme_seqs_model[k])  # 한 문장에서 맟춘 형태소 수
            hit_cnt += num_match    # 문장 내의 맞춘 형태소 수를 축적한다.
            total_cnt += len(morpheme_seqs_tgt[k])  # 문장의 형태소 수를 축적한다.

        recall_sofar = float(hit_cnt) / float(total_cnt)  # 지금까지 즉 batch h 까지의 recall 성능 계산.
        
        if h != 0 and h % 50 == 0:
            log(f" Validation has just finished for batch num = {h}. Recall so far = {recall_sofar}")

    recall = float(hit_cnt) / float(total_cnt)
    
    end_time = time.perf_counter()
    elapse = end_time - start_time
    log(f" Recall = {recall} ({timeformat(elapse)})")
    log("검증을 종료한다.")

elif do_training == 3:
    log("테스트 모드 수행")

    dataloader = DataLoader(data, sampler=SequentialSampler(data), batch_size=1, drop_last=True)
    batches = list(enumerate(dataloader))
    length_of_batches = len(dataloader)

    model.eval()
    generation_config = GenerationConfig (num_beams=4,do_sample=True,no_repeat_ngram_size=3,early_stopping=True,\
                            decoder_start_token=CLS_token_id,eos_token_id=SEP_token_id,\
                            pad_token_id=PAD_token_id,bos_token_id=CLS_token_id,\
                            max_new_tokens=Max_dec_seq,max_length=Max_dec_seq,min_length=1,\
                            num_return_sequences=1,length_penalty=1.0)

    user_input = None
    while user_input != -1 :
        user_input = input(f"배치번호({0}~{length_of_batches-1}) >> ")
        log(f"배치번호({0}~{length_of_batches-1}) >> {user_input}", False)
        
        start_time = time.perf_counter()
        if not user_input.isdigit() :
            user_input = -1
            continue

        user_input = int(user_input)
        if length_of_batches <= user_input :
            continue

        idx, batch = batches[user_input]
        batch = tuple(t.to(device) for t in batch)
        test_input_ids, test_att_masks, test_labels = batch

        generated = model.generate(input_ids=test_input_ids, attention_mask=test_att_masks, generation_config=generation_config)

        pred_decoder_id_seqs = generated.tolist()
        target_dec_id_seqs = test_labels.tolist()
        num_sentences_generated_model = len(pred_decoder_id_seqs)
        morpheme_seqs_model, morpheme_seqs_tgt = \
            get_morpheme_sequences_for_model_output(pred_decoder_id_seqs, target_dec_id_seqs, num_sentences_generated_model)

        num_match = match_simple(morpheme_seqs_tgt[0], morpheme_seqs_model[0])
        hit_cnt = num_match
        total_cnt = len(morpheme_seqs_tgt[0])
        recall = float(hit_cnt) / float(total_cnt)

        end_time = time.perf_counter()
        elapse = end_time - start_time

        log(f"   입력문장 : {e_sentences_str[user_input]}")
        log(f"   정답문장 : {morpheme_seqs_tgt[0]}")
        log(f"   모델문장 : {morpheme_seqs_model[0]}")
        log(f"   recall : {recall}")
        log(f"   ({timeformat(elapse)})\n")

else:
    log("수행모드 설정(do_training)이 잘못되어 있다. 1 or 2 or 3으로 설정하기 바람.")

fp_log.close()
log("program ends.")
log("...")
