import os
import argparse
import torch
import numpy as np
import json

import uuid
from tqdm import tqdm
from einops import rearrange
from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import glob
import time
import copy
from collections import Counter


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        # Set the logits of the blocked tokens to a large negative value
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores

parser = argparse.ArgumentParser()
parser.add_argument("--stage1_model", type=str, default="./stage1.exp31.8.30B.hf_ckpt")
parser.add_argument("--stage2_model", type=str, default="./stage2.exp32.2.927B.hf_ckpt")
parser.add_argument("--output_dir", type=str, default="./output")
parser.add_argument("--cuda_idx", type=int, default=0)
parser.add_argument("--max_new_tokens", type=int, default=3000)


args = parser.parse_args()
stage1_model = args.stage1_model
stage2_model = args.stage2_model
cuda_idx = args.cuda_idx
max_new_tokens = args.max_new_tokens
stage1_path = os.path.join(args.output_dir, f"stage1/max_new_tokens_{max_new_tokens}")
stage2_path = stage1_path.replace('stage1', 'stage2')
text_prompt_path = os.path.join(args.output_dir, 'text_prompt')
os.makedirs(stage1_path, exist_ok=True)
os.makedirs(stage2_path, exist_ok=True)
os.makedirs(text_prompt_path, exist_ok=True)

# load tokenizer and model
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
model = AutoModelForCausalLM.from_pretrained(
    stage1_model, 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # To enable flashattn, you have to install flash-attn
    )   
# to device, if gpu is available
model.to(device)
model.eval()


codectool = CodecManipulator("xcodec", 0, 1)
codectool_stage2 = CodecManipulator("xcodec", 0, 8)


stage1_output_set = []
# lyrics and genres
genres = "female blues airy vocal bright vocal piano sad romantic guitar jazz"
lyrics = ["[verse]\nIn the quiet of the evening, shadows start to fall\nWhispers of the night wind echo through the hall\nLost within the silence, I hear your gentle voice\nGuiding me back homeward, making my heart rejoice\n", "[chorus]\nDon't let this moment fade, hold me close tonight\nWith you here beside me, everything's alright\nCan't imagine life alone, don't want to let you go\nStay with me forever, let our love just flow\n", "[verse]\nMoonlight paints a picture upon your lovely face\nEvery glance between us fills the empty space\nTime stands still around us when you're in my arms\nNothing else can matter, safe from any harm\n", "[chorus]\nDon't let this moment fade, hold me close tonight\nWith you here beside me, everything's alright\nCan't imagine life alone, don't want to let you go\nStay with me forever, let our love just flow\n", "[bridge]\nEvery touch ignites a fire, burning deep within\nEvery smile you give to me makes my head spin\nPromise me you'll stay awhile, don't ever say goodbye\nTogether we'll chase every star across the sky\n", "[chorus]\nDon't let this moment fade, hold me close tonight\nWith you here beside me, everything's alright\nCan't imagine life alone, don't want to let you go\nStay with me forever, let our love just flow\n", "[outro]\nStay with me forever, let our love just flow\n"], "audio_length_in_sec": 329.9526530612245, "vocals_codec": "codeclm/launcher/scripts/pretrain/exp28/infer_msa_prompt/audio_prompt_codec/o6_argRl3r0.Vocals_xcodec_16k_0_compressed.npy", "instrumental_codec": "codeclm/launcher/scripts/pretrain/exp28/infer_msa_prompt/audio_prompt_codec/o6_argRl3r0.Instrumental_xcodec_16k_0_compressed.npy", "lyrics_sections": ["In the quiet of the evening, shadows start to fall\nWhispers of the night wind echo through the hall\nLost within the silence, I hear your gentle voice\nGuiding me back homeward, making my heart rejoice", "Don't let this moment fade, hold me close tonight\nWith you here beside me, everything's alright\nCan't imagine life alone, don't want to let you go\nStay with me forever, let our love just flow", "Moonlight paints a picture upon your lovely face\nEvery glance between us fills the empty space\nTime stands still around us when you're in my arms\nNothing else can matter, safe from any harm", "Don't let this moment fade, hold me close tonight\nWith you here beside me, everything's alright\nCan't imagine life alone, don't want to let you go\nStay with me forever, let our love just flow", "Every touch ignites a fire, burning deep within\nEvery smile you give to me makes my head spin\nPromise me you'll stay awhile, don't ever say goodbye\nTogether we'll chase every star across the sky", "Don't let this moment fade, hold me close tonight\nWith you here beside me, everything's alright\nCan't imagine life alone, don't want to let you go\nStay with me forever, let our love just flow", "Stay with me forever, let our love just flow"]
prompt_texts = [f'Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{"\n".join(lyrics)}']
prompt_texts += lyrics

try:
    random_id = uuid.uuid4()
    output_seq = None
    # decoding config
    top_p = 0.93
    temperature = 1.0
    repetition_penalty = 1.2
    # special tokens
    start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
    end_of_segment = mmtokenizer.tokenize('[end_of_segment]')
    # Generate
    for i, p in enumerate(prompt_texts):
        section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
        guidance_scale = 1.5 if i <=1 else 1.2
        if i==0:
            continue
        if i==1:
            head_id = mmtokenizer.tokenize(prompt_texts[0])
            prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
        else:
            prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids

        prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device) 
        input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
        # generate
        max_context = 16384-max_new_tokens-1
        if input_ids.shape[-1] > max_context:
            print(f'Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, now using the last {max_context} tokens.')
            input_ids = input_ids[:, -(max_context):]
        with torch.no_grad():
            output_seq = model.generate(
                input_ids=input_ids, 
                max_new_tokens=max_new_tokens, 
                min_new_tokens=100, 
                do_sample=True, 
                top_p=top_p,
                temperature=temperature, 
                repetition_penalty=repetition_penalty, 
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=LogitsProcessorList([BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)]),
                guidance_scale=guidance_scale,
                )
            if output_seq[0][-1].item() != mmtokenizer.eoa:
                tensor_eoa = torch.as_tensor([[mmtokenizer.eoa]]).to(model.device)
                output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
        if i > 1:
            raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1]:]], dim=1)
        else:
            raw_output = output_seq
        ids = raw_output[0].cpu().numpy()
        soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
        eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
        if len(soa_idx)!=len(eoa_idx):
            raise ValueError(f'section {i}: invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}\n')

    # save raw output and check sanity
    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
    if len(soa_idx)!=len(eoa_idx):
        raise ValueError(f'invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}')

    vocals = []
    instrumentals = []
    for i in range(0, len(soa_idx)):
        codec_ids = ids[soa_idx[i]+1:eoa_idx[i]]
        if codec_ids[0] == 32016:
            codec_ids = codec_ids[1:]
        codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]
        vocals_ids = codectool.ids2npy(rearrange(codec_ids,"(n b) -> b n", b=2)[0])
        vocals.append(vocals_ids)
        instrumentals_ids = codectool.ids2npy(rearrange(codec_ids,"(n b) -> b n", b=2)[1])
        instrumentals.append(instrumentals_ids)
    vocals = np.concatenate(vocals, axis=1)
    instrumentals = np.concatenate(instrumentals, axis=1)
    vocal_save_path = os.path.join(stage1_path, f'cot_{genres}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_vocal_{random_id}'.replace('.', '@')+'.npy')
    inst_save_path = os.path.join(stage1_path, f'cot_{genres}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_instrumental_{random_id}'.replace('.', '@')+'.npy')
    np.save(vocal_save_path, vocals)
    np.save(inst_save_path, instrumentals)
    stage1_output_set.append(vocal_save_path)
    stage1_output_set.append(inst_save_path)

except AssertionError as e:
    print(e)
    error_index = sorted(np.where(codec_ids < 45334)[0].tolist() + np.where(codec_ids >= 46358)[0].tolist())
    np.save(os.path.join(error_output_path, f'cot_{data_idx}_codec_id_assertion_error_{random_id}.npy'), codec_ids)
    if len(error_index) > 0:
        print('error code:', codec_ids[error_index[0]])
except ValueError as e:
    print(e)


# ############################### STAGE 2 ##############################
print("Stage 2 inference...")
model = AutoModelForCausalLM.from_pretrained(
    stage2_model, 
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
    )
model.to(device)
model.eval()

def stage2_inference(model_stage2, prompt, batch_size=5):
    codec_ids = codectool.unflatten(prompt, n_quantizer=1)
    codec_ids = codectool.offset_tok_ids(
                    codec_ids, 
                    global_offset=codectool.global_offset, 
                    codebook_size=codectool.codebook_size, 
                    num_codebooks=codectool.num_codebooks, 
                ).astype(np.int32)

    # split prompt then parrallel generate
    codec_list = []
    for i in range (batch_size):
        idx_begin = i*300
        idx_end = (i+1)*300
        codec_list.append(codec_ids[:, idx_begin:idx_end])

    codec_ids = np.concatenate(codec_list, axis=0)
    prompt_ids = np.concatenate(
        [
            np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_size, 1)),
            codec_ids,
            np.tile([mmtokenizer.stage_2], (batch_size, 1)),
        ],
        axis=1
    )
    codec_ids = torch.as_tensor(codec_ids).to(device)
    prompt_ids = torch.as_tensor(prompt_ids).to(device)
    len_prompt = prompt_ids.shape[1]
    block_list = LogitsProcessorList([BlockTokenRangeProcessor(0, 46358), BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)])
    # teacher forcing generate
    for frames_idx in range(codec_ids.shape[1]):
        cb0 = codec_ids[:, frames_idx: frames_idx+1]
        prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
        input_ids = prompt_ids

        with torch.no_grad():
            stage2_output = model_stage2.generate(input_ids=input_ids, 
                min_new_tokens=7,
                max_new_tokens=7,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=block_list,
                )
        
        assert stage2_output.shape[1]-prompt_ids.shape[1] == 7, f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
        prompt_ids = stage2_output
    # save output
    output = prompt_ids.cpu().numpy()[:, len_prompt:]
    output_list = [output[i] for i in range(batch_size)]
    output = np.concatenate(output_list, axis=0)

    # np.save(os.path.join(output_path, f"stage2_output_{idx}.npy"), codectool_stage2.ids2npy(output))
    return output

def stage2_inference_ending(model_stage2, prompt):
    codec_ids = codectool.unflatten(prompt, n_quantizer=1)
    codec_ids = codectool.offset_tok_ids(
                    codec_ids, 
                    global_offset=codectool.global_offset, 
                    codebook_size=codectool.codebook_size, 
                    num_codebooks=codectool.num_codebooks, 
                ).astype(np.int32)

    prompt_ids = np.concatenate([
        np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
        codec_ids.flatten(),  # Flatten the 2D array to 1D
        np.array([mmtokenizer.stage_2])
    ]).astype(np.int32)
    codec_ids = torch.as_tensor(codec_ids).to(device)
    prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
    len_prompt = prompt_ids.shape[-1]
    # teacher forcing generate
    block_list = LogitsProcessorList([BlockTokenRangeProcessor(0, 46358), BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)])
    for frames_idx in range(codec_ids.shape[1]):
        cb0 = codec_ids[:, frames_idx: frames_idx+1]
        prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
        with torch.no_grad():
            stage2_output = model_stage2.generate(input_ids=prompt_ids, 
                min_new_tokens=7,
                max_new_tokens=7,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=block_list,
                )
        
        assert stage2_output.shape[1]-prompt_ids.shape[1] == 7, f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
        prompt_ids = stage2_output
    # save output
    output = prompt_ids[0].cpu().numpy()[len_prompt:]
    # np.save(os.path.join(output_path, f"stage2_output_{idx}.npy"), codectool_stage2.ids2npy(output))
    return output

for i in tqdm(range(len(stage1_output_set))):
    start_time = time.time()
    output_filename = os.path.join(stage2_path, os.path.basename(stage1_output_set[i]))
    if os.path.exists(output_filename):
        print(f'{output_filename} stage2 has done.')
        continue
    prompt = np.load(stage1_output_set[i]).astype(np.int32)
    # only accept 6s segments
    output_duration = prompt.shape[-1]//50//6 * 6
    batch_size = output_duration//6
    if batch_size <= 16:
        # if batch_size is less than 16, prompt can be infer at once
        output = stage2_inference(model, prompt[:, :output_duration*50], batch_size=batch_size)
    else:
        times = 1
        while batch_size > 16:
            batch_size //= 2
            times *= 2
        segments=[]
        for seg in range(times):
            segment = stage2_inference(model, prompt[:, seg*batch_size*300: (seg+1)*batch_size*300], batch_size=batch_size)
            segments.append(segment)
        output = np.concatenate(segments, axis=0)
    ending = stage2_inference_ending(model, prompt[:, output_duration*50:])
    print(output.shape, ending.shape)
    output = np.concatenate([output, ending], axis=0)
    output = codectool_stage2.ids2npy(output)

    # fix invalid codes
    x2 = copy.deepcopy(output)
    fix = False
    # print(output)
    for i, line in enumerate(output):
        for j, element in enumerate(line):
            if element < 0 or element > 1023:
                counter = Counter(line)
                most_frequant = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                x2[i, j] = most_frequant
                fix = True
    # save output
    if fix:
        np.save(output_filename, x2)
    else:
        np.save(output_filename, output)
print('Stage 2 DONE.\n')
