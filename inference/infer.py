import os
import sys
sys.path.append('xcodec_mini_infer')
sys.path.append('xcodec_mini_infer/descriptaudiocodec')
import argparse
import torch
import numpy as np
import json
from omegaconf import OmegaConf
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf

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
from models.soundstream_hubert_new import SoundStream


parser = argparse.ArgumentParser()
parser.add_argument("--stage1_model", type=str, default="m-a-p/YuE-s1-7B-anneal-en-cot")
parser.add_argument("--stage2_model", type=str, default="m-a-p/YuE-s2-1B-general")
parser.add_argument("--max_new_tokens", type=int, default=3000)
parser.add_argument("--output_dir", type=str, default="./output")
parser.add_argument("--cuda_idx", type=int, default=0)
parser.add_argument("--use_audio_prompt", action="store_true")
parser.add_argument("--audio_prompt_path", type=str, default="")
parser.add_argument("--prompt_start_time", type=float, default=0.0)
parser.add_argument("--prompt_end_time", type=float, default=30.0)
parser.add_argument('--basic_model_config', default='./xcodec_mini_infer/final_ckpt/config.yaml', help='YAML files for configurations.')
parser.add_argument('--resume_path', default='./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth', help='Path to the model checkpoint.')


args = parser.parse_args()
if args.use_audio_prompt and not args.audio_prompt_path:
    raise FileNotFoundError("Please offer audio prompt filepath using '--audio_prompt_path', when you enable 'use_audio_prompt'!")
stage1_model = args.stage1_model
stage2_model = args.stage2_model
cuda_idx = args.cuda_idx
max_new_tokens = args.max_new_tokens
stage1_output_dir = os.path.join(args.output_dir, f"stage1")
stage2_output_dir = stage1_output_dir.replace('stage1', 'stage2')
os.makedirs(stage1_output_dir, exist_ok=True)
os.makedirs(stage2_output_dir, exist_ok=True)

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
model_config = OmegaConf.load(args.basic_model_config)
codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
parameter_dict = torch.load(args.resume_path, map_location='cpu')
codec_model.load_state_dict(parameter_dict['codec_model'])
codec_model.to(device)
codec_model.eval()

class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores

def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio

stage1_output_set = []
# Tips:
# genre tags support instrumental，genre，mood，vocal timbr and vocal gender
# all kinds of tags are needed
genres = "piano guitar jazz blues sad romantic bright vocal female"
lyrics = [
    "[verse]\nIn the quiet of the evening, shadows start to fall\nWhispers of the night wind echo through the hall\nLost within the silence, I hear your gentle voice\nGuiding me back homeward, making my heart rejoice\n", 
    "[chorus]\nDon't let this moment fade, hold me close tonight\nWith you here beside me, everything's alright\nCan't imagine life alone, don't want to let you go\nStay with me forever, let our love just flow\n", 
    ]
# intruction
full_lyrics = "\n".join(lyrics)
prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
prompt_texts += lyrics


random_id = uuid.uuid4()
output_seq = None
# Here is suggested decoding config
top_p = 0.93
temperature = 1.0
repetition_penalty = 1.2
# special tokens
start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
end_of_segment = mmtokenizer.tokenize('[end_of_segment]')
# Format text prompt
for i, p in enumerate(tqdm(prompt_texts)):
    section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
    guidance_scale = 1.5 if i <=1 else 1.2
    if i==0:
        continue
    if i==1:
        if args.use_audio_prompt:
            audio_prompt = load_audio_mono(args.audio_prompt_path)
            audio_prompt.unsqueeze_(0)
            with torch.no_grad():
                raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=0.5)
            raw_codes = raw_codes.transpose(0, 1)
            raw_codes = raw_codes.cpu().numpy().astype(np.int16)
            # Format audio prompt
            code_ids = codectool.npy2ids(raw_codes[0])
            audio_prompt_codec = code_ids[int(args.prompt_start_time *50): int(args.prompt_end_time *50)] # 50 is tps of xcodec
            audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [mmtokenizer.eoa]
            sentence_ids = mmtokenizer.tokenize("[start_of_reference]") +  audio_prompt_codec_ids + mmtokenizer.tokenize("[end_of_reference]")
            head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
        else:
            head_id = mmtokenizer.tokenize(prompt_texts[0])
        prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
    else:
        prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids

    prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device) 
    input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
    # Use window slicing in case output sequence exceeds the context of model
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

# save raw output and check sanity
ids = raw_output[0].cpu().numpy()
soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
if len(soa_idx)!=len(eoa_idx):
    raise ValueError(f'invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}')

vocals = []
instrumentals = []
range_begin = 1 if args.use_audio_prompt else 0
for i in range(range_begin, len(soa_idx)):
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
vocal_save_path = os.path.join(stage1_output_dir, f'cot_{genres}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_vocal_{random_id}'.replace('.', '@')+'.npy')
inst_save_path = os.path.join(stage1_output_dir, f'cot_{genres}_tp{top_p}_T{temperature}_rp{repetition_penalty}_maxtk{max_new_tokens}_instrumental_{random_id}'.replace('.', '@')+'.npy')
np.save(vocal_save_path, vocals)
np.save(inst_save_path, instrumentals)
stage1_output_set.append(vocal_save_path)
stage1_output_set.append(inst_save_path)


# offload model
model.cpu()
del model
torch.cuda.empty_cache()

print("Stage 2 inference...")
model_stage2 = AutoModelForCausalLM.from_pretrained(
    stage2_model, 
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
    )
model_stage2.to(device)
model_stage2.eval()

def stage2_generate(model, prompt, batch_size=16):
    codec_ids = codectool.unflatten(prompt, n_quantizer=1)
    codec_ids = codectool.offset_tok_ids(
                    codec_ids, 
                    global_offset=codectool.global_offset, 
                    codebook_size=codectool.codebook_size, 
                    num_codebooks=codectool.num_codebooks, 
                ).astype(np.int32)
    
    # Prepare prompt_ids based on batch size or single input
    if batch_size > 1:
        codec_list = []
        for i in range(batch_size):
            idx_begin = i * 300
            idx_end = (i + 1) * 300
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
    else:
        prompt_ids = np.concatenate([
            np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
            codec_ids.flatten(),  # Flatten the 2D array to 1D
            np.array([mmtokenizer.stage_2])
        ]).astype(np.int32)
        prompt_ids = prompt_ids[np.newaxis, ...]

    codec_ids = torch.as_tensor(codec_ids).to(device)
    prompt_ids = torch.as_tensor(prompt_ids).to(device)
    len_prompt = prompt_ids.shape[-1]
    
    block_list = LogitsProcessorList([BlockTokenRangeProcessor(0, 46358), BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)])

    # Teacher forcing generate loop
    for frames_idx in range(codec_ids.shape[1]):
        cb0 = codec_ids[:, frames_idx:frames_idx+1]
        prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
        input_ids = prompt_ids

        with torch.no_grad():
            stage2_output = model.generate(input_ids=input_ids, 
                min_new_tokens=7,
                max_new_tokens=7,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=block_list,
            )
        
        assert stage2_output.shape[1] - prompt_ids.shape[1] == 7, f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
        prompt_ids = stage2_output

    # Return output based on batch size
    if batch_size > 1:
        output = prompt_ids.cpu().numpy()[:, len_prompt:]
        output_list = [output[i] for i in range(batch_size)]
        output = np.concatenate(output_list, axis=0)
    else:
        output = prompt_ids[0].cpu().numpy()[len_prompt:]

    return output

def stage2_inference(model, stage1_output_set, stage2_output_dir, batch_size=4):
    stage2_result = []
    for i in tqdm(range(len(stage1_output_set))):
        output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))
        
        if os.path.exists(output_filename):
            print(f'{output_filename} stage2 has done.')
            continue
        
        # Load the prompt
        prompt = np.load(stage1_output_set[i]).astype(np.int32)
        
        # Only accept 6s segments
        output_duration = prompt.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6
        
        if num_batch <= batch_size:
            # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
            output = stage2_generate(model, prompt[:, :output_duration*50], batch_size=num_batch)
        else:
            # If num_batch is greater than batch_size, process in chunks of batch_size
            segments = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)

            for seg in range(num_segments):
                start_idx = seg * batch_size * 300
                # Ensure the end_idx does not exceed the available length
                end_idx = min((seg + 1) * batch_size * 300, output_duration*50)  # Adjust the last segment
                batch_size = batch_size if seg != num_segments-1 else num_batch % batch_size
                segment = stage2_generate(
                    model,
                    prompt[:, start_idx:end_idx],
                    batch_size=batch_size
                )
                segments.append(segment)

            # Concatenate all the segments
            output = np.concatenate(segments, axis=0)
        
        # Process the ending part of the prompt
        if output_duration*50 != prompt.shape[-1]:
            ending = stage2_generate(model, prompt[:, output_duration*50:], batch_size=1)
            output = np.concatenate([output, ending], axis=0)
        output = codectool_stage2.ids2npy(output)

        # Fix invalid codes (a dirty solution, which may harm the quality of audio)
        # We are trying to find better one
        fixed_output = copy.deepcopy(output)
        for i, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequant = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[i, j] = most_frequant
        # save output
        np.save(output_filename, fixed_output)
        stage2_result.append(output_filename)
    return stage2_result

stage2_result = stage2_inference(model_stage2, stage1_output_set, stage2_output_dir, batch_size=4)
print('Stage 2 DONE.\n')

# convert audio tokens to audio
def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)

tracks = []
for npy in stage2_result:
    codec_result = np.load(npy)
    decodec_rlt=[]
    decoded_waveform = codec_model.decode(torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device))
    decoded_waveform = decoded_waveform.cpu().squeeze(0)
    decodec_rlt.append(torch.as_tensor(decoded_waveform))
    decodec_rlt = torch.cat(decodec_rlt, dim=-1)
    save_path = os.path.splitext(npy)[0] + ".mp3"
    tracks.append(save_path)
    save_audio(decodec_rlt, save_path, 16000)

output_dir = args.output_dir
for inst_path in tracks:
    try:
        if (inst_path.endswith('.wav') or inst_path.endswith('.mp3')) \
            and 'instrumental' in inst_path:
            # find pair
            vocal_path = inst_path.replace('instrumental', 'vocal')
            if not os.path.exists(vocal_path):
                continue
            # mix
            mix_path = os.path.join(output_dir, os.path.basename(inst_path).replace('instrumental', 'mixed'))
            vocal_stem, sr = sf.read(inst_path)
            instrumental_stem, _ = sf.read(vocal_path)
            mix_stem = (vocal_stem + instrumental_stem) / 1
            sf.write(mix_path, mix_stem, sr)
    except Exception as e:
        print(e)