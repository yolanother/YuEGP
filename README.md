<p align="center">
    <img src="./assets/logo/白底.png" width="400" />
</p>

<p align="center">
    <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot">YuE-s1-7B-anneal-en-cot 🤗</a> &nbsp;|&nbsp; <a href="https://m-a-p-ai.feishu.cn/wiki/OhpXwDcOsih6dakLcskcX7vEnXc">Demo 🎶</a> 
    <br>
    📑 <a href="">Paper</a>&nbsp;&nbsp;|&nbsp;&nbsp;📑 <a href="">Blog</a>
</p>

---
Our model's name is **YuE (乐)**. In Chinese, the word means "music" and "happiness." Some of you may find words that start with Yu hard to pronounce. If so, you can just call it "yeah." We wrote a song with our model's name.

<audio controls src="https://cdn-uploads.huggingface.co/production/uploads/6555e8d8a0c34cd61a6b9ce3/rG-ELxMyzDU7zH-inB9DV.mpga"></audio>

YuE is a groundbreaking series of open-source foundation models designed for music generation, specifically for transforming lyrics into full songs (lyrics2song). It can generate a complete song, lasting several minutes, that includes both a catchy vocal track and complementary accompaniment, ensuring a polished and cohesive result. YuE is capable of modeling diverse genres/vocal styles. Below are examples of songs in the pop and metal genres. For more styles, please visit the demo page.

Pop:Quiet Evening
<audio controls src="https://cdn-uploads.huggingface.co/production/uploads/640701cb4dc5f2846c91d4eb/gnBULaFjcUyXYzzIwXLZq.mpga"></audio>
Metal: Step Back
<audio controls src="https://cdn-uploads.huggingface.co/production/uploads/6555e8d8a0c34cd61a6b9ce3/kmCwl4GRS70UYDEELL-Tn.mpga"></audio>

## News and Updates

* **2025.01.26 🔥**: We have released the **YuE** series.

<br>

## Requirements

Python >=3.8 is recommended.

Install dependencies with the following command:

```
pip install -r requirements.txt
```

### **Important: Install FlashAttention 2**
For saving GPU memory, **FlashAttention 2 is mandatory**. Without it, large sequence lengths will lead to out-of-memory (OOM) errors, especially on GPUs with limited memory. Install it using the following command:
```
pip install flash-attn --no-build-isolation
```
Before installing FlashAttention, ensure that your CUDA environment is correctly set up. 
For example, if you are using CUDA 11.8:
- If using a module system:
```module load cuda11.8/toolkit/11.8.0 ```
- Or manually configure CUDA in your shell:
```
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
    source ~/.bashrc
```

---

## GPU Memory Usage and Sessions

YuE requires significant GPU memory for generating long sequences. Below are the recommended configurations:

- **For GPUs with 24GB memory or less**: Run **up to 2 sessions** concurrently to avoid out-of-memory (OOM) errors.
- **For full song generation** (many sessions, e.g., 4 or more): Use **GPUs with at least 80GB memory**. This can be achieved by combining multiple GPUs and enabling tensor parallelism.

To customize the number of sessions, the interface allows you to specify the desired session count. By default, the model runs **2 sessions** for optimal memory usage.

---

## Quickstart

```
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://github.com/multimodal-art-projection/YuE.git

cd YuE/inference/
git lfs install
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
```

Here’s a quick guide to help you generate music with **YuE** using 🤗 Transformers. Before running the code, make sure your environment is properly set up, and that all dependencies are installed.

### Running the Script

In the following example, customize the `genres` and `lyrics` in the script, then execute it to generate a song with **YuE**.

```python
cd inference/
python infer.py \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt prompt_examples/genre.txt \
    --lyrics_txt prompt_examples/lyrics.txt \
    --output_dir ./output \
    --cuda_idx 0 \
    --max_new_tokens 3000 
```

If you want to use audio prompt, enable `--use_audio_prompt`, and provide audio prompt:
```python
cd inference/
python infer.py \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt prompt_examples/genre.txt \
    --lyrics_txt prompt_examples/lyrics.txt \
    --output_dir ./output \
    --cuda_idx 0 \
    --max_new_tokens 3000 \
    --audio_prompt_path {YOUR_AUDIO_FILE} \
    --prompt_start_time 0 \
    --prompt_end_time 30 
```


---

### **Execution Time**
On an **H800 GPU**, one session takes **70–100 seconds**.  
On an **RTX 4090 GPU**, one session takes approximately 180 seconds** (replace with exact value).  

**Tips:**
1. `genres` should include details like instruments, genre, mood, vocal timbre, and vocal gender.
2. The length of `lyrics` segments and the `--max_new_tokens` value should be matched. For example, if `--max_new_tokens` is set to 3000, the maximum duration for a segment is around 30 seconds. Ensure your lyrics fit this time frame.
3. If using audio prompt，the duration around 30s will be fine.
---

## License Agreement



---

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX

```
<br>