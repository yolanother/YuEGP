
<p align="center">
    <img src="./assets/logo/白底.png" width="400" />
</p>

<p align="center">
    <strong>YuE-s1-7B-anneal-en-cot</strong> <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot">🤗</a>  &nbsp;|&nbsp; Demo <a href="https://m-a-p-ai.feishu.cn/wiki/OhpXwDcOsih6dakLcskcX7vEnXc">🎶</a> 
    <br>
    📑 <a href="">Paper</a>&nbsp;&nbsp;|&nbsp;&nbsp;📑 <a href="">Blog</a>
</p>

---

**YuE** is a groundbreaking series of open-source foundation models designed for music generation, specifically for transforming lyrics into full songs (**lyrics2song**). It can generate a complete song, lasting several minutes, that includes both a catchy vocal track and complementary accompaniment, ensuring a polished and cohesive result.

## News and Updates

* **2025.01.26 🔥**: We have released the **YuE** series.

<br>

## Requirements

Install dependencies with the following command:

`pip install requirements.txt`


## Quickstart

```
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://github.com/multimodal-art-projection/YuE.git
```

Here’s a quick guide to help you generate music with **YuE** using 🤗 Transformers. Before running the code, make sure your environment is properly set up, and that all dependencies are installed.

### Running the Script

In the following example, customize the `genres` and `lyrics` in the script, then execute it to generate a song with **YuE**.

```python
cd inference/
python infer.py \
    --stage1_model {STAGE1_MODEL_PATH} \
    --stage2_model {STAGE2_MODEL_PATH} \
    --output_dir ./output \
    --cuda_idx 0 \
    --max_new_tokens 3000
```
**Tips:**
1. `genres` should include details like instruments, genre, mood, vocal timbre, and vocal gender.
2. The length of `lyrics` segments and the `--max_new_tokens` value should be matched. For example, if `--max_new_tokens` is set to 3000, the maximum duration for a segment is around 30 seconds. Ensure your lyrics fit this time frame.
<br>


## License Agreement

<br>

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX

```
<br>
