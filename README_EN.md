<p align="left">
    <a href="README_CN.md">中文</a> &nbsp;|&nbsp; <b>English</b>&nbsp;&nbsp;
</p>
<br><br>

<p align="center">
    <img src="./assets/logo/白底.png" width="400" />
</p>

<p align="center">
    <strong>YuE-7B</strong> (coming soon) 🤗 &nbsp;|&nbsp; Demo <a href="#">🤗</a> 
    <br>
    📑 <a href="https://arxiv.org/abs/2407.10759">Paper</a>&nbsp;&nbsp;|&nbsp;&nbsp;📑 <a href="https://qwenlm.github.io/blog/qwen2-audio">Blog</a>
</p>

---

**YuE** is a groundbreaking series of open-source foundation models designed for music generation, specifically for transforming lyrics into full songs (**lyrics2song**). It can generate a complete song, lasting several minutes, that includes both a catchy vocal track and complementary accompaniment, ensuring a polished and cohesive result.

## Architecture

Below is an overview of the three-stage training process of **YuE**.

<p align="center">
    <img src="" width="80%" />
</p>

## News and Updates

* **2025.01.26 🔥**: We have released the **YuE** series.

<br>

## Requirements

Install dependencies with the following command:

`pip install requirements.txt`


## Quickstart

Here are some simple examples to help you get started with **YuE** using 🤗 Transformers. Before running the code, ensure your environment is set up and the required packages are installed. Verify that you meet the above requirements, and install the necessary libraries.

#### Stage 1: Lyrics-Chain-of-Thoughts

The data format for Stage 1 follows this structure:

instruction + genre + full lyrics + lyrics of section1 + audio tokens of section1 + lyrics of section2 + audio tokens of section2 + ...

In Stage 1, the model generates tokens for each section sequentially, with each new section based on the previous sections and the instruction.

#### Stage 2: Audio Augmentation

Stage 2 data consists of a 6-second audio segment, which is designed as:
<SOA><stage_1>...6 seconds of codebook 0...<stage_2>...6 seconds of codebook 0-7 flattened...<EOA>

# Scripts
Here is an demostration.
```python
# stage1 and stage2 generation
python inference/infer.py \
    --stage1_model {STAGE1_MODEL_PATH} \
    --stage2_model {STAGE2_MODEL_PATH} \
    --output_dir ./output \
    --cuda_idx 0 \
    --max_new_tokens 3000 # this depends on the length of lyrics of each seciton

# audio tokens reconstruct to audio
cd ./xcodec_mini_infer
python reconstruct.py --input ../output/stage2/*_instrumental_*.npy --output ../output/stage2/instrumental.mp3
```
<br>


## License Agreement

<br>

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX

```
<br>
