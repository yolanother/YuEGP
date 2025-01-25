cd ./xcodec_mini_infer
python reconstruct.py --input ../output/stage2/max_new_tokens_3000/*_instrumental_*.npy --output ../output/stage2/max_new_tokens_3000/instrumental.mp3
python reconstruct.py --input ../output/stage2/max_new_tokens_3000/*_vocal_*.npy --output ../output/stage2/max_new_tokens_3000/vocal.mp3
python merge_mp3.py ../output/stage2/max_new_tokens_3000/instrumental.mp3 ../output/stage2/max_new_tokens_3000/vocal.mp3 ../output/stage2/max_new_tokens_3000/merged.mp3
cd ..
