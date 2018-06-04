echo $3
python3 test_hw5.py SKGRAM test --cell LSTM --gpu_fraction 0.9 --dropout_rate 0.45 --loss_function categorical_crossentropy -emb_dim 300 --vocab_size 60000 --load_model SKGRAM --test_path $1 --result_path $2 

