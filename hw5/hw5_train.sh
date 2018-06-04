echo $2
python3 hw5.py NEW train --cell LSTM --gpu_fraction 0.9 --dropout_rate 0.45 --loss_function categorical_crossentropy -emb_dim 300 --vocab_size 60000 --train_path $1 

