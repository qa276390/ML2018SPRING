sed -i '8d' Predict.py
wget https://www.dropbox.com/s/qiqkdxdr945ecu6/0679.h5?dl=1
mv 0679.h5?dl=1 best.h5
python3 Predict.py $1 $2 $3

