name='DML'  
data_dir="../../dataset/University-Release/train"
# data_dir='../../dataset/SUES-200-512x512/Dataset/Training/150' # SUES-200
test_dir="../../dataset/University-Release/test"
# test_dir='../../dataset/SUES-200-512x512/Dataset/Testing/150' # SUES-200
gpu_ids="0"
lr=0.01
batchsize=8
triplet_loss=0.3
num_epochs=200
views=2

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --views $views --lr $lr \
 --batchsize $batchsize --triplet_loss $triplet_loss --epochs $num_epochs \

# for ((j = 1; j < 3; j++));
#     do
#       python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --mode $j
#     done
