cd src
# train
python main.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --batch_size 16 --master_batch 7 --num_epochs 70 --lr_step 45,60 --gpus 0,1

python main.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --batch_size 16 --master_batch 7 --num_epochs 70 --lr_step 45,60 --gpus 0,1  --arch resFP_18
# test
python test.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --resume

CUDA_VISIBLE_DEVICES=3 python test.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --load_model ../models/model_180.pth --arch resFP_18 --gpus 3
cd ..

# 也ok?
python main.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --batch_size 16 --master_batch 9 --num_epochs 70 --lr_step 45,60 --gpus 1,2  --arch resFP_18 --resume

# 默认gpu 0
python main.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --batch_size 4  --num_epochs 70 --lr_step 45,60 --arch resFP_18

# 单GPU 可以CUDA_VISIBLE_DEVICES，--gpu同时指定
CUDA_VISIBLE_DEVICES=1 python main.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --batch_size 4  --num_epochs 70 --lr_step 45,60 --gpu 1 --arch resFP_18

# 多GPU可！
CUDA_VISIBLE_DEVICES=1,2 python main.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --batch_size 16 --master_batch 9  --num_epochs 70 --lr_step 45,60 --gpu 1,2 --arch resFP_18
# 多epoch
CUDA_VISIBLE_DEVICES=1,2 python main.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --batch_size 16 --master_batch 9  --num_epochs 300 --lr 0.0002 --lr_step 150,180 --gpu 1,2 --arch resFP_18 --resume