# my implementation of RTM3d based on CenterNet
I have to emphasize that the precision is low and you'd better refer to official code for good performance.  

## Environment
Python3.6+ torch 0.4.1 following the centernet.
Link-->[env prepare](https://github.com/xingyizhou/CenterNet/blob/master/readme/INSTALL.md)

## Installation

First,you need to prepare data like the centernet do.
Link-->[kitti data prepare](https://github.com/xingyizhou/CenterNet/blob/master/readme/DATA.md)

In order to compile src/lib/utils/energy.cpp,you need to install pybind11.
Link-->[pybind11](https://github.com/pybind/pybind11)

~~~
cd ~/Project/RTM/src/lib/utils

make 
~~~
 ~/PScanning dependencies of target energy
[ 50%] Building CXX object CMakeFiles/energy.dir/energy.cpp.o
[100%] Linking CXX shared module energy.cpython-36m-x86_64-linux-gnu.so
[100%] Built target energyroject/RTM/src/lib/utils$ make
Then test if you have install energy module successfully.
~~~

(CenterNet) kaixin1@213c6db174e2:~/Project/RTM/src/lib/utils$ python
Python 3.6.10 |Anaconda, Inc.| (default, Jan  7 2020, 21:14:29) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import energy
>>> 
~~~

##Train
~~~
python main.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --batch_size 4  --num_epochs 70 --lr_step 45,60 --arch resFP_18
~~~

##Test
~~~
CUDA_VISIBLE_DEVICES=0 python test.py ddd --exp_id 3dop --dataset kitti --kitti_split 3dop --load_model ../models/model_180.pth --arch resFP_18 --gpus 3
~~~
