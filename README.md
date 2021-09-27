# CoulombDet

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
 

## Projects
![0](projects.png)


**Note:**    
- Single GPU training: [SAVE_WEIGHTS_INTE](./libs/configs/cfgs.py) = iter_epoch * 1 
- Multi-GPU training (**better**): [SAVE_WEIGHTS_INTE](./libs/configs/cfgs.py) = iter_epoch * 2

## My Development Environment
**docker images: yangxue2docker/py3-tf1.15.2-nv-torch1.8.0-cuda11:v1.0**        
1. python3.5 (anaconda recommend)               
2. cuda 10.0                     
3. opencv-python 4.1.1.26         
4. [tfplot 0.2.0](https://github.com/wookayin/tensorflow-plot) (optional)            
5. tensorflow-gpu 1.13
6. tqdm 4.54.0
7. Shapely 1.7.1

**Note: For 30xx series graphics cards, I recommend this [blog](https://blog.csdn.net/qq_39543404/article/details/112171851) to install tf1.xx, or refer to [ngc](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow) and [tensorflow-release-notes](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_20-11.html#rel_20-11) to download docker image according to your environment, or just use my docker image (yangxue2docker/py3-tf1.15.2-nv-torch1.8.0-cuda11:v1.0)**

## Download Model
### Pretrain weights
Download a pretrain weight you need from the following three options, and then put it to $PATH_ROOT/dataloader/pretrained_weights. 
1. MxNet pretrain weights **(recommend in this repo, default in [NET_NAME](/Users/yangxue/Desktop/yangxue/code/RotationDetection/libs/configs/_base_/models/retinanet_r50_fpn.py))**: resnet_v1d, resnet_v1b, refer to [gluon2TF](./thirdparty/gluon2TF/README.md).    
* [Baidu Drive (5ht9)](https://pan.baidu.com/s/1GpqKg0dOaaWmwshvv1qWGg)          
* [Google Drive](https://drive.google.com/drive/folders/1BM8ffn1WnsRRb5RcuAcyJAHX8NS2M1Gz?usp=sharing)  
2. Tensorflow pretrain weights: [resnet50_v1](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz), [resnet101_v1](http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz), [resnet152_v1](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz), [efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), [mobilenet_v2](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz), darknet53 ([Baidu Drive (1jg2)](https://pan.baidu.com/s/1p8V9aaivo9LNxa_OjXjUwA), [Google Drive](https://drive.google.com/drive/folders/1zyg1bvdmLxNRIXOflo_YmJjNJdpHX2lJ?usp=sharing)).      
3. Pytorch pretrain weights, refer to [pretrain_zoo.py](./dataloader/pretrained_weights/pretrain_zoo.py) and [Others](./OTHERS.md).
* [Baidu Drive (oofm)](https://pan.baidu.com/s/16nHwlkPsszBvzhMv4h2IwA)          
* [Google Drive](https://drive.google.com/drive/folders/14Bx6TK4LVadTtzNFTQj293cKYk_5IurH?usp=sharing)      


### AWS Instance

1. Create an EC2 Instance with at least 2 GPUs and 220GB of Elastic Block Storage. I used the g4dn.12xlarge instance type.
I also added IAM role so I could access all S3 storage instances using the AWS CLI from within my EC2 instance.

2. Use the following security group settings using your IP instead:

3. Generate a public-private key pair so you can easily SHH into your instance.

4. Once you are inside your EC2 instance, clone this repository.

### Docker Container

After you have setup your AWS instance execute the following
    ```
    docker run --gpus all -it -v /home/ubuntu/qdot-detector/:/workspace/qdot-detector -p 6007:6007 yangxue2docker/py3-tf1.15.2-nv-torch1.8.0-cuda11:v1.0
    ```
    This downloads the docker container and starts it.
    
 You may find the following commands useful
 
    ```
    1. To see the IDs of all of the containers you have created (currently running or not): docker ps -aq
    2. To see the current running containers: docker container ls
    3. To attach to a container: docker attach [container-id]
    4. To start a container that has stopped running type: docker start [container-id]
    5. To exit a container without stopping it press CTRL+p then CTRL+q

    ```
 

### Trained weights
1. Please download trained models by this project, then put them to $PATH_ROOT/output/pretained_weights.
2. To get the weights for the R2CNN architecture, navigate to $PATH_ROOT/output and simply execute 
    ```sudo aws s3 sync s3://data-deep-learning-qmt/R2CNN-Output/ ./ ``` Perform this command ouside of the docker container since it does not 
    have the AWS CLI installed.

## Compile
    ```  
    cd $PATH_ROOT/libs/utils/cython_utils
    rm *.so
    rm *.c
    rm *.cpp
    python setup.py build_ext --inplace (or make)
    
    cd $PATH_ROOT/libs/utils/
    rm *.so
    rm *.c
    rm *.cpp
    python setup.py build_ext --inplace
    ```

## Train 

1. If you want to train your own dataset, please note:  
    ```(1) Select the detector and dataset you want to use, and mark them as #DETECTOR and #DATASET (such as #DETECTOR=retinanet and #DATASET=DOTA)
    (2) Modify parameters (such as CLASS_NUM, DATASET_NAME, VERSION, etc.) in $PATH_ROOT/libs/configs/#DATASET/#DETECTOR/cfgs_xxx.py
    (3) Copy $PATH_ROOT/libs/configs/#DATASET/#DETECTOR/cfgs_xxx.py to $PATH_ROOT/libs/configs/cfgs.py
    (4) Add category information in $PATH_ROOT/libs/label_name_dict/label_dict.py     
    (5) Add data_name to $PATH_ROOT/dataloader/dataset/read_tfrecord.py```     
    For the QDOT dataset we use #DATASET as QDOT.
2. To use the QDOT dataset navigate to $PATH_ROOT/dataloader/dataset/QDOT and call the following
    ```sudo aws s3 sync s3://data-deep-learning-qmt/QDOT/ ./``` this must be performed outside of the container since the docker instance does not have
    the AWS CLI installed.

2. Make tfrecord       
    
    This step needs to be performed in the docker container.
    
    If image is very large (such as DOTA dataset), the image needs to be cropped. Take DOTA dataset as a example:      
    ```  
    cd $PATH_ROOT/dataloader/dataset/DOTA
    python data_crop.py
    ```  
    If image does not need to be cropped, just convert the annotation file into xml format, refer to [example.xml](./example.xml).
    ```  
    cd $PATH_ROOT/dataloader/dataset/  
    python convert_data_to_tfrecord.py --root_dir='/PATH/TO/DOTA/' 
                                       --xml_dir='labeltxt'
                                       --image_dir='images'
                                       --save_name='train' 
                                       --img_format='.png' 
                                       --dataset='DOTA'
    ```  
    The images for the QDOT dataset do not need to be cropped so you can execute the following
    ```
    python convert_data_to_tfrecord.py --root_dir='/workspace/qdot-detector/dataloader/dataset/QDOT/train/' --xml_dir='labeltxt' --image_dir='images' --save_name='train' --img_format='.png' --dataset='QDOT'
    ```
    

3. Start training
    ```  
    cd $PATH_ROOT/tools/#DETECTOR
    python train.py
    ```

## Test
1. For large-scale image, take DOTA dataset as a example (the output file or visualization is in $PATH_ROOT/tools/#DETECTOR/test_dota/VERSION): 
    ```  
    cd $PATH_ROOT/tools/#DETECTOR
    python test_dota.py --test_dir='/PATH/TO/IMAGES/'  
                        --gpus=0,1,2,3,4,5,6,7  
                        -ms (multi-scale testing, optional)
                        -s (visualization, optional)
    
    or (better than multi-scale testing)
    
    python test_dota_sota.py --test_dir='/PATH/TO/IMAGES/'  
                             --gpus=0,1,2,3,4,5,6,7  
                             -s (visualization, optional)
    ``` 

    **Notice: In order to set the breakpoint conveniently, the read and write mode of the file is' a+'. If the model of the same #VERSION needs to be tested again, the original test results need to be deleted.**

2. For small-scale image, take HRSC2016 dataset as a example: 
    ```  
    cd $PATH_ROOT/tools/#DETECTOR
    python test_hrsc2016.py --test_dir='/PATH/TO/IMAGES/'  
                            --gpu=0
                            --image_ext='bmp'
                            --test_annotation_path='/PATH/TO/ANNOTATIONS'
                            -s (visualization, optional)
    ``` 
3. For the QDOT dataset, we can call
    ```
    python test_qdot.py --img_dir='/workspace/qdot-detector/dataloader/dataset/QDOT/test/images' --image_ext='.png' --test_annotation_path='/workspace/qdot-detector/dataloader/dataset/QDOT/test/labeltxt' --gpu='0' -s -ms
    ```

## Tensorboard
```  
cd $PATH_ROOT/output/summary
tensorboard --logdir=. --port 6007
``` 
This opens tensorboard on port 6007 for monitoring our training progress.
![1](images.png)

![2](scalars.png)


## Reference
1、https://github.com/endernewton/tf-faster-rcnn   
2、https://github.com/zengarden/light_head_rcnn   
3、https://github.com/tensorflow/models/tree/master/research/object_detection    
4、https://github.com/fizyr/keras-retinanet     


