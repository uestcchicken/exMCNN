# Extended Multi-Column Convolutional Neural Network for Crowd Counting


Implementation of PCM 2018 paper  
[Extended Multi-Column Convolutional Neural Network for Crowd Counting](https://link.springer.com/chapter/10.1007/978-3-030-00764-5_49)  
using **tensorflow**

### installation

1. Install tensorflow
2. git clone
3. download VGG16 model file (vgg16.npy) from https://github.com/machrisaa/tensorflow-vgg

### data setup

All the data setup process follows the pytorch version implementation:   
[svishwa/crowdcount-mcnn](https://github.com/svishwa/crowdcount-mcnn#data-setup)

### train

tensorflow:  
run ```python3 train.py A(or B)```  
model is saved to modelA/ or modelB/

### test 

run ```python3 test.py A(or B)```  

(uncomment code containing heatmap in network.py to generate heatmap)

### heatmap

actual: 1110  
![](sample/A/heat_A_2_act_1110.png)  
predicted: 1181  
![](sample/A/heat_A_2_pre_1181.png)


