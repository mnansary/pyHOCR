# Handwritten Bangla Symbol Recognition with DenseNet
    Version: 0.0.3  
    Author : Md. Nazmuddoha Ansary
    Python : 3.6.8  
![](/info/src_img/python.ico?raw=true )
![](/info/src_img/tensorflow.ico?raw=true)
![](/info/src_img/keras.ico?raw=true)
![](/info/src_img/col.ico?raw=true)
![](/info/src_img/buet.ico?raw=true)

# Symbol List
    'অ','আ','ই','ঈ','উ','ঊ',  
    'ঋ','এ','ঐ','ও','ঔ',  
    'ক','খ','গ','ঘ','ঙ',  
    'চ','ছ','জ','ঝ','ঞ',  
    'ট','ঠ','ড','ঢ','ণ',  
    'ত','থ','দ','ধ','ন',  
    'প','ফ','ব','ভ','ম',  
    'য','র','ল',  
    'শ','ষ','স','হ',  
    'ড়','ঢ়','য়',  
    'ৎ','ং','ঃ','ঁ'  
    'ঁ'  

*   'ঁ' *is not printable*   

# DenseNet 
The model is based on the original paper:[Densely Connected Convolutional Networks](https://ieeexplore.ieee.org/document/8099726)  
> Authors and Researchers: Gao Huang ; Zhuang Liu ; Laurens van der Maaten ; Kilian Q. Weinberger

The paper introduces **Dense Blocks** within the traditional convolutional neural network architechture.  
![](/info/dense1.png?raw=true)
The composite layers can also contain **bottoleneck layers**   
![](/info/dense2.png?raw=true)

As compared to well established CNN models (like : *FractNet* or *ResNet*) DenseNet has:  
    *   Less number of feature vector  
    *   Low information bottoleneck   
    *   Better Handling Of the *vanishing gradient* problem      

# Database:
[CMATERdb](https://code.google.com/archive/p/cmaterdb/)
> CMATERdb 3.1.2: Handwritten Bangla basic-character database  
### Data Sample
![](/info/cm.png?raw=true)
### Established Results
From:[Alom et. al. 2018](https://www.hindawi.com/journals/cin/2018/6747098/)
![](/info/alom1.png?raw=true)

# Version and Requirements
    Keras==2.2.5  
    numpy==1.16.4  
    tensorflow==1.13.1  
* *pip3 install -r requirements.txt*
# Colab and TPU(Tensor Processing Unit)
*TPU’s have been recently added to the Google Colab portfolio making it even more attractive for quick-and-dirty machine learning projects when your own local processing units are just not fast enough. While the **Tesla K80** available in Google Colab delivers respectable **1.87 TFlops** and has **12GB RAM**, the **TPUv2** available from within Google Colab comes with a whopping **180 TFlops**, give or take. It also comes with **64 GB** High Bandwidth Memory **(HBM)**.*
[Visit This For More Info](https://medium.com/@jannik.zuern/using-a-tpu-in-google-colab-54257328d7da)
**For this model the approx time/epoch=24s**
> Test data Prediction Accuracy [F1 accuracy]: 98.56666666666666
# Implemented Model Architechture
![](/info/model.png?raw=true)
