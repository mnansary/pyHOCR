# OCR for Handwritten Bangla Symbol Recognition 
    Version: 0.0.2  
    Author : Md. Nazmuddoha Ansary
    Python : 3.6.7  
![](/src_img/python.ico?raw=true )
![](/src_img/tensorflow.ico?raw=true)
![](/src_img/keras.ico?raw=true)

# Modules-
    numpy==1.16.2    
    scipy==1.2.1  
    cv2==3.4.5    
    tensorflow==1.13.1   
    keras==2.2.4
    scikit-learn==0.20.2
    scikit-image==0.14.2    

# Training
    usage: ocr_train.py [-h] datafolder  

    Bangla OCR 50 class Alphabet DenseNet Model -- Training Script  

    positional arguments:  
    datafolder  /path/to/Data/Folder  

    optional arguments:  
    -h, --help  show this help message and exit  

# Testing 
    usage: ocr_test.py [-h] datafolder model_path

    Bangla OCR 50 class Alphabet DenseNet Model -- Testing Script

    positional arguments:
    datafolder  /path/to/Data/Folder
    model_path  /path/to/test/Model.hdf5

    optional arguments:
    -h, --help  show this help message and exit

# Prediction (For Manually Taken images)
    usage: ocr_predict.py [-h] img_path model_path

    Bangla OCR 50 class Alphabet DenseNet Model -- Predictor Script

    positional arguments:
    img_path    /path/to/img/file
    model_path  /path/to/test/Model.hdf5

    optional arguments:
    -h, --help  show this help message and exit

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
    'ঁ' not printable :(

# H-OCR- PROJECT
![](/src_img/buet.ico?raw=true)