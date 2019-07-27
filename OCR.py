from flask import Flask,render_template,request,url_for
from binascii import a2b_base64
app=Flask(__name__)


def convertImgData(imgData):
   
    image_data=str(imgData).split(',')[1]
    print(image_data)
    binary_data = a2b_base64(image_data)
    print(binary_data)
    with open('data.png','wb') as data:
        data.write(binary_data)


@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict/',methods=['GET','POST'])

def predict():
    string_res="YES"
    imgData=request.get_data()
    convertImgData(imgData)
    print('Img Saved')
    return string_res



if __name__=='__main__':

    app.run(debug=True)
