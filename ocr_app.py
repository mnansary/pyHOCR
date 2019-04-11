"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored

from appJar import gui

from modelUseage.useage import PostProcessor

model_path='/home/ansary/WORK/OCR/Models/testModel.hdf5'

obj=PostProcessor(model_path)

def press(button):
    if button=="PREDICT":
        img_path=app.getEntry("img_path")
        app.setLabel('RES','[WORKING]')
        app_data=obj.predict_symbol(img_path,app_data=True)
        app.setLabel('RES','PREDICTION::{}'.format(app_data))
        
    

with gui("বাংলা OCR",bg='RoyalBlue',fg='White') as app:
    # GUI Properties
    app.setFont(size=15,weight='bold')
    app.label("Model: DenseNet", colspan=2,sticky="news",expand="both",bg="SlateBlue")
    app.label("Absolute Image Path", colspan=2,sticky="news",expand="both",bg="DarkSlateBlue")
    app.entry('img_path',value='')
    app.button('PREDICT',press,colspan=2,sticky="news",expand="both",bg="MediumSlateBlue",fg="WhiteSmoke")
    app.label('RES',value='[Prediction]',sticky="news",colspan=2,expand="both",bg="DarkSlateBlue",fg='Lime')
    