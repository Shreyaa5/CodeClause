import numpy as np
import streamlit as slt
import pandas as pd
import keras.models
import pickle


def predicting_mood(input_array):

    #Converting to numpy_array

    new_model=keras.models.load_model("nn.h5")
    scaler=pickle.load(open('scaler.pkl','rb'))
    input_array = np.array(input_array).reshape(1, -1)
    array_used = pd.DataFrame(scaler.transform(input_array),columns=['Mean','Standard Deviation','Danceability','Valence','Loudness(in dB)','KEY_NUM_VAL'],index=None)
    pred1 = (new_model.predict(array_used)>.5).astype(int)
    print(pred1)

    if pred1 == 0:
        return "Sad"
    else:
        return "Happy"

def main():

    #Giving Title to Web Page
    slt.title("Song Mood Detector Page")

    #Taking User Input

    Mean = slt.text_input("Enter the Mean of the Song")
    Standard_Deviation = slt.text_input("Enter the Standard Deviation of the song")
    Danceability = slt.text_input("Enter Danceability of the song")
    Valence = slt.text_input("Enter Valence")
    Loudness = slt.text_input("Enter Loudness (in db)")
    Key = slt.text_input("Enter Key_number_value")

    mood = ''

    #Creating a 'GO' Button
    if slt.button("Prediction"):
        mood = predicting_mood([Mean, Standard_Deviation, Danceability, Valence, Loudness, Key])
    
    slt.success(mood)


if __name__ == '__main__':
    main()