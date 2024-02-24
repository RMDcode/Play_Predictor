import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def MarvellousPlayPredictor(data_path):

    #step1: Load data
    data = pd.read_csv(data_path, index_col=0)

    print("Size of actual dataset",len(data))

    #step2 : Clean, Prepare and Manipulate data

    feature_names=['Weather','Temperature']

    print("Names of Features",feature_names)

    weather = data.Weather
    Temperature = data.Temperature
    play = data.Play

    #creating labelEncoder
    le = preprocessing.LabelEncoder()

    #converting string labels into numbers
    weather_encoded = le.fit_transform(weather)
    label = le.fit_transform(play)

    print(weather_encoded)

    #converting string labels into numbers
    temp_encoded = le.fit_transform(Temperature)
    label = le.fit_transform(play)

    print(temp_encoded)

    #Combining weather and temp into single list of tuples
    features = list(zip(weather_encoded,temp_encoded))

    #step3 : Train Data
    model = KNeighborsClassifier(n_neighbors=3)

    #Train the model using the training sets 
    model.fit(features,label) 

    #step4 : Test Data 
    predicted = model.predict([[0,2]]) #0 : Overcast, 2 : Mild

    print(predicted)

def main():
    print("<--- Marvellous Infosystems --->")
    print("Machine Learning Application")
    print("Play Predictor application using K Nearest Knighbor algorithm")

    MarvellousPlayPredictor("PlayPredictor.csv")

if __name__ == "__main__":
    main()
