from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

def Create_Dataset():
    Featuers, Labels=make_classification(n_samples=10000, n_features=10, n_informative=10, n_redundant=0, random_state=1)
    Labels = np.where(Labels == 0,-1,1)
    return Featuers, Labels 

def Standardize_Data(Featuers):
    mean = np.mean(Featuers, axis=0)
    std_dev = np.std(Featuers, axis=0)
    Featuers = (Featuers - mean) / std_dev
    return Featuers

def splitt_preprocess(Featuers, Labels):
    Featuers = Standardize_Data(Featuers)
    Train_Featuers, Test_Featuers, Train_Labels, Test_Labels= train_test_split(Featuers, Labels, test_size=0.3, random_state=42)
    return Train_Featuers, Test_Featuers, Train_Labels, Test_Labels

def TrainAdaline(Train_Featuers, Train_Labels):
    LearningRate = 0.01
    Weights = np.array([[0.5, 0.8, -0.2, 1.0, 0.3, -0.6, 0.7, -0.4, 0.9, -0.1]])
    bias = 0
    deltaWeights= 0 
    TempMSE = 10**6 
    MSE = 10
    MSEPlot = []
    max_epochs = 1000
    current_epoch = 0
    while MSE >= 10**-6 and current_epoch < max_epochs:
        current_epoch += 1
        Z = np.dot(Weights, Train_Featuers.T)+ bias
        predictedtrainLabel = np.where(Z >= 0.0,1,-1)

        deltaWeights = LearningRate*(Train_Labels.T - predictedtrainLabel)
        deltaWeights = np.dot(deltaWeights, Train_Featuers)
        Weights += deltaWeights

        deltabias = LearningRate* np.sum((Train_Labels.T - predictedtrainLabel))
        bias += deltabias
        MSE = Cost_Function(Train_Labels, predictedtrainLabel)        
        MSEPlot.append(MSE)
        if MSE<TempMSE:
            TempMSE = MSE
            BestWeights = Weights
            BestBias = bias
    
    return BestWeights,BestBias, MSEPlot 
   
def Cost_Function(Train_Labels, predictedtrainLabel):
    MSE = 0.5*1/len(Train_Labels) * np.sum((Train_Labels - predictedtrainLabel) ** 2)
    return MSE

def predict_Adaline(BestWeights, BestBias, Test_Featuers):
    PredictZ= np.dot(BestWeights, Test_Featuers.T)+ BestBias
    predictedtestLabel = np.where(PredictZ >= 0.0,1,-1)
    accuracey = (np.sum(Test_Labels == predictedtestLabel)/ len(Test_Labels))*100
    print(f"accuracey: {accuracey:.2f}%")
    return predictedtestLabel

def Visualization (predictedtestLabel, Test_Labels, MSEPlot):
    plt.figure(figsize=(10, 6))
    plt.plot(Test_Labels, label='True Labels', color='blue', alpha=0.7)
    plt.plot(predictedtestLabel.T, label='Predicted Labels', color='green', alpha=0.7)
    plt.title('True vs Predicted Labels')
    plt.xlabel('Sample Index')
    plt.ylabel('Label')
    plt.legend()
    plt.tight_layout()
    
    plt.figure(figsize=(10, 6))
    plt.plot(MSEPlot, label='MSE Curve', color='red', linewidth=2)  # Continuous curve
    plt.title('Performance (MSE vs Epochs)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show(block=True)

Featuers,Labels = Create_Dataset()
Train_Featuers, Test_Featuers, Train_Labels, Test_Labels = splitt_preprocess(Featuers, Labels)
BestWeights, BestBias, MSEPlot = TrainAdaline(Train_Featuers, Train_Labels)
predictedtestLabel = predict_Adaline(BestWeights, BestBias, Test_Featuers)
Visualization (predictedtestLabel, Test_Labels, MSEPlot)



