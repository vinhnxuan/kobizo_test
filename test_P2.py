
import pandas as pd
from backend.lib.fraud_transaction_detection import Classifier, DataLoader, Trainer
import pickle

METHOD_TYPE = "rule"
MODEL1_PATH = './data/model1.pkl'
MODEL2_PATH = './data/model2.pkl'
MODEL3_PATH = './data/model3.pkl'

def train(filepath:str):
    trans_data = pd.read_csv(filepath)
    data_loader = DataLoader(trans_data)
    trainer_1 = Trainer()
    model1 = trainer_1.fit(data_loader.X, data_loader.y1)
    # save
    with open(MODEL1_PATH,'wb') as f:
        pickle.dump(model1 , f)
    
    trainer_2 = Trainer()
    model2 = trainer_2.fit(data_loader.X, data_loader.y2)
    # save
    with open(MODEL2_PATH,'wb') as f:
        pickle.dump(model2 , f)

    trainer_3 = Trainer()
    model3 = trainer_3.fit(data_loader.X, data_loader.y3)
    # save
    with open(MODEL3_PATH,'wb') as f:
        pickle.dump(model3 , f)
    

def main(filepath:str):
    trans_data = pd.read_csv(filepath)
    data_loader = DataLoader(trans_data)
    transactions = data_loader.convert2Dict()
    if METHOD_TYPE == "model":
        with open(MODEL1_PATH, 'rb') as f:
            model1 = pickle.load(f)
        with open(MODEL2_PATH, 'rb') as f:
            model2 = pickle.load(f)
        with open(MODEL3_PATH, 'rb') as f:
            model3 = pickle.load(f)
        classifier = Classifier(type=METHOD_TYPE, models = [model1, model2, model3])
    else:
        classifier = Classifier(type=METHOD_TYPE)
    result = classifier.predict(transactions)
    return result

if __name__ == "__main__":
    filepath = "./data/P2_dataset_sample.csv"
    result = main(filepath)
    print(result)