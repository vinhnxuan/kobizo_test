

from typing import List
from backend.lib.types import HistTransactions, Transaction, TransactionClassificationResult, TransactionParams
from datetime import datetime
import re 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

LARGE_TRANSACTION_THRESH = 1000

def convert_datetime(dt:str):
    m = re.search(r"([A-Za-z]{3,5}-[0-9]{2}-[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2})", dt)
    found_datetime = m.group()
    return datetime.strptime(found_datetime, "%b-%d-%Y %H:%M:%S").timestamp()

def convert_value(value:str):
    return float(value[1:])


class DataLoader:
    COLUMN_NAMES ={
        "id": "transaction id",
        "time": "time stamp",
        "value": "value",
        "method": "method called",
        "from": "from",
        "to": "to",
    }
    def __init__(self, df) -> None:
        self.df = df
        self.prepare_dataset(df)

    def convert2Dict(self)-> List[Transaction]:
        trans_data = self.df
        transactions = []
        for index, row in trans_data.iterrows():
            params = TransactionParams(
                id = row[self.COLUMN_NAMES["id"]],
                time_stamp = row["time"],
                from_addr = row["from"],
                to_addr = row["to"],
                value = row["value"],
                method = row["method"],
            )

            hist_params_01 = TransactionParams(
                time_stamp = row["prev_time_01"],
                from_addr = row["from"],
                to_addr = row["to"],
                value = row["prev_value_01"],
            )
            hist_params_02 = TransactionParams(
                time_stamp = row["prev_time_02"],
                from_addr = row["from"],
                to_addr = row["to"],
                value = row["prev_value_02"],
            )
            hist_params_03 = TransactionParams(
                time_stamp = row["prev_time_03"],
                from_addr = row["from"],
                to_addr = row["to"],
                value = row["prev_value_03"],
            )
            hist_params_04 = TransactionParams(
                time_stamp = row["prev_time_04"],
                from_addr = row["from"],
                to_addr = row["to"],
                value = row["prev_value_04"],
            )

            hist = HistTransactions(
                hist = [hist_params_01,hist_params_02,hist_params_03,hist_params_04]
            )

            transactions.append(Transaction(
                params = params,
                hist = hist
            ))
        return transactions

    def prepare_dataset(self, trans_data):

        trans_data["time"] = trans_data[self.COLUMN_NAMES["time"]].apply(convert_datetime)
        trans_data["value"] = trans_data[self.COLUMN_NAMES["value"]].apply(convert_value) 
        trans_data['method'] = trans_data[self.COLUMN_NAMES["method"]].map({'buy':0,'transfer':1,'swap':2,'printMoney':3})
        
        trans_data = self.feature_engineering(trans_data)
        self.df = trans_data

        X = trans_data[["time", "value", "method", 
                        "prev_time_01", "prev_value_01",
                        "prev_time_02", "prev_value_02",
                        "prev_time_03", "prev_value_03",
                        "prev_time_04", "prev_value_04",
                        ]]
        y1 = trans_data['large transaction'].map({False:0,True:1})  
        y2 = trans_data['rapid transaction'].map({False:0,True:1})    
        y3 = trans_data['fraud transaction'].map({False:0,True:1}) 
        self.X = X
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3

    @staticmethod
    def feature_engineering(trans_data):
        trans_data = trans_data.sort_values(by=['time'], ascending=False)
        prev_01 = False
        prev_02 = False
        prev_03 = False
        prev_04 = False

        trans_data.loc[:, "prev_time_01"] = -1
        trans_data.loc[:, "prev_time_02"] = -1
        trans_data.loc[:, "prev_time_03"] = -1
        trans_data.loc[:, "prev_time_04"] = -1
        trans_data.loc[:, "prev_value_01"] = -1.0
        trans_data.loc[:, "prev_value_02"] = -1.0
        trans_data.loc[:, "prev_value_03"] = -1.0
        trans_data.loc[:, "prev_value_04"] = -1.0
        for index, row in trans_data.iterrows():
            print(row["time"])
            for cmp_index, cmp_row in trans_data.iterrows():
                if index != cmp_index or cmp_row ["time"] > row["time"]:
                    continue

                if row['from'] == cmp_row['from'] and row['to'] == cmp_row['to']:
                    print(cmp_row['from'])
                    if not prev_01:
                        trans_data.loc[index,'prev_time_01'] = cmp_row ["time"]
                        trans_data.loc[index,'prev_value_01'] = cmp_row ["value"]
                        prev_01 = True
                    elif not prev_02:
                        trans_data.loc[index,'prev_time_02'] = cmp_row ["time"]
                        trans_data.loc[index,'prev_value_02'] = cmp_row ["value"]
                        prev_02 = True
                    elif not prev_03:
                        trans_data.loc[index,'prev_time_03'] = cmp_row ["time"]
                        trans_data.loc[index,'prev_value_03'] = cmp_row ["value"]
                        prev_03 = True
                    elif not prev_04:
                        trans_data.loc[index,'prev_time_04'] = cmp_row ["time"]
                        trans_data.loc[index,'prev_value_04'] = cmp_row ["value"]
                        prev_04 = True
                    

        return trans_data

class Trainer ():
    def __init__(self) -> None:
        self.clf = RandomForestClassifier(max_depth=2, random_state=0)
    
    def fit(self, X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
        # summarize
        print('Train', X_train.shape, y_train.shape)
        print('Test', X_test.shape, y_test.shape)
        self.clf.fit(X, y)
        return self.clf
        


class Classifier ():
    def __init__(self, type = "rule", models = None) -> None:
        self.type = type
        self.models = None

    def predict(self, transactions: List[Transaction])->List[Transaction]:
        results =[]
        for transaction in transactions:
            if self.type == "model":
                result = self.classify_transaction_by_model(transaction)
            else:
                result = self.classify_transaction_by_rule(transaction)
        print(results)
        results.append(result)

    def classify_transaction_by_rule(self, transaction: Transaction):
        
        if not transaction.params.class_result:
            transaction.params.class_result = TransactionClassificationResult(
                
            )
        ##### if value of transaction larger than threshold then define as large #####
        if transaction.params.value > LARGE_TRANSACTION_THRESH:
            transaction.params.class_result.large = True


        # #### Stuck with this error - have to be fixed ######
        # #### Define as rapid type #####
        # ##### For buy transaction: if two transactions happen from the same account and to the same account, 
        # # and duration less than 1 second #####
        # print(transaction.hist.hist)
        # if transaction.params.method == 0:
        #     if transaction.hist.hist[0]:

        #         if transaction.params.time_stamp - transaction.hist.hist[0].time_stamp == 1:
        #             transaction.params.class_result.rapid = True
        # ##### For others: if three transactions happen from the same account and to the same account, 
        # # and duration less than 1 second #####
        # else:
        #     if transaction.hist.hist[0] and transaction.hist.hist[1]:
        #         if transaction.params.time_stamp - transaction.hist.hist[0].time_stamp == 1:
        #             if transaction.params.time_stamp - transaction.hist.hist[1].time_stamp == 2:
        #                 transaction.params.class_result.rapid = True

        # #### Define as fraud transaction #####
        # ##### For buy transaction: if two transactions happen from the same account and to the same account, 
        # # and duration less than 1 second #####
        # if transaction.params.method == 2:
        #     if transaction.hist.hist[0] and transaction.hist.hist[1] and transaction.hist.hist[2]:
        #         if transaction.params.time_stamp - transaction.hist.hist[0].time_stamp == 1:
        #             if transaction.params.time_stamp - transaction.hist.hist[1].time_stamp == 2:
        #                 if transaction.params.time_stamp - transaction.hist.hist[2].time_stamp == 3:
        #                     transaction.params.class_result.fraud = True
        # ##### For others: if three transactions happen from the same account and to the same account, 
        # # and duration less than 1 second #####
        # else:
        #     if transaction.hist.hist[0] and transaction.hist.hist[1]:
        #         if transaction.params.time_stamp - transaction.hist.hist[0].time_stamp == 1:
        #             if transaction.params.time_stamp - transaction.hist.hist[1].time_stamp == 2:
        #                 transaction.params.class_result.fraud = True

        return transaction

    def classify_transaction_by_model(self, transaction: Transaction):
        if self.models:
            self.models

        else:
            return self.classify_transaction_by_rule()
        