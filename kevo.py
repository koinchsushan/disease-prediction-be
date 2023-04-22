from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import random
import statistics
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import math
import csv
from starlette.middleware.cors import CORSMiddleware


class NaiveBayes:
    # split the dataset into test set and train set
    def split_dataset(self, dataset, ratio):
        train_size = int(len(dataset) * ratio)
        train_set = []
        test_set = list(dataset)
        while len(train_set) < train_size:
            index = random.randrange(len(test_set))
            train_set.append(test_set.pop(index))
        return [train_set, test_set]

    # split the dataset by class values & return a dictionary
    def separate_by_class(self, dataset):
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if class_value not in separated:
                separated[class_value] = list()
            separated[class_value].append(vector)

        for keys, values in separated.copy().items():
            if len(values) <= 1:
                del (separated[keys])
        return separated

    # calculate mean, standard deviation and count for each column in dataset
    def summarize_dataset(self, dataset):
        summaries = [(statistics.mean(column), statistics.stdev(column, ), len(column)) for column in zip(*dataset)]
        del (summaries[-1])
        return summaries

    # split dataset by class then calculate statistics for each row
    def summarize_by_class(self, dataset):
        separated = self.separate_by_class(dataset)
        summaries = {}
        for class_value, rows in separated.items():
            summaries[class_value] = self.summarize_dataset(rows)
        return summaries

    # calculate the gaussian probability distribution function for x
    def calculate_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    # calculate the probabilities of predicting each class for a given data
    def calculate_class_probabilities(self, summaries, input):
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = 1
            for i in range(len(class_summaries)):
                mean, standard_deviation, _ = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(input[i], mean, standard_deviation)
        return probabilities

    # predict the class for a given data
    def predict(self, summaries, row):
        probabilities = self.calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    def get_predictions(self, summary, test_data):
        predictions = []
        for i in range(len(test_data)):
            predictions.append(self.predict(summary, test_data[i]))
        return predictions


#  read the dataset
original_data = pd.read_csv("revisedDataset.csv")
dataset_copy = original_data.copy()
df = original_data[original_data.Disease != 'Typhoid']

cols = df.columns
data = df[cols].values.flatten()

s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)

df = pd.DataFrame(s, columns=df.columns)
# df.to_csv('revisedDataset.csv', index=False)

disease_encoder = LabelEncoder()
symptoms_encoder = LabelEncoder()

# file = open("disease.txt", "w")
# symp = open("symptoms.txt", "w")

X = df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5']]
y = df['Disease'].values.tolist()
# for i in range(len(np.unique(y))):
#     file.write(np.unique(y)[i])
#     file.write("\n")

# file.close()

y = disease_encoder.fit_transform(y)
diseases_encoded = np.unique(y)

symptoms = np.unique(X)
# for i in range(len(symptoms)):
#     symp.write(symptoms[i])
#     symp.write("\n")
# symp.close()

symptom_ids = symptoms_encoder.fit_transform(symptoms)
symptom_map = dict(zip(range(len(symptoms_encoder.classes_)), symptoms_encoder.classes_))

for a in X.columns:
    X[a] = symptoms_encoder.transform(X[a])

y = pd.DataFrame(y, columns=["Disease"])
dataset = pd.merge(X, y, left_index=True, right_index=True)
dataset = dataset.values.tolist()

naiveBayes = NaiveBayes()
plit_ratio = 0.85
train_set, test_set = naiveBayes.split_dataset(dataset, plit_ratio)
print("Split {0} rows into train :- {1} and test :- {2} rows.".format(len(dataset), len(train_set), len(test_set)))

# prepare model
summaries = naiveBayes.summarize_by_class(dataset)

# test the prepared model
prediction = naiveBayes.get_predictions(summaries, test_set)

# get the accuracy
y_test = pd.DataFrame(test_set)
y_test = y_test.iloc[:, -1]

# performance matrix
accuracy = accuracy_score(y_test, prediction)
f1_score = f1_score(y_test, prediction, average='weighted')
recall_score = recall_score(y_test, prediction, average='weighted')
precision_score = precision_score(y_test, prediction, average='weighted')
confusion_matrix = confusion_matrix(y_test, prediction)

print("Accuracy :- {0}%".format(accuracy))
print("F1 Score :- {0}".format(f1_score))
print("Recall Score :- {0}".format(recall_score))
print("Precision Score :- {0}".format(precision_score))
print("Confusion Matrix :- \n", confusion_matrix)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class Request_body(BaseModel):
    """
    Request body for the API.
    """
    symptom1: str
    symptom2: str
    symptom3: str
    symptom4: str
    symptom5: str


class Disease_request(BaseModel):
    disease: str
    symptom1: str
    symptom2: str
    symptom3: str
    symptom4: str
    symptom5: str


@app.get("/api/v1/symptoms")
def get_symptoms():
    """
    Returns all the unique symptoms to JSON format
    """
    symptoms_list = list(symptom_map.values())

    symptomsJson = []

    for symptom in symptoms_list:
        symptomDict = dict()
        symptomDict['name'] = symptom
        symptomDict['id'] = symptoms_list.index(symptom)
        symptomsJson.append(symptomDict)

    return symptomsJson


@app.get("/api/v1/data")
def get_data():
    dataJson = []
    dataArray = dataset_copy.values.tolist()
    for row in dataArray:
        dataDict = dict()
        dataDict["Disease"] = row[0]
        dataDict["Symptoms1"] = row[1]
        dataDict["Symptoms2"] = row[2]
        dataDict["Symptoms3"] = row[3]
        dataDict["Symptoms4"] = row[4]
        dataDict["Symptoms5"] = row[5]

        dataJson.append(dataDict)
    return dataJson


@app.post("/api/v1/disease")
def add_disease(disease: Disease_request):
    user_input = [disease.disease, disease.symptom1, disease.symptom2,
                  disease.symptom3, disease.symptom4, disease.symptom5]
    with open('revisedDataset.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow(user_input)


@app.post("/api/v1/predict")
def predict(request_body: Request_body):
    """
    Predict the probability of having a disease
    """
    user_input = [request_body.symptom1, request_body.symptom2, request_body.symptom3,
                  request_body.symptom4, request_body.symptom5]

    user_input = symptoms_encoder.transform(user_input)
    user_input = np.reshape(user_input, (1, 5))

    predict_user_input = naiveBayes.get_predictions(summaries, user_input)
    output = disease_encoder.inverse_transform(predict_user_input)
    return {"Disease": output[0]}
