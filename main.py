from datasets import load_dataset
import pandas as pd
from matplotlib import pyplot as plt

dataset = load_dataset("dair-ai/emotion",trust_remote_code=True)

#sadness 0
#joy 1
#love 2
#anger 3
#fear 4
#surprise 5

labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

#extract the train dataset (first one)
train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

train_dataset = pd.DataFrame(train_dataset)

# Create binary encoded features for each label
for label in labels:
    train_dataset[label] = (train_dataset['label'] == labels.index(label)).astype(int)

#remove the label column
train_dataset = train_dataset.drop(columns=['label'])

#print the features
print(train_dataset.head())
