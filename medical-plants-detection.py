

#necessary library to be installed  by "!pip install -U -q evaluate transformers datasets>=2.14.5 mlflow 2>/dev/null"

import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score
)
import evaluate
from datasets import Dataset, Image, ClassLabel
from transformers import (
    TrainingArguments,
    Trainer,
    ViTImageProcessor,
    ViTForImageClassification,
    DefaultDataCollator
)
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
    Resize,
    ToTensor
)

from PIL import ImageFile



ImageFile.LOAD_TRUNCATED_IMAGES = True


image_dict = {}


from pathlib import Path
from tqdm import tqdm
import os
MIN_SAMPLES = 100

file_names = []
labels = []

for file in sorted((Path('/kaggle/input/indian-medicinal-leaves-dataset/Indian Medicinal Leaves Image Datasets/').glob('*/*/*.jpg'))):

    sample_dir = '/'.join(str(file).split('/')[:-1])+'/'
    num_files_in_dir = [len(x) for _, _, x in os.walk(sample_dir)][0]
    if num_files_in_dir >= MIN_SAMPLES:
        file_names.append(str(file))
        label = str(file).split('/')[-2]
        labels.append(label)

print(len(file_names), len(labels), len(set(labels)))


dataset = Dataset.from_dict({"image": file_names, "label": labels}).cast_column("image", Image())


dataset[0]["image"]


labels_subset = labels[:5]


print(labels_subset)


labels_list = ['Amla', 'Curry', 'Betel', 'Bamboo', 'Palak(Spinach)', 'Coriender', 'Ashoka', 'Seethapala', 'Lemon_grass', 'Pappaya', 'Curry_Leaf', 'Lemon', 'Nooni',
               'Henna', 'Mango', 'Doddpathre', 'Amruta_Balli', 'Betel_Nut', 'Tulsi', 'Pomegranate',
                'Castor', 'Jackfruit', 'Insulin', 'Pepper', 'Raktachandini', 'Aloevera', 'Jasmine', 'Doddapatre', 'Neem',
                'Geranium', 'Rose', 'Gauva', 'Hibiscus', 'Nithyapushpa', 'Wood_sorel', 'Tamarind', 'Guava', 'Bhrami', 'Sapota', 'Basale', 'Avacado', 'Ashwagandha', 'Nagadali',
                'Arali', 'Ekka', 'Ganike', 'Tulasi', 'Honge', 'Mint', 'Catharanthus', 'Papaya', 'Brahmi']

label2id, id2label = dict(), dict()


for i, label in enumerate(labels_list):
    label2id[label] = i
    id2label[i] = label

print("Mapping of IDs to Labels:", id2label, '
')
print("Mapping of Labels to IDs:", label2id)


ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)


def map_label2id(example):
    example['label'] = ClassLabels.str2int(example['label'])
    return example

dataset = dataset.map(map_label2id, batched=True)


dataset = dataset.cast_column('label', ClassLabels)

dataset = dataset.train_test_split(test_size=0.2, shuffle=True, stratify_by_column="label")


train_data = dataset['train']


test_data = dataset['test']


model_str = 'dima806/medicinal_plants_image_detection'


processor = ViTImageProcessor.from_pretrained(model_str)


image_mean, image_std = processor.image_mean, processor.image_std


size = processor.size["height"]
print("Size: ", size)

normalize = Normalize(mean=image_mean, std=image_std)

_train_transforms = Compose(
    [
        Resize((size, size)),
        RandomRotation(90),
        RandomAdjustSharpness(2),
        RandomHorizontalFlip(0.5),
        ToTensor(),
        normalize
    ]
)

_val_transforms = Compose(
    [
        Resize((size, size)),
        ToTensor(),
        normalize
    ]
)

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples


def val_transforms(examples):
    examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples


train_data.set_transform(train_transforms)


test_data.set_transform(val_transforms)


def collate_fn(examples):

    pixel_values = torch.stack([example["pixel_values"] for example in examples])


    labels = torch.tensor([example['label'] for example in examples])


    return {"pixel_values": pixel_values, "labels": labels}


model = ViTForImageClassification.from_pretrained(model_str, num_labels=len(labels_list))


model.config.id2label = id2label
model.config.label2id = label2id


print(model.num_parameters(only_trainable=True) / 1e6)


accuracy = evaluate.load("accuracy")




#comparing performances using different metric
def compute_metrics(eval_pred):

    predictions = eval_pred.predictions


    label_ids = eval_pred.label_ids



    predicted_labels = predictions.argmax(axis=1)

    acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']


    return {
        "accuracy": acc_score
    }


metric_name = "accuracy"


model_name = "medicinal_plants_image_detection"


num_train_epochs = 10

args = TrainingArguments(

    output_dir=model_name,


    logging_dir='./logs',


    evaluation_strategy="epoch",


    learning_rate=2e-6,

    per_device_train_batch_size=32,


    per_device_eval_batch_size=8,


    num_train_epochs=num_train_epochs,


    weight_decay=0.02,


    warmup_steps=50,

    remove_unused_columns=False,


    save_strategy='epoch',


    load_best_model_at_end=True,


    save_total_limit=1,


    report_to="mlflow"
)



trainer = Trainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)



trainer.evaluate()

trainer.train()


trainer.evaluate()


outputs = trainer.predict(test_data)


print(outputs.metrics)


y_true = outputs.label_ids


y_pred = outputs.predictions.argmax(1)




def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8)):

    plt.figure(figsize=figsize)


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()


    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.0f'

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('True label')
    plt.xlabel('Predicted label')


    plt.tight_layout()

    plt.show()


accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')


print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")


if len(labels_list) <= 150:

    cm = confusion_matrix(y_true, y_pred)


    plot_confusion_matrix(cm, labels_list, figsize=(18, 16))




#saving the trained model
trainer.save_model()
