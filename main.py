import os

import pandas as pd
from data.train_image_enc import ImageEncoder
from data.train_text_enc import preprocess_data
from models.mlp import define_model
from scripts.trainer_mlp import train_model
from models.xgb import XGBModel

from joblib import Parallel, delayed 
import joblib 

def encode_train_data(dataset_path, image_dir):
    """Offline feature-building step. NOT CURRENTLY RUNNABLE.

    This was the path that turned raw images and report text into the
    `Frontal_features` / `Lateral_features` / `text_features` columns. Those
    columns are already precomputed in `dataset/iu_dataset.csv`, so training
    reads them directly and never calls this.

    It is kept for reference only. To revive it you would need to:
      - restore the image-encoding block below (uses ImageEncoder/MobileNetV2)
      - implement the `TextEncoder` class it refers to, which does not exist
        anywhere in this repo
    """
    original_dataset = pd.read_csv(dataset_path)

    #encoding image
    # feature_extractor = ImageEncoder(original_dataset,image_dir)
    # original_dataset = original_dataset.dropna(subset=['Frontal', 'Lateral'])
    # frontal_features, lateral_features = feature_extractor.extract_image_features()
    # frontal_features_dict = dict(zip(original_dataset['uid'], frontal_features))
    # lateral_features_dict = dict(zip(original_dataset['uid'], lateral_features))  
    # original_dataset['Frontal_features'] = original_dataset['uid'].map(frontal_features_dict)
    # original_dataset['Lateral_features'] = original_dataset['uid'].map(lateral_features_dict)
    # original_dataset.dropna(subset=['frontal_features','fateral_features'], inplace=True)

    #encoding text, impression text
    original_dataset['text'] = original_dataset['findings']+ ' ' + original_dataset['indication']
    text_encoder = TextEncoder(original_dataset,'impression')
    encoded_texts = text_encoder.encode_and_build_model()
    original_dataset['impression_features'] = list(encoded_texts) 

    text_encoder = TextEncoder(original_dataset,'text')
    encoded_texts = text_encoder.encode_and_build_model()
    original_dataset['text_features'] = list(encoded_texts)

    original_dataset=original_dataset[['Frontal_features','Lateral_features','text_features','impression_features']]
    return original_dataset


def train(dataset_path,model):

        # Read dataset and perform preprocessing
    dataset_path = dataset_path

    if model=='mlp':
        # Define model
        X_train, X_test, y_train, y_test, label_encoder = preprocess_data(dataset_path,model)
        input_shape = (X_train.shape[1],)
        output_shape = len(label_encoder.classes_)
        model = define_model(input_shape, output_shape)

        # Train model
        history = train_model(model, X_train, y_train, X_test, y_test)

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test)

        model_path = os.path.join('saved_model', 'mlp_model.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)

        print(f'Test Loss: {test_loss}')
        print(f'Test Accuracy: {test_accuracy}')
        
    elif model=='xgb':
        dataset_path = dataset_path
        X_train, X_test, y_train, y_test = preprocess_data(dataset_path,model)
        xgb_model = XGBModel()
        xgb_model.train_model(X_train, y_train, X_test, y_test, params=None, num_round=300, model_path='xgb_model.model')


if __name__ == "__main__":
    # dataset_path = 'dataset/iu_dataset.csv'  
    # image_dir = 'dataset/Images'
    # encoded_df=encode_train_data(dataset_path, image_dir)

    # encoded_df= pd.read_csv('dataset/iu_dataset.csv')
    # encoded_df=encoded_df[['Frontal_features','Lateral_features','text_features','impression','impression_features']]
    train('dataset/iu_dataset.csv','mlp')