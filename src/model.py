import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.utils import extract_value_and_unit, CFG
from src.image_processing import process_image
import pandas as pd

# Define Bunny-Llama model loading
def get_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        'BAAI/Bunny-Llama-3-8B-V',
        quantization_config=bnb_config,
        torch_dtype=torch.float16, # float32 for CPU (Here also)
        device_map=CFG.device, # 'cuda' for GPU only (should use auto here ?, make changes if u want to)
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        'BAAI/Bunny-Llama-3-8B-V',
        trust_remote_code=True
    )
    
    return model, tokenizer

# (Leave this as it is, it may be used when we want to further fine tune the model)

# Train a dummy model with Bunny-Llama
# def train_dummy_model(train_data):
#     print("Training dummy model with Bunny-Llama features...")
#     # Placeholder for further training if required
#     return None

# # Train the Bunny-Llama model
# def train_model(train_file='dataset/train.csv'):
#     data = pd.read_csv(train_file)
#     model, tokenizer = get_model()

#     # Placeholder to add actual training if needed
#     dummy_model = train_dummy_model(data)
    
#     return model, tokenizer


# Prediction on test data
def predict(model, tokenizer, test_file='dataset/test.csv'):
    test_data = pd.read_csv(test_file)
    
    predictions = []
    index_list = []

    for _, row in test_data.iterrows():
        index = row['index']
        image_url = row['image_link']
        entity_name = row['entity_name']

        text, features = process_image(
            image_url,
            model,
            tokenizer,
            prompt='Please transcribe the text contained in the following image. Provide the exact words as they appear in the image, maintaining the order and format of the text.?')
        prediction = extract_value_and_unit(text, entity_name)
        predictions.append(prediction)
        index_list.append(index)
    
    return index_list, predictions
