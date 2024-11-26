
from fine_tuning import fine_tuning_beit as fn
from params import params
from PIL import Image
from io import BytesIO
import torch
import requests
from torch.optim import Adam
from torch.nn import CrossEntropyLoss 
from transformers import BeitConfig, BeitImageProcessor, BeitForImageClassification
# from srcs.image_collector.image_collector import load_image_collection, RGB_convert
# from make_graph import make_graph
from matplotlib import pyplot as plt
BLUE = "\033[94m"
RESET = "\033[0m"  # Resets to default color


processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")

# Function to fetch and process cat images
def fetch_and_process_cat_images(num_images=50):
    cat_images = []
    for _ in range(num_images):
        response = requests.get("https://api.thecatapi.com/v1/images/search")
        if response.status_code == 200:
            img_url = response.json()[0]['url']
            img_data = requests.get(img_url).content  # Fetch image data
            img = Image.open(BytesIO(img_data))  # Convert to PIL image
            img_tensor = processor(images=img, return_tensors="pt").pixel_values  # Preprocess for model
            cat_images.append(img_tensor)
    return cat_images

# Function to fetch and process dog images
def fetch_and_process_dog_images(num_images=50):
    dog_images = []
    for _ in range(num_images):
        response = requests.get("https://dog.ceo/api/breeds/image/random")
        if response.status_code == 200:
            img_url = response.json()['message']
            img_data = requests.get(img_url).content  # Fetch image data
            img = Image.open(BytesIO(img_data))  # Convert to PIL image
            img_tensor = processor(images=img, return_tensors="pt").pixel_values  # Preprocess for model
            dog_images.append(img_tensor)
    return dog_images

# Define function to fetch cat images
def fetch_cat_images(num_images=50):
    cat_images = []
    for _ in range(num_images):
        response = requests.get("https://api.thecatapi.com/v1/images/search")
        if response.status_code == 200:
            cat_images.append(response.json()[0]['url'])
    print(cat_images)
    return cat_images
 
# Define function to fetch dog images
def fetch_dog_images(num_images=50):
    dog_images = []
    for _ in range(num_images):
        response = requests.get("https://dog.ceo/api/breeds/image/random")
        if response.status_code == 200:
            dog_images.append(response.json()['message'])
    return dog_images

def success_rate(collection_size, outputs):
    predictions = torch.argmax(outputs.logits, dim=1)
    good = 0
    for i in range(collection_size):
        if i < int(collection_size / 2) and predictions.tolist()[i] == 0:
            good += 1
        elif i >= int(collection_size / 2) and predictions.tolist()[i] == 1:
            good += 1
    print(f"{BLUE}Predicted classes for evaluation images: {predictions.tolist()}, {good}/20{RESET}", sep="")
    return good * int(100 / collection_size)


# def balanced_training(loss, evaluation_value):





def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def eval_model(model: BeitForImageClassification, processor: BeitImageProcessor, device: str, collection: list):
    # Puts the model into evaluation mode
    model.eval()

    # Move batch to processor
    val_batch = processor(images=collection, return_tensors="pt").to(device)

    # Run inference without calculating gradients
    with torch.no_grad():
        outputs = model(**val_batch)
    
    return(outputs)

def train_model(model: BeitForImageClassification, inputs: dict, labels: list, epochs: int, lr: float):
    # defines the loss function
    loss_fn = CrossEntropyLoss()

    # defines the optimizer, a necessary feature to improve the model    
    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        # Forward pass (predictions)
        outputs = model(**inputs)

        # Compares the model's output with the passed batch
        loss = loss_fn(outputs.logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()   # Clear previous gradients
        loss.backward()         # Calculate new gradients
        optimizer.step()        # Update weights

        total_loss += loss.item() # Add loss to total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    return (model)

def train_and_eval_model(model: BeitForImageClassification, processor, inputs: dict, labels: list, epochs: int, lr: float, val_collection):
    # defines the loss function
    loss_fn = CrossEntropyLoss()

    # defines the optimizer, a necessary feature to improve the model    
    optimizer = Adam(model.parameters(), lr=lr)
    success_rate = []
    losses = []
    for epoch in range(epochs):
        batch_loss = 0
        # Forward pass (predictions)
        outputs = model(pixel_values=inputs, labels=labels)

        # Compares the model's output with the passed batch
        loss = loss_fn(outputs.logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()   # Clear previous gradients
        loss.backward()         # Calculate new gradients
        optimizer.step()        # Update weights

        batch_loss = loss.item() # Add loss to total
        losses.append(batch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {batch_loss:.4f}")
        outputs = fn.eval_model(model, processor, "cpu", val_collection)
        success_rate.append(fn.success_rate(len(val_collection), outputs))
        print(f"success rate is {success_rate[-1]} %")
    return {"success_rate": success_rate, "losses": losses, "model": model}

def epoch_tester(range_epoch, turns, collection, val_collection, model, processor, batch, labels, lr):
    #-------------- Loop Epoch Tester ----------------------------------------------#
    lst_res = []
    for epochs in range(range_epoch[0], range_epoch[1]):
        res = 0
        for i in range(1, turns + 1):
            print(f"Testing collection of {len(collection)} images, {epochs} epochs(s), turn {i}/{turns} : " )
            train_model(model, batch, labels, epochs, lr)
            # Evaluation
            res += eval_model(model, processor, "cpu", val_collection)
            print(res)
            # Reset weights
            reset_weights(model)
        print(f"average: {res / turns}")
        lst_res.append(res / turns)
        print(lst_res)
    return lst_res
