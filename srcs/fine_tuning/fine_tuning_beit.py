
from params import params
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss 
from transformers import BeitConfig, BeitImageProcessor, BeitForImageClassification
# from srcs.image_collector.image_collector import load_image_collection, RGB_convert
# from make_graph import make_graph
from matplotlib import pyplot as plt
BLUE = "\033[94m"
RESET = "\033[0m"  # Resets to default color

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

    # Get the predicted classes by finding the index of the max logit value per image
    predictions = torch.argmax(outputs.logits, dim=1)

    # Print the readable predictions and success rate
    good = 0
    for i in range(20):
        if i < 10 and predictions.tolist()[i] == 0:
            good += 1
        elif i >= 10 and predictions.tolist()[i] == 1:
            good += 1
    print(f"{BLUE}Predicted classes for evaluation images: {predictions.tolist()}, {good}/20{RESET}", sep="")
    return good * 5

def basic_training_loop(model: BeitForImageClassification, inputs: dict, labels: list, epochs: int, lr: float):
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

        total_loss += loss.item()  # Add loss to total
        
        # lst_epoch.append(epoch)
        # lst_loss.append(total_loss)

        # Print the loss for the epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

def epoch_tester(range_epoch, turns, collection, val_collection, model, processor, batch, labels, lr):
    #-------------- Loop Epoch Tester ----------------------------------------------#
    lst_res = []
    for epochs in range(range_epoch[0], range_epoch[1]):
        res = 0
        for i in range(1, turns + 1):
            print(f"Testing collection of {len(collection)} images, {epochs} epochs(s), turn {i}/{turns} : " )
            basic_training_loop(model, batch, labels, epochs, lr)
            # Evaluation
            res += eval_model(model, processor, "cpu", val_collection)
            print(res)
            # Reset weights
            reset_weights(model)
        print(f"average: {res / turns}")
        lst_res.append(res / turns)
        print(lst_res)
    return lst_res
