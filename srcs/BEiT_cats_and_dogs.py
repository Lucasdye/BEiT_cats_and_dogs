import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss 
from transformers import BeitConfig, BeitImageProcessor, BeitForImageClassification
from srcs.image_collector.image_collector import load_image_collection, RGB_convert
from make_graph import make_graph
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
    

def main():
    #------------------------ Paths  -----------------------------------------#
    cat_path = "/home/lu/Coding/models/BEiT_cats_and_dogs/dataset/train/cats/cat_"
    dog_path = "/home/lu/Coding/models/BEiT_cats_and_dogs/dataset/train/dogs/dog_"
    dog_val_path = "/home/lu/Coding/models/BEiT_cats_and_dogs/dataset/val/dog_val_"
    cat_val_path = "/home/lu/Coding/models/BEiT_cats_and_dogs/dataset/val/cat_val_"

    #------------------------ Loading collection -----------------------------#
    cat_collection = RGB_convert(load_image_collection(cat_path, 10, ".jpg"))
    dog_collection = RGB_convert(load_image_collection(dog_path, 10, ".jpg"))
    dog_val_collection = RGB_convert(load_image_collection(dog_val_path, 10, ".jpg"))
    cat_val_collection = RGB_convert(load_image_collection(cat_val_path, 10, ".jpg"))
    val_collection = cat_val_collection + dog_val_collection
    all_images = cat_collection + dog_collection

    #-------------- Importing, initilazing and setting model and processor ---#
    # Load processor
    processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
    # Load the pre-trained model configuration
    config = BeitConfig.from_pretrained('microsoft/beit-base-patch16-224')
    # Set number of classes to 2 (cats and dogs)
    config.num_labels = 2
    # Initialize the model with the updated config
    model = BeitForImageClassification(config)
    # Set model in training mode
    model.train()

    #-------------- Loading model on device (cpu) ----------------------------#
    device = torch.device("cpu")
    model.to(device)

    #-------------- Defining collection and labels ---------------------------#
    collection =  cat_collection + dog_collection
    labels = ([0] * len(cat_collection)) + ([1] * len(dog_collection)) 
    
    #-------------- Converting images and labels into tensors ----------------#
    batch = processor(images=collection, return_tensors="pt") 
    labels = torch.tensor(labels) 
    
    #-------------- loading tensors on device (cpu) --------------------------#
    inputs = {k: v.to(device) for k, v in batch.items()}
    labels = labels.to(device)
    
    #-------------- Loop Tester ----------------------------------------------#
    lst_res = []
    fig1, ax1 = plt.subplots()
    turns = 2
    for epochs in range(3, 16):
        res = 0
        for i in range(1, turns + 1):
            print(f"Testing collection of {len(collection)} images, {epochs} epochs(s), turn {i}/{turns} : " )
            basic_training_loop(model, batch, labels, epochs, 1e-3)
            # Evaluation
            res += eval_model(model, processor, "cpu", val_collection)
            print(res)
            # Reset weights
            reset_weights(model)
        print(f"average: {res / turns}")
        lst_res.append(res / turns)

    fig1.suptitle("Binary Classification average success rate based upon number of epochs with a dataset of 20 images", fontsize=8)
    x = [el for el in range(3, 16)]
    make_graph(x, lst_res, "epochs", "success %", ax=ax1)
    plt.savefig("epoch_plots_2.png", dpi=300)
    plt.show() 

if __name__ == "__main__":
    main()