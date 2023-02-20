# pylint: disable=redefined-outer-name
import argparse
from torch.utils.data import DataLoader, random_split
import model as models
import torch
import torch.nn as nn
import torch.optim as optim
import bentoml
from tqdm import tqdm
from utils import DatasetGenerator

ce_loss = nn.CrossEntropyLoss()


def test_model(model, test_loader, device="cpu"):
    correct, num_predictions = 0, 0
    with tqdm(total=len(test_loader), desc="Test") as pb:
        model.eval()
        for batch_num, (img, img_label) in enumerate(test_loader):
            img = img.to(device)
            img_label = img_label.to(device)
            predictions = model(img)
            correct += (torch.argmax(predictions, dim=1) == img_label).sum().item()
            num_predictions += predictions.shape[0]
            pb.update(1)
    return correct, num_predictions


def train(train_loader, device="cpu"):

    class_num = 2
    # set hyperparameters
    lr = 0.001
    weight_decay = 0.0005
    epoch = 1
    # initialize models
    model = models.ResNet(class_num=class_num)
    model = model.to(device)

    # define optimizers
    opt_model = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # train
    model.train()
    torch.set_grad_enabled(True)

    for epo in range(1, epoch+1):
        correct_resnet = 0
        print("Epoch {}/{} \n".format(epo, epoch))
        with tqdm(total=len(train_loader), desc="Train") as pb:
            for batch_num, (img, img_label) in enumerate(train_loader):
                opt_model.zero_grad()
                img = img.to(device)
                img_label = img_label.to(device)
                outputs = model(img)
                correct_resnet += (torch.argmax(outputs, dim=1) == img_label).sum().item()
                loss = ce_loss(outputs, img_label)
                loss.backward()
                opt_model.step()
                pb.update(1)
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="BentoML PyTorch")
    parser.add_argument(
        "--cuda", action="store_true", default=True, help="enable CUDA training"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="pytorch",
        help="name for saved the model",
    )
    batch_size = 64
    data_root = "diffusion/pizza-not-pizza"
    args = parser.parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = DatasetGenerator(data_root)
    train_set_length = int(0.7 * len(dataset))
    test_set_length = len(dataset) - train_set_length
    # split the dataset
    train_set, test_set = random_split(dataset, [train_set_length, test_set_length])
    # define loaders
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size)

    trained_model = train(train_loader, device)
    correct, total = test_model(trained_model, test_loader, device)

    # training related
    metadata = {
        "acc": float(correct) / total
    }

    signatures = {"predict": {"batchable": True}}

    saved_model = bentoml.pytorch.save_model(
        args.model_name,
        trained_model,
        signatures=signatures,
        metadata=metadata,
        external_modules=[models],
    )
    print(f"Saved model: {saved_model}")
