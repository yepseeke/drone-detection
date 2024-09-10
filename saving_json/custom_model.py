import os
import torch
import uuid

import torchvision.models as models
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from dataset_processing import load_json, save_json


class CustomModel:

    def __init__(self, model_name: str, num_classes: int = 10, transform_type: str = 'wavelet-cmor1.2-3',
                 pretrained: bool = True):
        self.model_name = model_name.lower()
        self.model_id = str(uuid.uuid4())

        self.num_classes = num_classes
        self.transform_type = transform_type
        self.pretrained = pretrained

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self._get_model()
        self.model.to(self.device)

    def train_model(self, train_loader: DataLoader, valid_loader: DataLoader, epochs: int = 20, learning_rate=0.01,
                    is_accuracy=False, save=True, save_config=True):

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        avg_loss = 1

        for epoch in tqdm(range(1, epochs + 1)):
            losses = []

            self.model.train()
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device=self.device), targets.to(device=self.device)
                data = data.permute(0, 3, 1, 2).float()
                scores = self.model(data)
                loss = criterion(scores, targets)

                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = sum(losses) / len(losses)

            if is_accuracy:
                self.check_accuracy(valid_loader)

            print(f'Epoch {epoch}: Average epoch loss = {avg_loss:.4f}')

        if save:
            self.save(optimizer, epochs, avg_loss)

        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if save_config:
            new_entry = {
                "model_name": self.model_name,
                "model_id": self.model_id,
                "end_time": end_time,
                "epoch": epochs,
                "loss": avg_loss,
                "tranform": self.transform_type
            }

            config_filepath = "conf.json"
            if os.path.exists(config_filepath):
                data = load_json(config_filepath)
                print(data)
                data.append(new_entry)
            else:
                data = [new_entry]
            save_json(config_filepath, data)

    # TODO: update model_id, model_name, transform_type
    def load_weights(self, weights_path: str):
        checkpoint = torch.load(weights_path)
        self.model.load_state_dict(checkpoint)

    def check_accuracy(self, loader: DataLoader):
        num_correct = 0
        num_samples = 0
        self.model.eval()

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device=self.device), y.to(device=self.device)
                x = x.permute(0, 3, 1, 2).float()
                scores = self.model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum().item()
                num_samples += predictions.size(0)

        accuracy = num_correct / num_samples
        print(f'Accuracy: {accuracy * 100:.5f}%')

        return accuracy

    def update_device(self, device_name):
        self.device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def summary(self):
        print(self.model)

    def save(self, optimizer, epoch, loss):
        checkpoint_filename = f'{self.model_name}_{self.model_id}.pth'
        checkpoint_filepath = os.path.join('models', checkpoint_filename)

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_filepath)

    def _get_model(self):
        if self.model_name == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
        elif self.model_name == 'resnet34':
            model = models.resnet34(pretrained=self.pretrained)
        elif self.model_name == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
        elif self.model_name == 'resnet101':
            model = models.resnet101(pretrained=self.pretrained)
        elif self.model_name == 'resnet152':
            model = models.resnet152(pretrained=self.pretrained)
        elif self.model_name == 'alexnet':
            model = models.alexnet(pretrained=self.pretrained)
        elif self.model_name == 'vgg11':
            model = models.vgg11(pretrained=self.pretrained)
        elif self.model_name == 'vgg11_bn':
            model = models.vgg11_bn(pretrained=self.pretrained)
        elif self.model_name == 'vgg13':
            model = models.vgg11(pretrained=self.pretrained)
        elif self.model_name == 'vgg13_bn':
            model = models.vgg13_bn(pretrained=self.pretrained)
        elif self.model_name == 'vgg16':
            model = models.vgg11(pretrained=self.pretrained)
        elif self.model_name == 'vgg16_bn':
            model = models.vgg13_bn(pretrained=self.pretrained)
        elif self.model_name == 'vgg19':
            model = models.vgg11(pretrained=self.pretrained)
        elif self.model_name == 'vgg19_bn':
            model = models.vgg13_bn(pretrained=self.pretrained)
        elif self.model_name == 'densenet121':
            model = models.densenet121(pretrained=self.pretrained)
        elif self.model_name == 'densenet161':
            model = models.densenet161(pretrained=self.pretrained)
        elif self.model_name == 'densenet169':
            model = models.densenet169(pretrained=self.pretrained)
        elif self.model_name == 'densenet201':
            model = models.densenet201(pretrained=self.pretrained)
        else:
            raise ValueError(f"Model {self.model_name} isn't supported")

        if 'resnet' in self.model_name:
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif 'densetnet' in self.model_name:
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)
        else:
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        return model
