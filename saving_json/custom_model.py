import os
import torch
import uuid
import numpy as np

import torchvision.models as models
import torchmetrics
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from datetime import datetime

from dataset_processing import load_json, save_json


# TODO get rid of permutations
class CustomModel:
    """
    The CustomModel class represents a customizable model designed for working with various architectures and data
    transformations.

    Attributes:
    ----------

    model_name : str
        The name of the model architecture to be used. For example 'resnet18', 'vgg13_bn'
    num_classes : int, default is 10
        The number of output classes the model will predict. Defaults to 10 classes.
    transform_type : str, default is 'wavelet-cmor1.2-3'
        The type of transformation to apply to input data before feeding it into the model. By default, it uses
        a Continuous Morlet Wavelet Transform with parameters 1.2 and 3 (cmor1.2-3).
    pretrained : bool, default is True
        If True, the model will load pretrained weights.

    Example usage:
    --------------
    model = CustomModel('resnet50', num_classes=100, transform_type='fft', pretrained=False)

    In this example, a ResNet50 model is created with 100 output classes, using an FFT transformation on the input data
    without loading pretrained weights.
    """

    def __init__(self, model_name: str = 'resnet18', num_classes: int = 5, transform_type: str = 'wavelet=cmor1.2-3',
                 pretrained: bool = True, device='cpu'):
        self.model_name = model_name.lower()
        self.model_id = str(uuid.uuid4())

        self.num_classes = num_classes
        self.transform_type = transform_type
        self.pretrained = pretrained
        self.epochs_trained = 0

        self.device = torch.device(device)

        self.model = self._get_model()
        self.model.to(self.device)

    def train(self, train_loader: DataLoader, valid_loader: DataLoader, epochs: int = 20, learning_rate=0.01,
              save=True, save_config=True, stop_loss=0.001):
        """
            Trains the model using the provided training and validation datasets,
            with options to save the model and its configuration.

            Parameters:
            -----------

            train_loader : DataLoader
                DataLoader for the training dataset, supplying batches of data for model training.
            valid_loader : DataLoader
                DataLoader for the validation dataset, used to assess model performance after each epoch.
            epochs : int, default is 20
                The number of complete passes through the training dataset.
            learning_rate : float, default is 0.01
                The learning rate for the optimization algorithm, determining the step size in gradient descent.
            save : bool, default is True
                If True, the trained model is saved to disk at the end of training.
                ../models/new_model.pth
            save_config : bool, default is True
                If True, the model's configuration (such as model_name, model_id, end time of training,
                epochs, loss and transform type)
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        avg_loss = 1
        val_accuracy = 0
        val_precision = 0
        val_recall = 0
        epochs_trained = epochs

        for epoch in tqdm(range(1, epochs + 1)):
            losses = []

            self.model.train()
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)

                scores = self.model(data)
                loss = criterion(scores, targets)

                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = sum(losses) / len(losses)

            val_accuracy = self.calculate_accuracy(valid_loader)
            val_precision = self.calculate_precision(valid_loader)
            val_recall = self.calculate_recall(valid_loader)

            print(f'Accuracy: {val_accuracy}, Precision: {val_precision}, Recall: {val_recall}.')

            print(f'Epoch {epoch}: Average epoch loss = {avg_loss:.4f}')

            if avg_loss < stop_loss:
                epochs_trained = epoch
                break

        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.model_id = str(uuid.uuid4())

        self.epochs_trained += epochs_trained

        if save:
            self.save(optimizer)

        if save_config:
            new_entry = {
                'model_id': self.model_id,
                'model_name': self.model_name,
                'end_time': end_time,
                'epoch': self.epochs_trained,
                'loss': avg_loss,
                'accuracy': val_accuracy,
                'precision': val_precision,
                'recall': val_recall,
                'transform': self.transform_type
            }

            config_filepath = "conf.json"
            if os.path.exists(config_filepath):
                data = load_json(config_filepath)
                print(data)
                data.append(new_entry)
            else:
                data = [new_entry]
            save_json(config_filepath, data)

    def load(self, model_path: str, device='cpu'):
        checkpoint = torch.load(model_path, map_location=device)
        if not checkpoint.get('model_name'):
            raise Exception('Unable to get model name.')
        self.model_name = checkpoint['model_name']

        if not checkpoint.get('transform'):
            raise Exception('Unable to load data transformation.')
        self.transform_type = checkpoint['transform']

        self.num_classes = checkpoint.get('num_classes', 5)
        if not checkpoint.get('num_classes'):
            print('Unable to load number of classes. The value is set to 5.')

        self.model = self._get_model()

        if not checkpoint.get('model_state_dict'):
            raise Exception('Unable to load model.')
        model_state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(model_state_dict)

        self.model_id = checkpoint.get('model_id', str(uuid.uuid4()))

        if checkpoint.get('epochs'):
            self.epochs_trained = checkpoint['epochs']

        if checkpoint.get('optimizer_state_dict'):
            pass

    def update_device(self, device_name):
        self.device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def summary(self):
        print('Info:')
        print(f'Model id: {self.model_id}')
        print(f'Model name: {self.model_name}')
        print(f'Epochs trained: {self.epochs_trained}')
        print(f'Num of classes: {self.num_classes}')
        print(f'Transform type: {self.transform_type}')
        print(self.model)

    def save(self, optimizer):
        checkpoint_filename = f'{self.model_id}.pth'

        if not os.path.isdir('models'):
            os.mkdir('models')

        checkpoint_filepath = os.path.join('models', checkpoint_filename)

        torch.save({
            'model_id': self.model_id,
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'transform': self.transform_type,
            'epochs': self.epochs_trained,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_filepath)

    # Calculation of metrics:
    def calculate_accuracy(self, loader: DataLoader):
        """
            Evaluates the accuracy of the model on the provided dataset.

            Parameters:
            -----------
                loader : DataLoader
                A DataLoader object containing the dataset over which the accuracy will be calculated.
            Returns:
            --------
                float:
        """
        num_correct = 0
        num_samples = 0
        self.model.eval()

        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(device=self.device), targets.to(device=self.device)

                scores = self.model(data)
                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum().item()
                num_samples += predictions.size(0)

        accuracy = float(num_correct / num_samples)

        return accuracy

    def calculate_precision(self, loader):
        self.model.eval()

        precision_metric = torchmetrics.Precision(num_classes=self.num_classes,
                                                  average='macro', task='multiclass').to(self.device)
        for data, targets in loader:
            data, targets = data.to(self.device), targets.to(self.device)

            scores = self.model(data)
            _, preds = torch.max(scores, 1)

            precision_metric.update(preds, targets)

        precision = precision_metric.compute().item()
        return precision

    def calculate_recall(self, loader):
        self.model.eval()

        recall_metric = torchmetrics.Recall(num_classes=self.num_classes, average='macro',
                                            task='multiclass').to(self.device)
        for data, targets in loader:
            data, targets = data.to(self.device), targets.to(self.device)

            scores = self.model(data)
            _, preds = torch.max(scores, 1)

            recall_metric.update(preds, targets)

        recall = recall_metric.compute().item()
        return recall

    def confusion_matrix(self, loader: DataLoader):
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(self.device), targets.to(self.device)

                scores = self.model(data)
                _, preds = scores.max(1)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        cm = confusion_matrix(all_targets, all_preds)

        return cm

    # def plot_confusion_matrix(self, ax, loader: DataLoader, cmap='viridis'):
    #     cm = self.confusion_matrix(loader)
    #
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #     disp.plot(ax=ax, cmap=cmap, colorbar=False)

    def _get_model(self):
        model = None
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
        elif 'densenet' in self.model_name:
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)
        else:
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, self.num_classes)

        return model
