import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from baseline.dataset_config import dataset_config 
from data.normal_dataset import NormalPoseDataset
from torchvision.transforms import ToTensor, Compose
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from model.efficientnet import EfficientNet
from utils.focal_loss import FocalLoss
from model.base_module import LinearBlock
from utils.general import adjust_learning_rate
from utils.general import load_config, accuracy, plot_confusion_matrix, colorstr, accuracy
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score

CLASSES = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

# CLASSES = ["1", "2", "3"]

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained("efficientnet-b0")
        self.dropout1 = nn.Dropout(0.4)
        self.linear = LinearBlock(1280, 3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs):
        # print(inputs.shape)
        inputs = inputs.float()
        feature = self.feature_extractor.extract_features(inputs)
        return self.linear(feature)
    
def get_model():
    return Model()

def build_dataloader(data_cfg):
    # Build train/test dataloader
    train_dataset = NormalPoseDataset(data_cfg['data_dir'],
                                      data_cfg['train_list'],
                                      data_cfg['mapping_file_train'],
                                      data_cfg['image_dir'],
                                      classes=CLASSES,
                                      augment_config_path=data_cfg['augmentation_config_path'],
                                      transform = Compose([eval(data_cfg["test_transform"])()]))
    test_dataset = NormalPoseDataset(data_cfg['data_dir'],
                                     data_cfg['test_list'],
                                     data_cfg['mapping_file_test'],
                                     data_cfg['image_dir'],
                                     classes=CLASSES,
                                     transform = Compose([eval(data_cfg["test_transform"])()]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=data_cfg['batch_size'],
                                                   shuffle=data_cfg['shuffle'],
                                                   num_workers=data_cfg['num_workers'],
                                                   pin_memory=data_cfg['pin_memory'])
    
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=data_cfg['batch_size'],
                                                   shuffle=data_cfg['shuffle'],
                                                   num_workers=data_cfg['num_workers'],
                                                   pin_memory=data_cfg['pin_memory'])
    return train_dataloader, test_dataloader

def train(model, train_dataloader, test_dataloader, num_epochs=50):
    # define loss and optimizer
    train_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], 0.1)   
    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # train
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        adjust_learning_rate(optimizer, epoch, 0.001)
        for i, batch in enumerate(tqdm(train_dataloader)):
            image, label = batch[0].to(device), batch[2].to(device)
            optimizer.zero_grad()
            loss = train_step(model, image, label, train_criterion)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Train Loss: {loss}')
        results = validate(model, test_dataloader, train_criterion)
        epoch_end(epoch, results)

    print(colorstr("Final test"))
    # Final test
    model.eval()
    correct = 0
    total = 0
    prediction = []
    labels = []
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            image, label = batch[0].to(device), batch[2].to(device)
            output = model(image)
            _, pred = torch.max(output.data, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()

            label = label.cpu().numpy()
            labels += label.tolist()
            output = torch.argmax(output, axis=1)
            prediction += output.cpu().numpy().tolist()
        
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    precision, recall, fscore, support = score(labels, prediction)
    report = classification_report(labels, prediction)
    print('\nprecision: {}'.format(precision))
    print('\nrecall: {}'.format(recall))
    print('\nfscore: {}'.format(fscore))
    print('\nsupport: {}\n'.format(support))
    print(report)

    plot_confusion_matrix(labels, prediction, ["1", "2", "3"], normalize=False, title="Non-normalized confusion matrix (all)",
                        savepath=f"non_normalize.png")
    plot_confusion_matrix(labels, prediction, ["1", "2", "3"], normalize=True, title="Normalized confusion matrix (all)",
                        savepath=f"normalize.png")
    print("------------------------------------------------------")


def train_step(model, image, label, criterion):
    model.train()
    out = model(image)
    loss = criterion(out, label)
    return loss 

def validate(model, test_loader, criterion):
    model.eval()
    with torch.no_grad():
        outputs = [validate_step(model, batch, criterion) for batch in test_loader]
    
    return validate_epoch_end(outputs)

def validate_step(model, batch, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image, label = batch[0].to(device), batch[2].to(device)
    out = model(image)
    loss = criterion(out, label)
    acc = accuracy(out, label)
    return {"val_loss": loss, "val_acc": acc}

def validate_epoch_end(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def epoch_end(epoch, result):
    print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['val_loss'], result['val_acc']))

if __name__ == "__main__":
    model = get_model()
    train_dataloader, test_dataloader = build_dataloader(dataset_config)
    train(model, train_dataloader, test_dataloader, num_epochs=30)


