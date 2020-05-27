import torch
from torchvision import datasets, models, transforms

num_classes = 95
input_size = 224
batch_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, num_classes)

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create training and validation datasets
image_datasets = datasets.ImageFolder("./data/multiple_fruits", data_transforms['test'])
# Create training and validation dataloaders
dataloaders_dict = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=4)

checkpoint = torch.load('./checkpoint/ckpt_densenet.t7')
model.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

model.eval()
