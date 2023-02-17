import torch
from torchvision import datasets, transforms


#Input [data directory]

def load_data(data_directory):
    train_dir = data_directory + '/train'
    test_dir = data_directory + '/test'
    
    
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    
    test_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])


    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    
    data_set = []
    data_set.append(trainloader)
    data_set.append(testloader)
    class_to_idx = train_data.class_to_idx
    
    return data_set, class_to_idx
    