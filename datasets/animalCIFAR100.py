"""
CIFAR-100 Dataset
This dataset was built in a try to extract all the image classes from the CIFAR100 dataset
Has not been used in our project in the end.
"""
import os
import ai8x
import torch
import torchvision
import torchvision.transforms as transforms

# Define a list of the animal class labels
animal_classes = ['bear','beaver','bee', 'butterfly', 'camel', 'caterpillar', 'cattle', 'chimpanzee', 'cockroach', 'crab', 'crocodile', 'dolphin', 'elephant', 'flatfish', 'hamster',
                'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
                'kangaroo', 'leopard','lizard', 'lion', 'seal', 'shark']

# Create a custom dataset that loads only the animal classes
class AnimalCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        self.targets = [self.classes.index(c) for c in animal_classes]
        self.data = self.data[torch.tensor(self.targets)]
        self.targets = torch.tensor(self.targets)

def animal_cifar100_get_datasets(data, load_train=True, load_test=True):
    (data_dir, args) = data

    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            #transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            # Normalize the image with mean and standard deviation
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            lambda x: x + torch.randn_like(x) * 0.1,  # Add Gaussian noise with std=0.1
            ai8x.normalize(args=args)
        ])

        train_dataset = AnimalCIFAR100(root=os.path.join(data_dir, 'CIFAR100'), 
                                                        train=True, download=True,
                                                        transform=train_transform)

    else:
        train_dataset = None

    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = AnimalCIFAR100(root=os.path.join(data_dir, 'CIFAR100'),
                                                    train=False, download=True,
                                                    transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1] 
    else:
        test_dataset = None

    return train_dataset, test_dataset                                           

print(animal_cifar100_get_datasets)
datasets = [
    {
        'name': 'ANIMALCIFAR100',
        'input': (3, 32, 32),
        'output': ('bear','beaver','bee', 'butterfly', 'camel', 'caterpillar', 'cattle', 'chimpanzee', 'cockroach', 'crab', 'crocodile', 'dolphin', 'elephant', 'flatfish', 'hamster',
                    'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
                    'kangaroo', 'leopard','lizard', 'lion', 'seal', 'shark'),
        'loader': animal_cifar100_get_datasets,
    },
]



#animal_data_loader = torch.utils.data.DataLoader(animal_dataset, batch_size=16, shuffle=True, num_workers=2)

# You can now use the animal_data_loader to train your model on the animal classes of CIFAR100
