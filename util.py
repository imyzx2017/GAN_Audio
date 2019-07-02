import torchvision.datasets as Datasets
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
audio_path = 'D:/Projects/Projects/pytorch_Projects/GAN/Datasets/audio2spec_datasets0315UsingResize_512'


dataset = Datasets.ImageFolder(root=audio_path,
                               transform=transforms.Compose([
                                   transforms.ToTensor()
                               ]))
# print(dataset)
data_loader = DataLoader(dataset,
                    batch_size=4,
                    shuffle=True,
                    drop_last=True
                    )
