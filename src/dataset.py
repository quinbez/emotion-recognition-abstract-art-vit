from pathlib import Path
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# ArtEmisDataset
class ArtEmisDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / row['filename']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row['label'], dtype=torch.long)
        return image, label
    
def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_test_transforms

def get_dataloaders(train_df, val_df, test_df, image_dir, batch_size=32):
    train_transforms, val_test_transforms = get_transforms()
    
    train_loader = DataLoader(
        ArtEmisDataset(train_df, image_dir, train_transforms),
        batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        ArtEmisDataset(val_df, image_dir, val_test_transforms),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        ArtEmisDataset(test_df, image_dir, val_test_transforms),
        batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    return train_loader, val_loader, test_loader