
import os
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from traenslenzor.font_detector.font_size_model.data_gen import FONT_CONFIGS, render_text_box

# Configuration
FONTS = [
    "Roboto-Regular",
    "RobotoMono-Regular",
    "Inter-Regular",
    "Lato-Regular",
    "IBMPlexSans-Regular",
]
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001
SAMPLES_PER_FONT = 2000  # 10k total images

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "classifier"
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "classifier"

def generate_classifier_data():
    """Generate synthetic images for classification."""
    print("Generating training data...")
    
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)

    # Text source (random words/sentences)
    # We'll use a mix of random characters and lorem ipsum-like structure
    words = [
        "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog",
        "Lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
        "1234567890", "Payment", "Invoice", "Contract", "Agreement", "Date", "Signature",
        "Section", "Article", "Clause", "Party", "Whereas", "Hereby", "Shall", "May"
    ]

    for font_name in FONTS:
        print(f"  Generating {font_name}...")
        font_dir = DATA_DIR / font_name
        font_dir.mkdir()
        
        # FONT_CONFIGS is {name: [path1, path2...]}
        font_path = FONT_CONFIGS[font_name][0]
        
        # Fix path if it's relative to data_gen.py
        if not os.path.exists(font_path):
             # Try relative to this script
             font_path = str(BASE_DIR / "fonts" / Path(font_path).name)

        for i in tqdm(range(SAMPLES_PER_FONT)):
            # Random text length
            n_words = random.randint(1, 10)
            text = " ".join(random.choices(words, k=n_words))
            
            # Random font size
            font_size = random.randint(10, 40)
            
            try:
                img, _, _ = render_text_box(
                    text, font_path, font_size, padding=random.randint(0, 10), max_width=800
                )
                
                # Convert to RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Save
                img.save(font_dir / f"{i}.png")
            except Exception as e:
                print(f"Error generating sample: {e}")

class FontDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        
        for cls_name in self.classes:
            cls_dir = root_dir / cls_name
            for img_path in cls_dir.glob("*.png"):
                self.images.append((str(img_path), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train():
    # 1. Generate Data
    generate_classifier_data()
    
    # 2. Setup Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Transforms: Resize to 224x224 (Letterbox-style or simple resize)
    # For simplicity and speed, we'll use simple resize for now,
    # but a smart pad-then-resize would be better.
    # Let's use a transform that pads to square then resizes.
    
    class SquarePad:
        def __call__(self, image):
            w, h = image.size
            max_wh = max(w, h)
            hp = int((max_wh - w) / 2)
            vp = int((max_wh - h) / 2)
            padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
            return transforms.functional.pad(image, padding, fill=255, padding_mode='constant')

    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = FontDataset(DATA_DIR, transform=transform)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 3. Model Setup
    print("Initializing ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(FONTS))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        print(f"  Train Loss: {running_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        print(f"  Val Acc: {val_acc:.2f}%")

    # 5. Save Model
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CHECKPOINT_DIR / "resnet18_fonts.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes,
        'class_to_idx': dataset.class_to_idx
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()
