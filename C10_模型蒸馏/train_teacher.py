import torch, torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ========== 1. 数据 ==========
def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    train_set = datasets.CIFAR10(root='data', train=True,  download=True, transform=transform)
    test_set  = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=4)
    test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False, num_workers=4)
    return train_loader, test_loader

# ========== 2. 训练 + 验证 ==========
def train_teacher():
    train_loader, test_loader = get_data()
    model = models.resnet50(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_acc = 0.0
    for epoch in range(50):
        # ---- 训练 ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f'Epoch{epoch+1:02d}'):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = out.max(1)
            total   += y.size(0)
            correct += pred.eq(y).sum().item()
        train_acc = 100.*correct/total
        print(f'[train] loss={running_loss/len(train_loader):.3f}  acc={train_acc:.2f}%')

        # ---- 验证 ----
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                _, pred = out.max(1)
                correct += pred.eq(y).sum().item()
                total   += y.size(0)
        val_acc = 100.*correct/total
        print(f'[val]   acc={val_acc:.2f}%')
        scheduler.step(val_acc)

        # ---- 保存最佳 ----
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'teacher_best.pth')
            print(f'>>> save best teacher with acc={best_acc:.2f}%')

# ========== 3. 保护入口 ==========
if __name__ == '__main__':
    train_teacher()