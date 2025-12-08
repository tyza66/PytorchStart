# teacher 已提前训练好并保存为 teacher_best.pth
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4)

# 2. 模型
teacher = models.resnet50(num_classes=10).to(device)
teacher.load_state_dict(torch.load('teacher_best.pth'))
teacher.eval()  # 老师只推理
for p in teacher.parameters():
    p.requires_grad = False

student = models.resnet18(num_classes=10).to(device)

# 3. 损失 & 优化器
criterion_hard = nn.CrossEntropyLoss()
criterion_soft = nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
T, alpha = 4.0, 0.7  # 温度与软损失权重

# 4. 训练循环
epochs = 30
for epoch in range(epochs):
    student.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            t_logits = teacher(x)  # 老师 logit
        s_logits = student(x)  # 学生 logit

        # soft loss（KL，需对 student 做 log_softmax）
        soft_loss = criterion_soft(F.log_softmax(s_logits / T, dim=1),
                                   F.softmax(t_logits / T, dim=1)) * (T ** 2)

        # hard loss
        hard_loss = criterion_hard(s_logits, y)

        loss = alpha * soft_loss + (1 - alpha) * hard_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = s_logits.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

    print(f'Epoch {epoch + 1:02d}  loss={running_loss / len(train_loader):.4f}  '
          f'acc={100. * correct / total:.2f}%')
