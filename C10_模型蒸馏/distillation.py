# teacher 已提前训练好并保存为 teacher_best.pth
# 模型蒸馏（Knowledge Distillation）就是把一个“大老师”模型的知识迁移到一个“小学生”模型里，让小模型既轻量又能复现大模型的效果。
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4)

# 模型
teacher = models.resnet50(num_classes=10).to(device)
teacher.load_state_dict(torch.load('teacher_best.pth'))
teacher.eval()  # 老师只推理
for p in teacher.parameters():
    p.requires_grad = False  # 冻结老师参数

student = models.resnet18(num_classes=10).to(device)  # 学生模型 默认是不带预训练的

# 损失 & 优化器
criterion_hard = nn.CrossEntropyLoss()  # 交叉熵损失
criterion_soft = nn.KLDivLoss(reduction='batchmean')  # KL 散度损失
optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)  # Adam动量自适应优化器   90% 的实验/论文里 Adam 就是默认第一选择
T, alpha = 4.0, 0.7  # 温度（温度高 → 老师把分数“打得很散”，连“很像”和“一点点像”都告诉、温度低 → 老师只告诉你“最像”的那个）与软损失权重 （最终的总损失 = 软损失 × 0.7 + 硬损失 × 0.3）

# 训练循环
epochs = 30
for epoch in range(epochs):
    student.train()  # 学生要训练
    running_loss, correct, total = 0.0, 0, 0
    # 读取本轮的数据
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            t_logits = teacher(x)  # 老师 logit 拿到老师推理结果 不需要梯度
        s_logits = student(x)  # 学生 logit 学生需要梯度 拿到学生推理结果


        # 经典 KD 损失计算

        # soft loss（KL，需对 student 做 log_softmax） 软损失（考虑到像的程度）
        soft_loss = criterion_soft(F.log_softmax(s_logits / T, dim=1),
                                   F.softmax(t_logits / T, dim=1)) * (T ** 2)

        # hard loss
        hard_loss = criterion_hard(s_logits, y) # 硬损失（只管正确与否）

        loss = alpha * soft_loss + (1 - alpha) * hard_loss # 总损失 = 软损失 × 0.7 + 硬损失 × 0.3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = s_logits.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()

    print(f'Epoch {epoch + 1:02d}  loss={running_loss / len(train_loader):.4f}  '
          f'acc={100. * correct / total:.2f}%')
