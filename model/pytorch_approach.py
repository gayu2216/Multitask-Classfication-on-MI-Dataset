import torch
import torch.nn as nn
import torch.nn.functional as F

class MIMultitaskNet(nn.Module):
    def __init__(self, input_dim=111, num_complications=11, num_mortality_classes=7):
        super(MIMultitaskNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.prelu1 = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.prelu2 = nn.PReLU()
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(256, 128)
        self.prelu3 = nn.PReLU()
        self.bn3 = nn.BatchNorm1d(128)
        
        self.complications_head = nn.Linear(128, num_complications)
        self.mortality_head = nn.Linear(128, num_mortality_classes)

    def forward(self, x):
        x = self.dropout1(self.bn1(self.prelu1(self.fc1(x))))
        x = self.dropout2(self.bn2(self.prelu2(self.fc2(x))))
        x = self.bn3(self.prelu3(self.fc3(x)))
        
        complications = self.complications_head(x)
        mortality = self.mortality_head(x)
        
        return complications, mortality

model = MIMultitaskNet()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

criterion_complications = nn.BCEWithLogitsLoss() 
criterion_mortality = nn.CrossEntropyLoss()

def train_step(data, target_comp, target_mort):
    optimizer.zero_grad()
    
    out_comp, out_mort = model(data)
    
    loss_comp = criterion_complications(out_comp, target_comp)
    loss_mort = criterion_mortality(out_mort, target_mort)
    
    total_loss = (3.0 * loss_comp) + (1.0 * loss_mort)
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()