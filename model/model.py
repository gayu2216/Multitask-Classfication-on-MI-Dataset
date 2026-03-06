import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import joblib

def prepare_data(csv_path):
    df = pd.read_csv(csv_path, header=None, na_values='?')
    
    X = df.iloc[:, 1:93] 
    y_comp = df.iloc[:, 112:123].fillna(0)
    y_mort = df.iloc[:, 123].fillna(0)
    
    imputer = IterativeImputer(max_iter=10, random_state=42)
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    X_train, X_test, y_comp_train, y_comp_test, y_mort_train, y_mort_test = train_test_split(
        X_scaled, y_comp, y_mort, test_size=0.2, random_state=42
    )
    
    dummy_idx = np.arange(len(X_train))
    X_with_idx = np.column_stack([X_train, dummy_idx])
    
    smote = SMOTE(random_state=42)
    X_resampled_with_idx, y_mort_train_bal = smote.fit_resample(X_with_idx, y_mort_train)
    
    X_train_bal = X_resampled_with_idx[:, :-1]
    resampled_indices = X_resampled_with_idx[:, -1].astype(int)
    y_comp_train_bal = y_comp_train.iloc[resampled_indices]
    
    return (torch.FloatTensor(X_train_bal), 
            torch.FloatTensor(X_test), 
            torch.FloatTensor(y_comp_train_bal.values.copy()), 
            torch.LongTensor(y_mort_train_bal.values.copy()),
            scaler,
            torch.FloatTensor(y_comp_test.values.copy()),
            torch.LongTensor(y_mort_test.values.copy()))

class FTTransformer(nn.Module):
    def __init__(self, num_features=92, embedding_dim=32, n_heads=4, n_layers=3, num_mortality=8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, embedding_dim))
        self.bias = nn.Parameter(torch.zeros(num_features, embedding_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, 
            dim_feedforward=embedding_dim*4, dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        flat_dim = num_features * embedding_dim
        self.comp_head = nn.Linear(flat_dim, 11)   
        self.mort_head = nn.Linear(flat_dim, num_mortality) 

    def forward(self, x):
        x = x.unsqueeze(-1) * self.weight + self.bias
        x = self.transformer(x)
        x = x.view(x.size(0), -1) 
        return self.comp_head(x), self.mort_head(x)

def train_model(train_x, train_y_comp, train_y_mort):
    device = torch.device("cpu")
    model = FTTransformer(num_features=92, num_mortality=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    criterion_comp = nn.BCEWithLogitsLoss()
    criterion_mort = nn.CrossEntropyLoss()
    
    model.train()
    epochs = 50
    print(f"Starting Multitask Training (Complications + Mortality) on CPU...")

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        optimizer.zero_grad()
        out_comp, out_mort = model(train_x)
        
        loss_comp = criterion_comp(out_comp, train_y_comp)
        loss_mort = criterion_mort(out_mort, train_y_mort)
        total_loss = (1.0 * loss_comp) + (3.0 * loss_mort)
        
        total_loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {total_loss.item():.4f}")

    print("Training Complete.")
    return model

def evaluate_results(model, test_x, test_y_comp, test_y_mort):
    model.eval()
    with torch.no_grad():
        out_comp, out_mort = model(test_x)
        pred_comp = (torch.sigmoid(out_comp) > 0.2).int().cpu().numpy()
        pred_mort = torch.argmax(out_mort, dim=1).cpu().numpy()

    print("\n" + "="*50)
    print("TASK 1: COMPLICATIONS (Multi-label Results)")
    print("="*50)
    print(classification_report_imbalanced(test_y_comp.cpu().numpy(), pred_comp, zero_division=0))

    print("\n" + "="*50)
    print("TASK 2: MORTALITY (Multi-class Results)")
    print("="*50)
    mort_names = ["Survival"] + [f"Cause_{i}" for i in range(1, 8)]
    print(classification_report_imbalanced(test_y_mort.cpu().numpy(), pred_mort, target_names=mort_names, zero_division=0))

def save_clinical_model(model, scaler, filename="mi_multitask_model.pth"):
    torch.save(model.state_dict(), filename)
    joblib.dump(scaler, "clinical_scaler.pkl")
    print(f"\n[SUCCESS] Model saved to {filename}")
    print("[SUCCESS] Scaler saved to clinical_scaler.pkl")

if __name__ == "__main__":
    train_x, test_x, train_y_comp, train_y_mort, scaler, test_y_comp, test_y_mort = prepare_data("MI.data")
    model = train_model(train_x, train_y_comp, train_y_mort)
    save_clinical_model(model, scaler)
    evaluate_results(model, test_x, test_y_comp, test_y_mort)
