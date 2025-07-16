import torch
import torch.nn.functional as F
from typing import Dict, List
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import autocast

class StructuredBM:
    def __init__(self, n_vis, n_branch, n_hid, n_grp, temp, device='cuda'):
        self.device = device
        self.a = torch.zeros(n_vis, device=device)
        self.c = torch.zeros(n_branch, device=device)
        self.d = torch.zeros(n_hid, device=device)
        self.e = torch.full((n_grp,), -1.0, device=device)
        self.T = torch.tensor(temp, device=device)
        
        self.W = 0.01 * torch.randn(n_vis, n_branch, device=device)
        self.U = 0.01 * torch.randn(n_branch, n_hid, device=device)
        self.V = 0.01 * torch.randn(n_hid, n_grp, device=device)
        self.R = 0.01 * torch.randn(n_grp, n_grp, device=device)
        self.R.fill_diagonal_(0.)
    
    def sample_b(self, v, h):
        p = torch.sigmoid(self.c + v @ self.W + h @ self.U.T)
        return (torch.rand_like(p) < p).float()
    
    def sample_h(self, b, g):
        p = torch.sigmoid(self.d + b @ self.U + g @ self.V.T)
        return (torch.rand_like(p) < p).float()
    
    def sample_g(self, h, g_prev):
        p = torch.sigmoid((self.e + h @ self.V) / self.T + g_prev @ self.R.T)
        return (torch.rand_like(p) < p).float()
    
    def sample_v(self, b):
        p = torch.sigmoid(self.a + b @ self.W.T)
        return (torch.rand_like(p) < p).float()
    
    def cd_k(self, v0, k=5, lr=1e-3):
        # Positive phase
        b_pos = self.sample_b(v0, torch.zeros((v0.size(0), self.d.size(0)), device=self.device))
        h_pos = self.sample_h(b_pos, torch.zeros((v0.size(0), self.e.size(0)), device=self.device))
        g_pos = self.sample_g(h_pos, torch.zeros((v0.size(0), self.e.size(0)), device=self.device))
        # Negative phase
        v_neg, b_neg, h_neg, g_neg = v0.clone(), b_pos.clone(), h_pos.clone(), g_pos.clone()
        for _ in range(k):
            v_neg = self.sample_v(b_neg)
            b_neg = self.sample_b(v_neg, h_neg)
            h_neg = self.sample_h(b_neg, g_neg)
            g_neg = self.sample_g(h_neg, g_neg)
        # Gradients (fp32 for stability)
        with torch.cuda.amp.autocast(enabled=False):
            dW = (v0.T @ b_pos - v_neg.T @ b_neg) / v0.size(0)
            dU = (b_pos.T @ h_pos - b_neg.T @ h_neg) / v0.size(0)
            dV = (h_pos.T @ g_pos - h_neg.T @ g_neg) / v0.size(0)
            dR = (g_pos.T @ g_pos - g_neg.T @ g_neg) / v0.size(0); dR.fill_diagonal_(0.)
            da = torch.mean(v0 - v_neg, dim=0)
            dc = torch.mean(b_pos - b_neg, dim=0)
            dd = torch.mean(h_pos - h_neg, dim=0)
            de = torch.mean(g_pos - g_neg, dim=0)
        self.W += lr * dW
        self.U += lr * dU
        self.V += lr * dV
        self.a += lr * da
        self.c += lr * dc
        self.d += lr * dd
        self.e += lr * de
        self.R += lr * dR
    
    def structural_plasticity(self, theta_prune=0.01, p_add=0.02, sigma=0.01, recent_activities=None):
        self.W[torch.abs(self.W) < theta_prune] = 0
        self.U[torch.abs(self.U) < theta_prune] = 0
        
        if recent_activities is not None:
            v_acts = recent_activities['v']
            b_acts = recent_activities['b']
            h_acts = recent_activities['h']
            # Corrcoef in PyTorch (simple version)
            def corrcoef(x, y):
                x_mean = x.mean(dim=0, keepdim=True)
                y_mean = y.mean(dim=0, keepdim=True)
                x_centered = x - x_mean
                y_centered = y - y_mean
                cov = (x_centered.T @ y_centered) / (x.shape[0] - 1)
                std_x = torch.std(x, dim=0, keepdim=True)
                std_y = torch.std(y, dim=0, keepdim=True)
                corr = cov / (std_x.T * std_y + 1e-8)
                return corr
            corr_W = corrcoef(v_acts, b_acts)
            zero_mask_W = (self.W == 0)
            cand_scores = corr_W[zero_mask_W]
            if cand_scores.numel() == 0:
                return
            th = torch.quantile(cand_scores, 1 - p_add)
            high_mask_flat = (corr_W > th) & zero_mask_W
            idx_vis, idx_br = torch.nonzero(high_mask_flat, as_tuple=True)
            num_add = idx_vis.numel()
            if num_add > 0:
                new_weights = torch.randn(num_add, device=self.device) * sigma
                self.W[idx_vis, idx_br] = new_weights
            corr_U = corrcoef(b_acts, h_acts)
            zero_mask_U = (self.U == 0)
            th = torch.quantile(corr_U[zero_mask_U], 1 - p_add)
            high_mask_flat = (corr_U > th) & zero_mask_U
            idx_br, idx_hid = torch.nonzero(high_mask_flat, as_tuple=True)
            num_add = idx_br.numel()
            if num_add > 0:
                new_weights = torch.randn(num_add, device=self.device) * sigma
                self.U[idx_br, idx_hid] = new_weights

def load_mnist_binary(npz_path='data/mnist.npz'):
    data = np.load(npz_path)
    train_images = data['x_train']
    train_labels = data['y_train']
    test_images = data['x_test']
    test_labels = data['y_test']
    # Flatten and normalize to [0,1]
    train_images = train_images.reshape(-1, 784).astype(np.float32) / 255.0
    test_images = test_images.reshape(-1, 784).astype(np.float32) / 255.0
    # Filter 0 and 1, binarize (threshold 0.5)
    train_mask = (train_labels == 0) | (train_labels == 1)
    test_mask = (test_labels == 0) | (test_labels == 1)
    train_data = (train_images[train_mask] > 0.5).astype(np.float32)
    test_data = (test_images[test_mask] > 0.5).astype(np.float32)
    return train_data, test_data

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = StructuredBM(784, 400, 150, 5, [1.2, 1.0, 0.8, 0.6, 0.4], device=device)
    train_data_np, test_data_np = load_mnist_binary()
    train_data = torch.from_numpy(train_data_np).to(device)
    test_data = torch.from_numpy(test_data_np).to(device)
    batch_size = 500
    num_epochs = 100
    lr = 1e-3
    
    activities: Dict[str, List[torch.Tensor]] = {'v': [], 'b': [], 'h': []}
    
    for epoch in range(num_epochs):
        lr_current = 1e-3 * 0.5 * (
            1 + torch.cos(torch.pi * epoch / num_epochs)
        )
        activities = {'v': [], 'b': [], 'h': []}
        batch_count = 0
        for v0 in train_data:
            v0 = v0.to(device)
            with autocast(enabled=True, dtype=torch.float16):
                model.cd_k(v0, k=5, lr=lr_current.item())
            if batch_count < 5:
                with autocast(enabled=True, dtype=torch.float16):
                    b = model.sample_b(v0, torch.zeros((v0.shape[0], model.d.shape[0]), device=device))
                    h = model.sample_h(b, torch.zeros((v0.shape[0], model.e.shape[0]), device=device))
                activities['v'].append(v0)
                activities['b'].append(b)
                activities['h'].append(h)
                batch_count += 1
        if activities['v']:
            recent_acts = {k: torch.cat(activities[k], dim=0) for k in activities}
            model.structural_plasticity(theta_prune=0.005, p_add=0.05, sigma=0.02, recent_activities=recent_acts)
        
        v_test = test_data[:100]
        with autocast(enabled=True, dtype=torch.float16):
            b_test = model.sample_b(v_test, torch.zeros((100, model.d.shape[0]), device=device))
            v_recon = model.sample_v(b_test)
        recon_error = torch.mean((v_test - v_recon)**2).item()
        print(f"Epoch {epoch+1}: Reconstruction error = {recon_error}")
    
    # Evaluation with kNN and t-SNE (on CPU for sklearn)
    train_h = []
    for i in range(0, len(train_data), batch_size):
        v = train_data[i:i+batch_size]
        with autocast(enabled=True, dtype=torch.float16):
            b = model.sample_b(v, torch.zeros((v.shape[0], model.d.shape[0]), device=device))
            h = model.sample_h(b, torch.zeros((v.shape[0], model.e.shape[0]), device=device))
        train_h.append(h.cpu().numpy())
    train_h = np.vstack(train_h)
    
    test_h = []
    for i in range(0, len(test_data), batch_size):
        v = test_data[i:i+batch_size]
        with autocast(enabled=True, dtype=torch.float16):
            b = model.sample_b(v, torch.zeros((v.shape[0], model.d.shape[0]), device=device))
            h = model.sample_h(b, torch.zeros((v.shape[0], model.e.shape[0]), device=device))
        test_h.append(h.cpu().numpy())
    test_h = np.vstack(test_h)
    
    data = np.load('data/mnist.npz')
    train_labels = data['y_train']
    test_labels = data['y_test']
    train_mask = (train_labels == 0) | (train_labels == 1)
    test_mask = (test_labels == 0) | (test_labels == 1)
    train_labels_filtered = train_labels[train_mask]
    test_labels_filtered = test_labels[test_mask]
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_h, train_labels_filtered)
    pred = knn.predict(test_h)
    acc = accuracy_score(test_labels_filtered, pred)
    print(f"kNN Accuracy: {acc}")
    
    test_b = []
    test_g = []
    for i in range(0, len(test_data), batch_size):
        v = test_data[i:i+batch_size]
        with autocast(enabled=True, dtype=torch.float16):
            b = model.sample_b(v, torch.zeros((v.shape[0], model.d.shape[0]), device=device))
            h = model.sample_h(b, torch.zeros((v.shape[0], model.e.shape[0]), device=device))
            g = model.sample_g(h, torch.zeros((v.shape[0], model.e.shape[0]), device=device))
            # Burn-in for g
            for _ in range(2):
                g = model.sample_g(h, g)
        test_b.append(b.cpu().numpy())
        test_g.append(g.cpu().numpy())
    test_b = np.vstack(test_b)
    test_g = np.vstack(test_g)
    tsne = TSNE(n_components=2)
    b_embed = tsne.fit_transform(test_b[:1000])
    plt.figure()
    for m in range(model.e.shape[0]):
        mask = test_g[:1000, m] > 0.5
        plt.scatter(b_embed[mask, 0], b_embed[mask, 1], label=f'Group {m}')
    plt.legend()
    plt.savefig('tsne_b_groups.png')
    print("t-SNE plot saved to tsne_b_groups.png") 