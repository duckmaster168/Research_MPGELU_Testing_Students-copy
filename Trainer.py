import torch
from collections import defaultdict

class Trainer:
    '''
    Trainer class handling training loops, evaluation steps, layer-wise gradient norm collection,
    and sharpness parameter extraction.
    '''
    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 calculate_accuracy,
                 device: torch.device,
                 loss_steps: int = 1):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.calculate_accuracy = calculate_accuracy
        self.device = device
        self.loss_steps = loss_steps

    def train(self, data_loader: torch.utils.data.DataLoader, epoch: int = None):
        train_loss = 0.0
        train_acc = 0.0
        batch_count = 0

        self.model.to(self.device)
        self.model.train()

        batch_grad_norms = []

        for batch, (X, y) in enumerate(data_loader):
            batch_count += 1
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)

            train_loss += loss.item()

            # calculate_accuracy is expected to accept (y_true=..., y_pred=...)
            try:
                train_acc += self.calculate_accuracy(y_true=y, y_pred=y_pred.argmax(dim=1))
            except Exception:
                # fallback: assume calculate_accuracy returns fraction
                pred_labels = y_pred.argmax(dim=1)
                train_acc += (pred_labels == y).float().mean().item()

            self.optimizer.zero_grad()
            loss.backward()

            # Record gradient norms across parameters
            grad_norms = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.detach().norm(2).item()
            batch_grad_norms.append(grad_norms)

            self.optimizer.step()

        # Avoid division by zero
        if batch_count == 0:
            avg_loss = 0.0
            avg_acc = 0.0
        else:
            avg_loss = train_loss / batch_count
            avg_acc = train_acc / batch_count

        # Calculate average gradient norm for the epoch (handles missing keys)
        mean_grad_norms = {}
        if batch_grad_norms:
            sums = defaultdict(float)
            counts = defaultdict(int)
            for b in batch_grad_norms:
                for k, v in b.items():
                    sums[k] += v
                    counts[k] += 1
            for k in sums:
                mean_grad_norms[k] = sums[k] / counts[k]

        # Extract sharpness parameters (lambda)
        lambdas = []
        for module in self.model.modules():
            if hasattr(module, 'lambda_param'):
                try:
                    lambdas.append(module.lambda_param.detach().cpu().item())
                except Exception:
                    pass
            elif hasattr(module, 'get_lambda') and callable(getattr(module, 'get_lambda')):
                try:
                    lam = module.get_lambda()
                    if isinstance(lam, torch.Tensor):
                        lambdas.append(lam.detach().cpu().item())
                    else:
                        # try to convert
                        lambdas.append(float(lam))
                except Exception:
                    pass

        if epoch is not None and self.loss_steps and epoch % self.loss_steps == 0:
            print(f"Epoch {epoch:02d} | Training Loss: {avg_loss:.5f} | Training Accuracy: {avg_acc:.2f}%")

        return avg_loss, avg_acc, mean_grad_norms, lambdas

    def test(self, data_loader: torch.utils.data.DataLoader, epoch: int = None):
        test_loss = 0.0
        test_acc = 0.0
        batch_count = 0

        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for X, y in data_loader:
                batch_count += 1
                X, y = X.to(self.device), y.to(self.device)
                test_pred = self.model(X)
                loss = self.loss_fn(test_pred, y)
                test_loss += loss.item()
                try:
                    test_acc += self.calculate_accuracy(y_true=y, y_pred=test_pred.argmax(dim=1))
                except Exception:
                    test_acc += (test_pred.argmax(dim=1) == y).float().mean().item()

        if batch_count == 0:
            avg_loss = 0.0
            avg_acc = 0.0
        else:
            avg_loss = test_loss / batch_count
            avg_acc = test_acc / batch_count

        if epoch is not None and self.loss_steps and epoch % self.loss_steps == 0:
            print(f"Epoch {epoch:02d} | Test Loss: {avg_loss:.5f} | Test Accuracy: {avg_acc:.2f}%")

        return avg_loss, avg_acc
