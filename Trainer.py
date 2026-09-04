import torch

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
        train_loss, train_acc = 0.0, 0.0
        self.model.train()
        
        batch_grad_norms = []

        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()
            train_acc += self.calculate_accuracy(y_true=y, y_pred=y_pred.argmax(dim=1))
            
            self.optimizer.zero_grad()
            loss.backward()

            # Record gradient norms across parameters
            grad_norms = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norms[name] = param.grad.detach().norm(2).item()
            batch_grad_norms.append(grad_norms)

            self.optimizer.step()

        train_loss /= len(data_loader)
        train_acc /= len(data_loader)

        # Calculate average gradient norm for the epoch
        mean_grad_norms = {}
        if batch_grad_norms:
            keys = batch_grad_norms[0].keys()
            for k in keys:
                mean_grad_norms[k] = sum(b[k] for b in batch_grad_norms) / len(batch_grad_norms)

        # Extract sharpness parameters (lambda)
        lambdas = []
        for module in self.model.modules():
            if hasattr(module, 'lambda_param'):
                lambdas.append(module.lambda_param.detach().cpu().item())
            elif hasattr(module, 'get_lambda'):
                lambdas.append(module.get_lambda().detach().cpu().item())

        if epoch is not None and epoch % self.loss_steps == 0:
            print(f"Epoch {epoch:02d} | Training Loss: {train_loss:.5f} | Training Accuracy: {train_acc:.2f}%")
            
        return train_loss, train_acc, mean_grad_norms, lambdas

    def test(self, data_loader: torch.utils.data.DataLoader, epoch: int = None):
        test_loss, test_acc = 0.0, 0.0
        self.model.to(self.device)
        self.model.eval()
        with torch.inference_mode():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                test_pred = self.model(X)
                loss = self.loss_fn(test_pred, y)
                test_loss += loss.item()
                test_acc += self.calculate_accuracy(y_true=y, y_pred=test_pred.argmax(dim=1))

            test_loss /= len(data_loader)
            test_acc /= len(data_loader)
            if epoch is not None and epoch % self.loss_steps == 0:
                print(f"Epoch {epoch:02d} | Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")
            return test_loss, test_acc
