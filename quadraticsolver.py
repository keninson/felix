import torch
from torch import nn
from MyNNModels import QuadraticFunctionModel
from PltUtilities import PltUtilities
import matplotlib.pyplot as plt


weight1, weight2, bias = 50, 5, 50.10
start = -4
end = 4
step = 0.05
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight1 * (X**2) + weight2 * X + bias
print(y)

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

torch.manual_seed(1)
model = QuadraticFunctionModel()
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
epochs = 40000

print(model.state_dict())
data_visualizer = PltUtilities()
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        if epoch % 100 == 0:
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test)
            print(
                f"Epoch {epoch} | Loss {loss.detach().item()} | Test Loss {test_loss}"
            )
            if epoch % 10000 == 0:
                full_graph = model(X)
                print(model.state_dict())
                data_visualizer.load_data(X_train, y_train, X_test, y_test, test_pred)
                data_visualizer.show(title=f"Epoch {epoch}")
                plt.figure(figsize=(10, 7))
                l1 = plt.plot(X, y, c="g", label="Изначальный график")
                l2 = plt.plot(
                    X, full_graph, c="r", label="График построенный нейросетью"
                )
                plt.legend(prop={"size": 14})
                plt.show()

with torch.inference_mode():
    y_preds = model(X_test)
    data_visualizer.load_data(X_train, y_train, X_test, y_test, test_pred)
    data_visualizer.show(title="Final Result")
    print(model.state_dict())

    full_graph = model(X)
    plt.figure(figsize=(10, 7))
    l1 = plt.plot(X, y, c="g", label="Изначальный график")
    l2 = plt.plot(X, full_graph, c="r", label="График построенный нейросетью")
    plt.legend(prop={"size": 14})
    plt.show()
