import torch
from torchviz import make_dot

x = torch.tensor([[1.0, 2.0]], requires_grad=True)

w0 = torch.tensor([[0.5, -0.3]], requires_grad=True)
b0 = torch.tensor([0.1], requires_grad=True)
w1 = torch.tensor([[0.2, 0.8]], requires_grad=True)
b1 = torch.tensor([-0.2], requires_grad=True)
w2 = torch.tensor([[0.7, 0.5]], requires_grad=True)
b2 = torch.tensor([0.3], requires_grad=True)

z0 = x @ w0.T + b0
z1 = x @ w1.T + b1
z2 = x @ w2.T + b2

a0 = torch.relu(z0)
a1 = torch.relu(z1)
a2 = torch.relu(z2)

out1 = torch.cat([a0, a1, a2], dim=1)
print("Layer 1 output:", out1)

u0 = torch.tensor([[0.6, -0.1]], requires_grad=True)
c0 = torch.tensor([0.05], requires_grad=True)
u1 = torch.tensor([[-0.4, 0.9]], requires_grad=True)
c1 = torch.tensor([0.2], requires_grad=True)

h0 = x @ u0.T + c0
h1 = x @ u1.T + c1

s0 = torch.sigmoid(h0)
s1 = torch.sigmoid(h1)

out2 = torch.cat([s0, s1], dim=1)
print("Layer 2 output:", out2)

combined = out1 + out2
final = torch.tanh(combined)
print("Final output:", final)

y = torch.tensor([[0.5, 0.7, 0.2]], requires_grad=False)
loss = torch.mean((final - y[:, :final.size(1)])**2)
print("Loss:", loss.item())

loss.backward()

graph = make_dot(loss, params={
    'x': x, 'w0': w0, 'b0': b0, 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2,
    'u0': u0, 'c0': c0, 'u1': u1, 'c1': c1
})
graph.render("ai_platform_graph", format="png")
