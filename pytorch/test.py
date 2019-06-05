import torch
x = torch.rand(5,3)
print(x)

flag = torch.cuda.is_available()

print("cuda is available :%d" %(flag))
