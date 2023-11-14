import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
import torch.nn as nn
import torch.nn.functional as F

print(torch.version.__version__)

# @torch.jit.script
# def compute(x, y):
#     if bool(x[0][0] == 42):
#         z = 5
#     else:
#         z = 10
#     return x.matmul(y) + z

class Compute(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        if bool(x[0][0] == 42):
            z = 5
        else:
            z = 10
        return x.matmul(y) + z

model = Compute()
model.eval()

scripted_module = torch.jit.script(model)
optimized_scripted_module = optimize_for_mobile(scripted_module)
optimized_scripted_module._save_for_lite_interpreter("app/src/main/assets/compute.ptl")
