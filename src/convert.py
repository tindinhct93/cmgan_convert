import torch
from models.generator import TSCNet

n_fft = 400  
model = TSCNet(num_channel=64, num_features=n_fft//2+1).cuda()
model.eval()
dummy_input = torch.rand([1, 2, 1151, 201]).cuda()
torch.onnx.export(model, dummy_input, "denseEncoder.onnx")