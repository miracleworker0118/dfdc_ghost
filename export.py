from pathlib import Path
import torch
from torch import nn, optim
import timm


input_model_path = 'last_models/best_model.pth'
output_model_path = 'last_models/best_model.onnx'
IMAGE_SHAPE = (3,256,256)
model = timm.create_model('ghostnetv2_160', pretrained=True)

model.classifier = nn.Linear(model.classifier.in_features, 1)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model = nn.DataParallel(model, device_ids=[0, 1])

sample_x = torch.randn(*((1,) + IMAGE_SHAPE))

model.load_state_dict(torch.load(input_model_path))
model_to_export = model.module
torch.onnx.export(
    model_to_export,
    sample_x,
    output_model_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input':{0:'batch_size'},
        'output':{0:'batch_size'},
    }
)