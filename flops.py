import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from model import efficientnetv2_s as create_model

with torch.cuda.device(0):
  # net = models.densenet161()
  net = create_model(num_classes=37)
  macs, params = get_model_complexity_info(net, ((3, 224, 224),(3, 224, 224),(3, 224, 224)), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))