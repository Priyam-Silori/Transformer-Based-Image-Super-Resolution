import torch
import SRTransG
import model

netG = SRTransG.SRTransG()
netD = model.Discriminator()

x = torch.randn(1, 3, 64, 64)

with torch.no_grad():
    y = netG(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)