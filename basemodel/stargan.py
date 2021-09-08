from network import model_stargan
import torch

def get_rafd_stargan(version='v4'):
    model = model_stargan.Generator(conv_dim=64, c_dim=8, repeat_num=6)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('I2I_model/stargan-rafd-aug-{}.pth'.format(version)))
    return model

if __name__ == "__main__":
    pass
