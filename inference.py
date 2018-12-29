import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms

from model import Generator
from style_model import StyleGenerator
from opts import InferenceOptions
from torchvision.utils import save_image
import torch

def main(opts):
    # Create the model
    if opts.type == 'style':
        G = StyleGenerator().to(opts.device)
    else:
        G = Generator().to(opts.device)
    state = torch.load(opts.resume)
    G.load_state_dict(state['G'])

    # Generate!
    result = []
    for i in range(opts.num_face):
        z = torch.randn([1, 512, 1, 1]).to(opts.device)
        fake_img = G(z)
        result.append(fake_img[0])
    result = torch.stack(result, 0)
    save_image(result.data, opts.det, nrow = 5, normalize = True)

if __name__ == '__main__':
    opts = InferenceOptions().parse()
    main(opts)
