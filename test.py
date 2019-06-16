import numpy as np
import torchvision


import torch
import os
import numpy as np
#import argparse
from PIL import Image

from torchvision import transforms
from torch.autograd import Variable
from torchvision import utils as vutils
from network.Transformer import Transformer
import shutil

#parser = argparse.ArgumentParser()
#parser.add_argument('--input_dir', default = 'test_img')
#parser.add_argument('--load_size', default = 450)
#parser.add_argument('--model_path', default = './pretrained_model')
# parser.add_argument('--style', default = 'Shinkai') # Hayao, Hosoda, Paprika, Shinkai
#parser.add_argument('--output_dir', default = 'output_lcy')
#parser.add_argument('--gpu', type=int, default = 0)

#opt = parser.parse_args()

params = {
        'input_dir': 'test_img',
        'load_size': 450,
        'model_path': './pretrained_model',
        # 'style': 'Shinkai',
        'output_dir': 'test_output',
        'gpu': 0
        }

valid_ext = ['.jpg', '.png']

# if not os.path.exists(opt.output_dir): os.mkdir(opt.output_dir)
shutil.rmtree(params['output_dir'])
os.makedirs(params['output_dir'])

# load pretrained model
model1 = Transformer()
model1.load_state_dict(torch.load(os.path.join(params['model_path'], 'Hayao_net_G_float.pth')))
model1.eval()

model2 = Transformer()
model2.load_state_dict(torch.load(os.path.join(params['model_path'], 'Hosoda_net_G_float.pth')))
model2.eval()

model3 = Transformer()
model3.load_state_dict(torch.load(os.path.join(params['model_path'], 'Paprika_net_G_float.pth')))
model3.eval()

model4 = Transformer()
model4.load_state_dict(torch.load(os.path.join(params['model_path'], 'Shinkai_net_G_float.pth')))
model4.eval()

if params['gpu'] > -1:
    print('GPU mode')
    model1.cuda()
    model2.cuda()
    model3.cuda()
    model4.cuda()
else:
    print('CPU mode')
    model1.float()
    model2.float()
    model3.float()
    model4.float()

for files in os.listdir(params['input_dir']):
    ext = os.path.splitext(files)[1]
    if ext not in valid_ext:
        continue
    # load image
    input_image = Image.open(os.path.join(params['input_dir'], files)).convert("RGB")
    # resize image, keep aspect ratio
    h = input_image.size[0]
    w = input_image.size[1]
    ratio = h *1.0 / w
    if ratio > 1:
        h = params['load_size']
        w = int(h*1.0/ratio)
    else:
        w = params['load_size']
        h = int(w * ratio)
    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)
    # RGB -> BGR
    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    # preprocess, (-1, 1)
    input_image = -1 + 2 * input_image 
    with torch.no_grad():
        if params['gpu'] > -1:
            input_image = input_image.cuda()
        else:
            input_image = input_image.float()
        
    # forward
    
    output_image1 = model1(input_image)
    output_image1 = output_image1[0]
    # BGR -> RGB
    output_image1 = output_image1[[2, 1, 0], :, :]
    # deprocess, (0, 1)
    output_image1 = output_image1.data.cpu().float() * 0.5 + 0.5
    # save
    vutils.save_image(output_image1, os.path.join(params['output_dir'], files[:-4] + '_' + 'Hayao' + '.jpg'))
    
    output_image2 = model2(input_image)
    output_image2 = output_image2[0]
    # BGR -> RGB
    output_image2 = output_image2[[2, 1, 0], :, :]
    # deprocess, (0, 1)
    output_image2 = output_image2.data.cpu().float() * 0.5 + 0.5
    # save
    vutils.save_image(output_image2, os.path.join(params['output_dir'], files[:-4] + '_' + 'Hosoda' + '.jpg'))
    
    output_image3 = model3(input_image)
    output_image3 = output_image3[0]
    # BGR -> RGB
    output_image3 = output_image3[[2, 1, 0], :, :]
    # deprocess, (0, 1)
    output_image3 = output_image3.data.cpu().float() * 0.5 + 0.5
    # save
    vutils.save_image(output_image3, os.path.join(params['output_dir'], files[:-4] + '_' + 'Paprika' + '.jpg'))
    
    output_image4 = model4(input_image)
    output_image4 = output_image4[0]
    # BGR -> RGB
    output_image4 = output_image4[[2, 1, 0], :, :]
    # deprocess, (0, 1)
    output_image4 = output_image4.data.cpu().float() * 0.5 + 0.5
    # save
    vutils.save_image(output_image4, os.path.join(params['output_dir'], files[:-4] + '_' + 'Shinkai' + '.jpg'))

print('Finished!')
