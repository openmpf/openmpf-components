import torch
import clip
from PIL import Image
import torch
import os

import trainers

device = 'cuda:1'

vitl14_path = os.path.join(os.getcwd(), 'vit_l14_ep50_16shots/nctx16_cscFalse_ctpend/seed3/prompt_learner/model.pth.tar-50')
vitb32_path = os.path.join(os.getcwd(), 'vit_b32_ep50_1shots/nctx16_cscFalse_ctpend/seed3/prompt_learner/model.pth.tar-50')

vitl14_check = torch.load(vitl14_path, map_location=None)

vitl14_state_dict = vitl14_check['state_dict']
vitl14_tnsr = vitl14_state_dict['ctx']

images = [
    Image.open('/ckb-nfs/home/zcafego/test_images/sturgeon.JPEG'),
    Image.open('/ckb-nfs/home/zcafego/test_images/val2017/000000000139.jpg'),
    Image.open('/ckb-nfs/home/zcafego/test_images/val2017/000000000285.jpg'),
    ]

# l14_model, l14_preprocessor = clip.load('ViT-L/14', device=device)

categories = []
with open('/ckb-nfs/home/zcafego/imagenet_labels/synset_words.txt') as f:
    for line in f.readlines():
        line = line.strip()
        categories.append(' '.join(line.split(' ')[1:]).split(', ')[0])

tokens = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device)

print(vitl14_check.keys())

def get_classifications_l14(img):
    # for img in images:
    preproc_img = l14_preprocessor(img).unsqueeze(0).to(device)
    return l14_model(preproc_img)

        # with torch.no_grad():
            # image_features = l14_model.encode_image(preproc_img)
            # text_features = l14_model.encode_text(tokens)

        
        # image_features /= image_features.norm(dim=-1, keepdim=True)
        # text_features /= text_features.norm(dim=-1, keepdim=True)
        # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)    
        # values, indices = similarity[0].topk(5)
        
        # print("\nTop predictions:\n")
        # for value, index in zip(values, indices):
        #     print(f"{categories[index]:>20s}: {100 * value.item():.2f}%")

# print("WITHOUT STATE DICT:\n")
# get_classifications_l14(images)

# l14_model.load_state_dict(vitl14_state_dict, strict=False)
# print(get_classifications_l14(images[0]))
# print("\nWITH STATE DICT:\n")
# get_classifications_l14(images[0])