from PIL import Image
import os
import random
import numpy as np
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
from torch.nn.parallel.data_parallel import DataParallel

import clip
from clip.simple_tokenizer import SimpleTokenizer

_tokenizer = SimpleTokenizer()
device = 'cuda:1'

def load_clip_to_cpu(cfg):
    backbone_name = cfg['model_name']
    url = clip.clip._MODELS[backbone_name]
    model_path = clip.clip._download(url, "~/.cache/clip")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.model.build_model(state_dict or model.state_dict())

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg['n_ctx']
        ctx_init = cfg['ctx_init']
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg['input_size']
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            print("random initialization")
            # random initialization
            if cfg['csc']:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        ctx_vectors = torch.load('/ckb-nfs/home/zcafego/ctx_vectors.pt')
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg['class_token_position']

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


def load_checkpoint(p, cstm_clip):
    checkpoint = torch.load(p, map_location=device)
    state_dict = checkpoint['state_dict']
    if "token_prefix" in state_dict:
        del state_dict["token_prefix"]

    if "token_suffix" in state_dict:
        del state_dict["token_suffix"]

    cstm_clip.load_state_dict(state_dict, strict=False)

def ready_model(cfg, categories, device, state_dict_path):
    print("Loading CLIP model...")
    model, preproc = clip.load(cfg['model_name'], device=device)
    print("Creating custom CLIP...")
    cstm_clip = CustomCLIP(cfg, categories, model)
    print("Created!\n")

    # for _, param in cstm_clip.named_parameters():

    #     param.requires_grad_(False)
    
    print("Loading state dict...")
    load_checkpoint(state_dict_path, cstm_clip)
    print("Loaded!\n")
    cstm_clip.prompt_learner.eval()

    return cstm_clip, preproc

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


cfg = {
    'n_ctx': 16,
    'ctx_init': '',
    'input_size': 224,
    'csc': False,
    'class_token_position': 'end',
    'model_name': 'ViT-L/14'
}

state_dict_path = '/ckb-nfs/home/zcafego/git/CoOp/output/imagenet/CoOp/vit_l14_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
# images = [
#     Image.open('/ckb-nfs/home/zcafego/imagenet/images/val/n01440764/ILSVRC2012_val_00000293.JPEG'), # tench
#     Image.open('/ckb-nfs/home/zcafego/imagenet/images/val/n01443537/ILSVRC2012_val_00000236.JPEG'), # goldfish
#     Image.open('/ckb-nfs/home/zcafego/imagenet/images/val/n01484850/ILSVRC2012_val_00002338.JPEG'), # great white shark
#     ]

categories = []
with open('/ckb-nfs/home/zcafego/custom_clip_classnames.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        categories.append(line)

category_codes = {}
with open('/ckb-nfs/home/zcafego/imagenet/classnames.txt') as f:
    for line in f.readlines():
        line = line.strip()
        classification = line.split(' ')[0]
        category_codes[classification] = ' '.join(line.split(' ')[1:])

set_random_seed(1)
image_directory = '/ckb-nfs/home/zcafego/imagenet/images/val/'

custom_clip_model = torch.load('/ckb-nfs/home/zcafego/custom_clip_model.pt').to(device)
their_custom_clip = CustomCLIP(cfg, categories, custom_clip_model).to(device)
_, preproc = clip.load('ViT-L/14')
with torch.no_grad():
    with open('coop_test_results.txt', 'w') as f:
        f.write("CoOp Results:\n\n")
        results = {}
        num_categories = 0
        for path, folder, files in os.walk(image_directory):
            if 'n' not in path.split('/')[-1]:
                continue
            num_categories += 1
            category = category_codes[path.split('/')[-1]]
            results[category] = 0
            for file in files:
                image = Image.open(os.path.join(path, file))
                img = preproc(image).to(device)
                result = their_custom_clip(img[None, ...])
                pred = result.max(1)[1].item()
                if categories[pred] == category:
                    results[category] += 1
            print(f"{category}: {results[category]}")
            f.write(f"{category}: {results[category]}/50\n")
        
        correct = sum(results.values())
        f.write(f"\n\nAccuracy Results:\nTotal: 50,000\nCorrect: {correct:,}\nAccuracy: {correct/50000:.1%}")
    






    # for f in tqdm(os.listdir(image_directory)):
    #     image = Image.open(os.path.join(image_directory, f))
    #     img = preproc(image).to(device)
    #     result = their_custom_clip(img[None, ...])
    #     pred = result.max(1)[1].item()
    #     if categories[pred] not in results:
    #         results[categories[pred]] = 0
    #     results[categories[pred]] += 1
    
    # print(results)

# with torch.no_grad():
#     cstm_clip, preproc = ready_model(cfg, categories, device, state_dict_path)

#     with open('custom_named_generator_cuda.txt', 'w') as f:
#         for n, p in cstm_clip.named_parameters():
#             f.write(str(p.size()) + '\n')
#             f.write(str(p) + '\n')

    # debug_image = torch.load('/ckb-nfs/home/zcafego/debug_tensor.pt').to(device)
    # print(f"Input dims: {debug_image.size()}, {debug_image.dtype}")
    # debug_output = torch.load('/ckb-nfs/home/zcafego/debug_output.pt').to(device)
    # print(f"Debug output dims: {debug_output.size()}, {debug_output.dtype}")
    # debug_image_features = torch.load('/ckb-nfs/home/zcafego/debug_image_features.pt').to(device)
    # print(f"Debug image feature dims: {debug_image_features.size()}, {debug_image_features.dtype}")

    # result = cstm_clip(debug_image)
    # print(f"Result output dims: {result.size()}, {result.dtype}")
    # print(torch.equal(debug_image_features, result.to(dtype=torch.float16)))
    # print(torch.isclose(debug_image_features, result.to(dtype=torch.float16)))

    # results = {}
    # for f in tqdm(os.listdir(image_directory)):
    #     image = Image.open(os.path.join(image_directory, f))
    #     img = preproc(image).to(device)
    #     result = cstm_clip(img[None, ...])
    #     pred = result.max(1)[1].item()
    #     if categories[pred] not in results:
    #         results[categories[pred]] = 0
    #     results[categories[pred]] += 1
    
    # print(results)


# with open('/ckb-nfs/home/zcafego/model_module.pkl', 'rb') as inp:
#     module = pickle.load(inp)

print("Done")