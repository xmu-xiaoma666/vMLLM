import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPTokenizer, CLIPTextModel, CLIPModel


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        # self.select_layer = -1
        
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.clip_model = CLIPModel.from_pretrained(self.vision_tower_name, device_map=device_map)

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        # self.vision_tower = self.clip_model.text_model


        self.tokenizer = CLIPTokenizer.from_pretrained(self.vision_tower_name)
        self.text_tower = CLIPTextModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        # self.text_tower = self.clip_model.vision_model

        self.vision_tower.requires_grad_(False)
        self.text_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # print(self.select_layer)
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features, image_forward_outs

    @torch.no_grad()
    def forward_text(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)
        # Pass the inputs through the model to get text features
        text_forward_out = self.text_tower(**inputs.to(self.device), output_hidden_states=True)
        # # get cls token( PLAN A)
        # last_hidden_state = text_forward_out.last_hidden_state
        # text_cls_embeddings = last_hidden_state[:, :1, :]
        # get cls token( PLAN B)
        text_cls_embeddings = text_forward_out.pooler_output.unsqueeze(1)
        
        # project text feature
        text_cls_embeddings = self.clip_model.text_projection(text_cls_embeddings)
        return text_cls_embeddings, text_forward_out


    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature, image_forward_outs = self.feature_select(image_forward_out)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features, image_forward_outs = self.feature_select(image_forward_outs)

        # project img feature
        # image_features = self.clip_model.visual_projection(image_features)
        return image_features.to(images.dtype), image_forward_outs

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
