import cv2
import PIL
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1

from src.iou import BoundingBox, intersection_box, iou, bb_to_np


class BoxEncoder():
    def __init__(self, img_size: tuple[int, int] = (224, 224)):
        
        self.img_size = img_size
        self.model = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    @staticmethod
    def img_to_tensor(img_path: str) -> torch.Tensor:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        return img

    def encode_bounding_boxes(self, img: torch.Tensor, bb_list: list[BoundingBox]) -> torch.Tensor:
        crops = []
        with torch.no_grad():
            for bb in bb_list:
                left, top, width, height = bb_to_np(bb)
                left, top, width, height = int(left), int(top), int(width), int(height)
                cropped_img = img[:, top : top + height, left : left + width]
                cropped_img = transforms.ToPILImage()(cropped_img)
                cropped_img = self.transform(cropped_img)

                crops.append((cropped_img))
            return self.model(torch.stack(crops))
    
            
