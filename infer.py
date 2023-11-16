from model import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

test_transform = A.Compose([
    A.Resize(height = 800, width = 1120),
    ToTensorV2()
    
])


checkpoint_path = "modelUNet_ep_30.pth"
model = create_pretrained_model(checkpoint_path)
# model = ResUnet(3)
image_path = input("Enter image link (Availabel image: test_image.jpeg): ")


img = Image.open(image_path)
img_np = np.array(img)
image = test_transform(image = img_np)['image']

image_input = image.unsqueeze(0) / 255
ouput = model(image_input)
plt.imshow(F.one_hot(torch.argmax(ouput.squeeze(), 0).cpu()).float())
plt.show()