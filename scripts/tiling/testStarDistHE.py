from stardist.models import StarDist2D
from stardist.plot import render_label

import PIL
import matplotlib.pyplot as plt
import numpy as np

# Load the model

model = StarDist2D.from_pretrained('2D_versatile_he')

# Load the image


image = PIL.Image.open("/omics/odcf/analysis/OE0585_projects/chromothripsis/histopathology/tiles/1024/TCGA/ffpe/TCGA-B0-5709-01Z-00-DX1/TCGA-B0-5709-01Z-00-DX1_30720_61440.jpg")
image_np = np.asarray(image)

labels, _ = model.predict_instances(image_np/255.0)



plt.subplot(1,2,1)
plt.imshow(img)
plt.axis("off")
plt.title("input image")

plt.subplot(1,2,2)
plt.imshow(render_label(labels, img=img))
plt.axis("off")
plt.title("prediction + input overlay")
