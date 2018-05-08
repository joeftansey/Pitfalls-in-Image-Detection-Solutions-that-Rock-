# Pitfalls in Image Detection: Solutions that Rock!

A video 5 minute presentation is available [here](https://www.youtube.com/watch?v=ZAWS0fOCNmg).

Convolutional neural networks (CNN) are the current state of the art solution to the image classification problem. When balanced, distinct classes are present, CNN produce highly accurate and reliable results. However, outlier and minority classes, often not present in training sets, can easily produce human-obvious missclassifications. This is a significant issue for autonomous vehicles, where human lives depend on classifier accuracy. This problem is also relevant in security, where attackers can engineer "adversarial example" images fool classifiers. Comprehensive training sets and confidence thresholding are shown to be good solutions to the outlier missclassification problem.

The workflow is split between three independent jupyter notebooks. Each notebook trains a convolutional nueral net, which is very taxing on memory. If running the notebooks, I recommend that you close each notebook completely before starting the next one. A powerful GPU and the appropriate drivers is also recommended. See the [PyTorch documentation](https://pytorch.org/docs/master/cuda.html) for more details. 

