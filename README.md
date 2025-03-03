## Introduction

Layer-wise Relevance Propagation (LRP) is an explainable AI technique designed to visualize and interpret the decision-making process of neural networks. Developed to address the "black box" nature of deep learning models, LRP works by backpropagating the prediction score through the network layers to determine which input features (such as pixels in an image) contributed most significantly to the final classification decision.

LRP assigns relevance scores to each input feature, revealing which parts of the input were most influential in the network's decision. This visualization helps us understand:
- Which regions of an image were decisive for classification
- What patterns the network has learned to recognize
- Potential biases or unexpected decision criteria in the model

Beyond images, LRP can be applied to various data types, including text, time series, and tabular data, making it a versatile tool for model interpretability across domains.

## Training the miniVGG Network

For our experiment, we implemented a miniVGG network architecture, a simplified version of the original VGGNet but retaining key design principles like small convolutional filters and increasing depth.

### Model Architecture
As we see, we used some dropouts to prevent overfitting and RelU activations; the simplified model architecture is the following:
```Conv -> Conv -> MaxPool -> Conv -> Conv -> MaxPool -> FC -> FC```

```python
class MiniVGG(nn.Module):
    def __init__(self, num_classes=10):
        super(MiniVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.45),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Forward pass through features
        x = self.features(x)
        x = self.classifier(x)
        return x

model = MiniVGG(num_classes=10).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0004)
```

### Training Process

We trained the network on the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 classes. We trained the model for 14 epochs.

```python
#Number of epochs
epochs = 14
train_loss_history = []
val_loss_history = []
val_acc_history = []

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation step
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
```

Our model achieved approximately 80% accuracy on the test set after training.

## Applying the LRP Algorithm

To explain the model's decisions, we implemented LRP using the `zennit` library, which offers various interpretation methods for neural networks.

### Setting up the LRP Analyzer

```python
from zennit.composites import EpsilonPlusFlat
from zennit.attribution import Gradient

composite = EpsilonPlusFlat(epsilon=1e-6)

# Create an Attribution object
attr = Gradient(model, composite)

example_loader = DataLoader(test_set, batch_size=5, shuffle=True)
images, labels = next(iter(example_loader))
images, labels = images.cuda(), labels.cuda()
```
We specifically used the LRP-ε variant (EpsilonPlusFlat), which stabilizes the propagation by adding a small ε term in the denominator, preventing divisions by zero while maintaining the conservation principle.

### Libraries Used

- **Pytorch**: For building and training the neural network
- *Zennit*: For implementing the LRP interpretability method
- **NumPy**: For array operations
- **Matplotlib**: For visualization of results

## Results Analysis

The LRP analysis produced heatmaps highlighting the regions of input images that contributed most significantly to the classification decisions. Below are some key observations:

1. **Focused Attention**: The network correctly focused on distinctive features of objects (e.g., wings for birds, windows for cars)
2. **Background Discrimination**: The model learned to largely ignore background elements when making classifications
3. **Feature Hierarchy**: Different layers showed attention to different aspects - early layers focused on edges, while deeper layers captured more semantic features
4. The heatmaps use a color spectrum where red indicates positive relevance (features supporting the classification) and blue indicates negative relevance (features contradicting the classification).

Next, we have some examples of the results:

For the first example, we'll look at the test image of the plane. The model correctly identifies it as a plane. The corresponding heatmap shows concentrated red areas highlighting significant features such as the wing shape and fuselage contours, indicating that the model successfully leverages these key characteristics for its decision-making process.

![Example 1](Examples/output3.png)

In the second example, the image shows a car. The model correctly predicts the class and focuses on relevant features, such as the wheels and overall shape. The heatmap confirms that the model disregards the less important background, reinforcing its ability to isolate the critical components needed for accurate classification.

![Example 2](Examples/output2.png)

The third example presents a more challenging scenario. The input image is of a truck viewed from behind, yet the model misclassifies it as a car. By applying LRP, we can see that the heatmap highlights features common to cars, such as the shape and structure that the model has learned to associate with that class. This example illustrates the model's confusion and underscores the need for further refinement, especially in handling edge cases and ambiguous perspectives.

![Example 3](Examples/output4.png)

Here, the input image is a cat, but the model predicts it as a deer. At first, this might seem confusing, but applying LRP helps us understand why the model got it wrong. We see that the model focuses on certain textures in the image. Specifically, the two areas at the bottom, which seem to correspond to parts of the cat's fur, could resemble deer horns in shape or pattern. The model likely learned to associate similar textures with deer during training. Interestingly, the model does not focus on key cat features like the face shape, ears, or whiskers—things that a human would easily use to identify a cat. Instead, it gives importance to the wrong textures, leading to this misclassification.

This example highlights the value of LRP. By visualizing where the model is looking, we can identify its mistakes and figure out how to fix them.

![Example 4](Examples/output1.png)


## Conclusion

Layer-wise Relevance Propagation proved to be an effective tool for understanding our miniVGG model's decision-making process. Although the model sometimes misclassifies images, the LRP visualizations provide clear insights into the underlying reasons for these errors.

LRP helps bridge the gap between the high performance of deep neural networks and their interpretability, making them more trustworthy for critical applications. This method can be particularly valuable in domains like medical imaging or autonomous driving, where understanding why a model made a specific decision is as important as the decision itself.

Future work could explore comparing different LRP rule variants and extending this interpretability approach to more complex architectures and datasets.
