import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Function to load the image and preprocess it
def load_img(img_path, max_size=400, shape=None):
    # Open the image
    img = Image.open(img_path).convert("RGB")

    # Resize it
    if max(img.size) > max_size:
        size = max_size
    else:
        size = max(img.size)
    if shape is not None:
        size = shape

    # Transform pipeline
    trans = transforms.Compose([
        transforms.Resize(size),  # Resize image
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize
    ])
    img = trans(img)[:3, :, :].unsqueeze(0)  # Limit to RGB channels and add batch dimension
    return img

# Convert tensor back to an image
def tensor_to_image(tensor):
    # Copy tensor to CPU, detach from computation graph
    img = tensor.clone().detach().cpu().numpy().squeeze()
    img = img.transpose(1, 2, 0)  # Rearrange dimensions
    img = img * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)  # Undo normalization
    img = img.clip(0, 1)  # Clamp values between 0 and 1
    return img

# Calculate Gram matrix
def gram_matrix(input_tensor):
    # Get dimensions
    _, d, h, w = input_tensor.size()
    # Flatten spatial dimensions
    tensor = input_tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())  # Matrix multiplication

# Feature extraction function
def extract_features(img, model, layers=None):
    if layers is None:  # Define default layers if none provided
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content representation
            '28': 'conv5_1'
        }
    features = {}
    x = img
    for name, layer in model._modules.items():
        x = layer(x)  # Pass input through each layer
        if name in layers:
            features[layers[name]] = x
    return features

# Loss calculation functions
def calc_content_loss(target_feat, content_feat):
    return torch.mean((target_feat - content_feat) ** 2)

def calc_style_loss(target_feat, style_grams):
    loss = 0
    for layer in style_grams:
        target_gram = gram_matrix(target_feat[layer])
        style_gram = style_grams[layer]
        loss += torch.mean((target_gram - style_gram) ** 2)
    return loss

# Load pre-trained VGG19 model
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
for p in vgg.parameters():
    p.requires_grad = False  # Freeze the model

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# Paths to content and style images
content_path = r"D:\.vscode\CODTECH\content.jpeg"
style_path = r"D:\.vscode\CODTECH\style.jpg"

# Load images
content_img = load_img(content_path).to(device)
style_img = load_img(style_path, shape=content_img.shape[-2:]).to(device)

# Initialize target as a clone of the content image
target_img = content_img.clone().requires_grad_(True).to(device)

# Hyperparameters
content_weight = 1e5
style_weight = 1e-3
learning_rate = 0.003
num_epochs = 300

# Optimizer
optimizer = torch.optim.Adam([target_img], lr=learning_rate)

# Extract features for content and style
content_feats = extract_features(content_img, vgg)
style_feats = extract_features(style_img, vgg)
style_grams = {layer: gram_matrix(style_feats[layer]) for layer in style_feats}

# Training loop
for epoch in range(num_epochs):
    # Get features for the target image
    target_feats = extract_features(target_img, vgg)

    # Calculate content and style losses
    content_loss = calc_content_loss(target_feats['conv4_2'], content_feats['conv4_2'])
    style_loss = calc_style_loss(target_feats, style_grams)

    # Compute total loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    # Backpropagation and optimization
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Print loss every 50 epochs
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}")

# Convert the final tensor to an image
final_output = tensor_to_image(target_img)

# Show the final image
plt.imshow(final_output)
plt.axis("off")
plt.title("Stylized Output")
plt.show()

# Save the stylized image
final_output_img = Image.fromarray((final_output * 255).astype("uint8"))
final_output_img.save("stylized_output.jpg")
print("Image saved as 'stylized_output.jpg'")
