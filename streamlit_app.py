
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import SimpleMnistModel

def transform_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image)
    return image.unsqueeze(0)

st.title('MNIST Digit Classification')
st.write('Upload an image to classify it using a pre-trained feedforward neural network.')

uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    image_tensor = transform_image(image)
    
    net = SimpleMnistModel()
    net.load_state_dict(torch.load('mnist_net.pth'))
    net.eval()
    
    with torch.no_grad():
        outputs = net(image_tensor)
        _, predicted = torch.max(outputs, 1)
        st.write(f'Prediction: {predicted.item()}')
