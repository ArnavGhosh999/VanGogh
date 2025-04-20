import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import io
import time
import base64
from streamlit_lottie import st_lottie
import json
import requests
import matplotlib.pyplot as plt

# Initialize session state to store persistent data between reruns
if 'output_image' not in st.session_state:
    st.session_state.output_image = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'loss_history' not in st.session_state:
    st.session_state.loss_history = None
if 'intermediate_results' not in st.session_state:
    st.session_state.intermediate_results = []

# Set page config and title
st.set_page_config(
    page_title="Van Gogh Style Transfer App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load animations
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load different lottie animations - using more dynamic animations
artist_lottie = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_gj6yxzk9.json")  # Animated brush
paintbrush_lottie = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_ksz8hlec.json")  # Color splash
processing_lottie = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_ycfkgtmz.json")  # More dynamic processing
transformation_lottie = load_lottieurl("https://assets7.lottiefiles.com/temp/lf20_dgjK9i.json")  # Transformation animation

# Add custom CSS for styling with enhanced animations
st.markdown("""
<style>
    /* Background styling - black with gradient overlay */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #0f0f1a 100%);
        background-attachment: fixed;
    }
    
    /* Enhanced title animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes glowPulse {
        0% { text-shadow: 0 0 10px rgba(233, 69, 96, 0.5); }
        50% { text-shadow: 0 0 20px rgba(233, 69, 96, 0.8), 0 0 30px rgba(233, 69, 96, 0.4); }
        100% { text-shadow: 0 0 10px rgba(233, 69, 96, 0.5); }
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50% }
        50% { background-position: 100% 50% }
        100% { background-position: 0% 50% }
    }
    
    .main-title {
        text-align: center;
        font-size: 3.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #e94560, #ff8a71, #e94560);
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeInUp 1.2s ease-out, gradientShift 5s ease infinite;
        font-weight: 800;
    }
    
    .sub-title {
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 2rem;
        color: #f0f0f0;
        animation: fadeInUp 1.5s ease-out;
        opacity: 0.85;
    }
    
    /* Enhanced image containers */
    .images-row {
        display: flex;
        justify-content: space-around;
        margin-bottom: 2rem;
        animation: fadeInUp 1.2s ease-out;
    }
    
    .result-container {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
        animation: fadeInUp 1.5s ease-out;
    }
    
    /* Glassmorphism image cards with subtle hover animations */
    .image-card {
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
        padding: 1rem;
        background-color: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(7px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        max-height: 380px;
        overflow: hidden;
        width: 90%;
        margin: 0 auto;
        border: 1px solid rgba(255, 255, 255, 0.07);
    }
    
    .image-card img {
        max-height: 310px;
        object-fit: contain;
        margin: 0 auto;
        display: block;
        border-radius: 8px;
        transition: transform 0.5s ease;
    }
    
    .image-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.5), 0 0 15px rgba(233, 69, 96, 0.2);
        border: 1px solid rgba(233, 69, 96, 0.2);
    }
    
    .image-card:hover img {
        transform: scale(1.03);
    }
    
    .image-title {
        text-align: center;
        font-size: 1.3rem;
        margin-bottom: 0.8rem;
        color: #f0f0f0;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    /* Enhanced result card */
    .result-card {
        border-radius: 16px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4), 0 0 20px rgba(233, 69, 96, 0.15);
        padding: 1.2rem;
        background: linear-gradient(135deg, rgba(233, 69, 96, 0.1) 0%, rgba(15, 52, 96, 0.1) 100%);
        backdrop-filter: blur(10px);
        transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        max-height: 500px;
        overflow: hidden;
        width: 85%;
        margin: 0 auto;
        border: 1px solid rgba(233, 69, 96, 0.15);
        animation: fadeInUp 1.2s ease-out;
    }
    
    .result-card img {
        max-height: 400px;
        object-fit: contain;
        margin: 0 auto;
        display: block;
        border-radius: 10px;
        transition: transform 0.6s ease;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    .result-card:hover img {
        transform: scale(1.02);
    }
    
    .result-title {
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #e94560, #ff8a71);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    /* Enhanced progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #e94560, #ff8a71);
        transition: width 0.4s ease;
    }
    
    /* Enhanced processing animation */
    .processing-animation {
        width: 100%;
        height: 4px;
        background: linear-gradient(to right, #e94560, #0f3460, #e94560);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        margin-bottom: 1rem;
        border-radius: 2px;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Enhanced process button */
    .btn-process {
        background: linear-gradient(90deg, #e94560, #ff647f);
        color: white;
        padding: 0.9rem 1.8rem;
        border: none;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        display: block;
        margin: 1.5rem auto;
        width: 85%;
        max-width: 320px;
        letter-spacing: 1px;
        box-shadow: 0 10px 20px rgba(233, 69, 96, 0.3);
    }
    
    .btn-process:hover {
        background: linear-gradient(90deg, #d43152, #e94560);
        transform: translateY(-3px) scale(1.03);
        box-shadow: 0 15px 30px rgba(233, 69, 96, 0.4);
    }
    
    .btn-process:active {
        transform: translateY(1px);
    }
    
    /* General containers with enhanced animations */
    .upload-section {
        animation: fadeInUp 1.2s ease-out;
    }
    
    .lottie-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1.5rem;
        transform-origin: center;
        animation: pulse 2s infinite alternate;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        100% { transform: scale(1.05); }
    }
    
    /* Enhanced evolution display */
    .evolution-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 18px;
        margin-top: 30px;
        background: linear-gradient(135deg, rgba(15, 52, 96, 0.15) 0%, rgba(23, 37, 84, 0.15) 100%);
        padding: 25px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .evolution-image {
        border: 1px solid rgba(233, 69, 96, 0.2);
        border-radius: 10px;
        padding: 6px;
        max-width: 160px;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        background-color: rgba(255, 255, 255, 0.03);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .evolution-image:hover {
        transform: scale(1.15) translateY(-5px);
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.3), 0 0 10px rgba(233, 69, 96, 0.2);
        border: 1px solid rgba(233, 69, 96, 0.3);
        z-index: 10;
    }
    
    /* Fixed sizing for images with enhanced transitions */
    .stImage img {
        max-height: 420px;
        object-fit: contain;
        transition: all 0.4s ease;
    }
    
    /* Enhanced loss history styling */
    .loss-history {
        margin-top: 40px;
        margin-bottom: 40px;
        border: 1px solid rgba(233, 69, 96, 0.1);
        border-radius: 12px;
        padding: 20px;
        background: linear-gradient(135deg, rgba(15, 52, 96, 0.15) 0%, rgba(23, 37, 84, 0.15) 100%);
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* Enhanced download button styling */
    .stDownloadButton button {
        background: linear-gradient(90deg, #e94560, #ff647f) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.8rem 1.5rem !important;
        margin: 1.5rem auto !important;
        display: block !important;
        width: 100% !important;
        max-width: 280px !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        box-shadow: 0 10px 20px rgba(233, 69, 96, 0.3) !important;
    }
    
    .stDownloadButton button:hover {
        background: linear-gradient(90deg, #d43152, #e94560) !important;
        transform: translateY(-3px) scale(1.03) !important;
        box-shadow: 0 15px 30px rgba(233, 69, 96, 0.4) !important;
    }
    
    /* Enhanced section headers */
    h3 {
        color: #f0f0f0 !important;
        text-align: center !important;
        margin-top: 2.5rem !important;
        margin-bottom: 1.8rem !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        background: linear-gradient(90deg, #e0e0e0, #f0f0f0) !important;
        -webkit-background-clip: text !important;
        background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    
    /* Enhanced expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, rgba(233, 69, 96, 0.1), rgba(23, 37, 84, 0.1)) !important;
        border-radius: 10px !important;
        color: #f0f0f0 !important;
        font-weight: 600 !important;
        padding: 12px 15px !important;
        transition: all 0.3s ease !important;
        border: 1px solid rgba(233, 69, 96, 0.1) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(90deg, rgba(233, 69, 96, 0.15), rgba(23, 37, 84, 0.15)) !important;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(15, 52, 96, 0.1) !important;
        border-radius: 0 0 10px 10px !important;
        padding: 15px !important;
        border: 1px solid rgba(233, 69, 96, 0.05) !important;
        border-top: none !important;
    }
    
    /* Enhanced sidebar styling */
    .css-6qob1r, .css-1544g2n {
        background: linear-gradient(180deg, #0a0a14 0%, #111122 100%) !important;
        border-right: 1px solid rgba(233, 69, 96, 0.1) !important;
    }
    
    .css-6qob1r label, .css-6qob1r p, .css-1544g2n label, .css-1544g2n p {
        color: #f0f0f0 !important;
        font-weight: 500 !important;
    }
    
    /* Enhanced tooltip styling */
    .stTooltip {
        background-color: rgba(20, 20, 35, 0.95) !important;
        border: 1px solid rgba(233, 69, 96, 0.3) !important;
        color: #f0f0f0 !important;
        border-radius: 8px !important;
        padding: 10px 15px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Enhanced footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 12px;
        background: linear-gradient(90deg, rgba(0, 0, 0, 0.8), rgba(10, 10, 20, 0.8));
        backdrop-filter: blur(10px);
        z-index: 1000;
        border-top: 1px solid rgba(233, 69, 96, 0.1);
    }
    
    /* Enhanced upload area */
    .drop-zone {
        border: 2px dashed rgba(233, 69, 96, 0.3);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        background-color: rgba(255, 255, 255, 0.02);
        margin-bottom: 20px;
    }
    
    .drop-zone:hover {
        border-color: rgba(233, 69, 96, 0.6);
        background-color: rgba(233, 69, 96, 0.05);
    }
    
    /* File uploader styling */
    .stFileUploader > div > button {
        background: linear-gradient(90deg, #e94560, #ff647f) !important;
        color: white !important;
    }
    
    .stFileUploader > div {
        border: 1px solid rgba(233, 69, 96, 0.2) !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader > div:hover {
        border-color: rgba(233, 69, 96, 0.5) !important;
    }
    
    /* Animate hover effect on all buttons */
    button {
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    }
    
    button:hover {
        transform: translateY(-2px) !important;
    }
</style>
""", unsafe_allow_html=True)

def load_image(img_file, max_size=512):
    """Load and preprocess an uploaded image"""
    if img_file is not None:
        img = Image.open(img_file).convert('RGB')
        
        # Resize maintaining aspect ratio
        if max(img.size) > max_size:
            if img.width > img.height:
                size = (max_size, int(img.height * max_size / img.width))
            else:
                size = (int(img.width * max_size / img.height), max_size)
            img = img.resize(size, Image.LANCZOS)
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor, img
    return None, None

def tensor_to_image(tensor):
    """Convert tensor to numpy image for display"""
    image = tensor.cpu().clone().detach().numpy()
    image = image.squeeze(0)
    image = image.transpose(1, 2, 0)
    image = np.clip(image, 0, 1)
    return image

def gram_matrix(input_tensor):
    """Calculate Gram Matrix for style representation"""
    a, b, c, d = input_tensor.size()
    features = input_tensor.view(a, b, c * d)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(b * c * d)

def run_style_transfer(content_img, style_img, num_steps=100, style_weight=1000000, content_weight=1, device='cpu'):
    """Neural style transfer implementation with enhanced optimization"""
    # Start with a copy of the content image
    input_img = content_img.clone().detach().requires_grad_(True)
    
    # Load VGG19 model with newer weights parameter instead of pretrained
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    
    # Freeze the VGG parameters
    for param in vgg.parameters():
        param.requires_grad = False
        
    # Define layer mapping
    layer_mapping = {
        '0': 'conv1_1',   # Style layer
        '5': 'conv2_1',   # Style layer
        '10': 'conv3_1',  # Style layer
        '19': 'conv4_1',  # Style layer
        '21': 'conv4_2',  # Content layer
        '28': 'conv5_1'   # Style layer
    }
    
    style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    content_layers = ['conv4_2']
    
    # Precompute style features
    style_features = {}
    x = style_img
    for name, layer in enumerate(vgg):
        x = layer(x)
        if str(name) in layer_mapping and layer_mapping[str(name)] in style_layers:
            style_features[layer_mapping[str(name)]] = x
    
    # Calculate Gram matrices for style features
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # Precompute content features
    content_features = {}
    x = content_img
    for name, layer in enumerate(vgg):
        x = layer(x)
        if str(name) in layer_mapping and layer_mapping[str(name)] in content_layers:
            content_features[layer_mapping[str(name)]] = x
    
    # Use Adam optimizer with higher learning rate
    optimizer = torch.optim.Adam([input_img], lr=0.05)
    
    # Progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # For animation, store intermediate results
    intermediate_results = []
    loss_history = {"content": [], "style": [], "total": [], "step": []}
    
    # Optimization loop
    for step in range(num_steps):
        # Update progress
        progress = (step + 1) / num_steps
        progress_bar.progress(progress)
        status_text.text(f"Processing: Step {step+1}/{num_steps}")
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass through the model for current image
        current_features = {}
        x = input_img
        for name, layer in enumerate(vgg):
            x = layer(x)
            if str(name) in layer_mapping:
                layer_name = layer_mapping[str(name)]
                if layer_name in style_layers or layer_name in content_layers:
                    current_features[layer_name] = x
        
        # Content loss - comparing content representations
        content_loss = F.mse_loss(current_features[content_layers[0]], content_features[content_layers[0]])
        
        # Style loss - comparing gram matrices
        style_loss = 0
        for layer in style_layers:
            if layer in current_features:
                current_gram = gram_matrix(current_features[layer])
                layer_style_loss = F.mse_loss(current_gram, style_grams[layer])
                style_loss += layer_style_loss / len(style_layers)
        
        # Apply higher weight to make style loss more noticeable
        weighted_style_loss = style_weight * style_loss
        
        # Total loss
        total_loss = content_weight * content_loss + weighted_style_loss
        
        # Track losses
        loss_history["content"].append(content_loss.item())
        loss_history["style"].append(style_loss.item())
        loss_history["total"].append(total_loss.item())
        loss_history["step"].append(step)
        
        # Backward pass
        total_loss.backward()
        
        # Update image
        optimizer.step()
        
        # Clamp values to valid range
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        # Save intermediate result
        if step % max(1, num_steps // 5) == 0 or step == num_steps - 1:
            intermediate_results.append(input_img.detach().clone())
        
        # Report progress
        if step % 10 == 0:
            st.write(f"Step {step}, Content Loss: {content_loss.item():.6f}, Style Loss: {style_loss.item():.6f}")
    
    # Clear progress display
    progress_bar.empty()
    status_text.text("Style transfer completed! ✨")
    
    return input_img.detach(), intermediate_results, loss_history

# Page title and description with enhanced animation
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 class='main-title'>Van Gogh Style Transfer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Transform photos into artistic masterpieces</p>", unsafe_allow_html=True)
with col3:
    if artist_lottie:
        st_lottie(artist_lottie, height=180, key="artist_animation", speed=1.2)

# Create main upload section in the center
st.markdown("<div style='max-width: 900px; margin: 0 auto;'>", unsafe_allow_html=True)

# Upload sections side by side with enhanced styling
col1, col2 = st.columns([1, 1])
with col1:
    if paintbrush_lottie:
        st_lottie(paintbrush_lottie, height=100, key="paintbrush_animation", speed=1.3)
    st.markdown("<h4 style='text-align: center; color: #f0f0f0; font-weight: 600; letter-spacing: 1px;'>Content Image</h4>", unsafe_allow_html=True)
    st.markdown("<div class='drop-zone'>", unsafe_allow_html=True)
    content_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="content_upload")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if transformation_lottie:
        st_lottie(transformation_lottie, height=100, key="transform_animation", speed=1.3)
    st.markdown("<h4 style='text-align: center; color: #f0f0f0; font-weight: 600; letter-spacing: 1px;'>Style Image</h4>", unsafe_allow_html=True)
    st.markdown("<div class='drop-zone'>", unsafe_allow_html=True)
    style_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="style_upload")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Create a sidebar for parameters with enhanced styling
with st.sidebar:
    st.markdown("<h3 style='color: #e94560; text-align: center; font-weight: 700; letter-spacing: 1px; margin-bottom: 1.5rem;'>Style Transfer Controls</h3>", unsafe_allow_html=True)
    
    # Add some space and a styled divider
    st.markdown("<div style='margin: 1.5em 0; border-bottom: 1px solid rgba(233,69,96,0.3);'></div>", unsafe_allow_html=True)
    
    # Parameter sliders with enhanced styling
    st.markdown("<p style='color: #f0f0f0; font-weight: 600; letter-spacing: 1px;'>Processing Steps</p>", unsafe_allow_html=True)
    num_steps = st.slider("", min_value=20, max_value=150, value=50, step=5, 
                         help="More steps give better results but take longer to process")
    
    st.markdown("<p style='color: #f0f0f0; font-weight: 600; letter-spacing: 1px; margin-top: 2em;'>Style Strength</p>", unsafe_allow_html=True)
    style_weight = st.slider("", min_value=10000, max_value=2000000, value=800000, step=10000, format="%e",
                            help="Higher values emphasize the style more")
    
    # Add some space
    st.markdown("<div style='margin: 2.5em 0;'></div>", unsafe_allow_html=True)
    
    # Create a styled button using HTML
    process_button = st.button("✨ Generate Masterpiece ✨", key="process_button", 
                              use_container_width=True,
                              help="Run the style transfer algorithm")

# Main content area for displaying images with enhanced styling
if content_file is not None and style_file is not None:
    # Load the images
    content_tensor, content_img = load_image(content_file)
    style_tensor, style_img = load_image(style_file)
    
    # Display input images side by side with enhanced cards
    st.markdown("<h3 style='text-align: center;'>Input Images</h3>", unsafe_allow_html=True)
    col1, spacer, col2 = st.columns([10, 1, 10])
    
    with col1:
        st.markdown("<div class='image-card'><p class='image-title'>Content Image</p>", unsafe_allow_html=True)
        st.image(content_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='image-card'><p class='image-title'>Style Image</p>", unsafe_allow_html=True)
        st.image(style_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create placeholders for results
    result_container = st.container()
    loss_chart_container = st.empty()
    
    # Check if we already have processed results in the session state
    if st.session_state.processed and st.session_state.output_image is not None:
        # Display the result with enhanced styling
        with result_container:
            st.markdown("<h3 style='text-align: center;'>Stylized Result</h3>", unsafe_allow_html=True)
            
            # Display image with enhanced styling
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown("<p class='result-title'>Stylized Masterpiece</p>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.image(st.session_state.output_image, use_container_width=True, clamp=True)
            
            # Add a download button for the result
            buf = io.BytesIO()
            st.session_state.output_image.save(buf, format="PNG")
            
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col2:
                download_button = st.download_button(
                    label="💾 Download Masterpiece",
                    data=buf.getvalue(),
                    file_name="stylized_image.png",
                    mime="image/png",
                    key="download_button"
                )
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Show the progression steps and loss values in an enhanced container
            if len(st.session_state.intermediate_results) > 1:
                st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'>Style Transfer Evolution</h3>", unsafe_allow_html=True)
                
                # Create an enhanced container for the evolution display
                st.markdown("<div class='evolution-container'>", unsafe_allow_html=True)
                
                # Display sample loss values in an enhanced format
                loss_steps = [0, len(st.session_state.loss_history["step"])//4, len(st.session_state.loss_history["step"])//2, 
                              3*len(st.session_state.loss_history["step"])//4, len(st.session_state.loss_history["step"])-1]
                loss_data = []
                
                for step in loss_steps:
                    if step < len(st.session_state.loss_history["step"]):
                        idx = step
                        loss_data.append({
                            "Step": st.session_state.loss_history["step"][idx],
                            "Content Loss": f"{st.session_state.loss_history['content'][idx]:.6f}",
                            "Style Loss": f"{st.session_state.loss_history['style'][idx]:.6f}"
                        })
                
                # Explicitly set the frame size for the evolution frames
                st.markdown("""
                <style>
                .evolution-images img {
                    max-width: 150px !important;
                    max-height: 150px !important;
                    object-fit: contain;
                    margin: 8px !important;
                    border-radius: 12px !important;
                    box-shadow: 0 8px 20px rgba(0,0,0,0.3) !important;
                    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
                    border: 1px solid rgba(233, 69, 96, 0.2) !important;
                }
                
                .evolution-images img:hover {
                    transform: scale(1.15) translateY(-5px) !important;
                    box-shadow: 0 15px 30px rgba(0,0,0,0.4), 0 0 15px rgba(233, 69, 96, 0.3) !important;
                    z-index: 10 !important;
                    border: 1px solid rgba(233, 69, 96, 0.4) !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Only include a subset of frames to keep it manageable
                display_indices = list(range(0, len(st.session_state.intermediate_results), max(1, len(st.session_state.intermediate_results)//6)))
                if len(st.session_state.intermediate_results)-1 not in display_indices:
                    display_indices.append(len(st.session_state.intermediate_results)-1)
                
                display_frames = [st.session_state.intermediate_results[i] for i in display_indices]
                captions = [f"Step {i * (len(st.session_state.loss_history['step']) // len(st.session_state.intermediate_results))}" for i in display_indices]
                
                # Display with enhanced styling in a sleeker container
                col1, col2 = st.columns([3, 7])
                
                with col1:
                    st.markdown("<p style='color: #e94560; font-weight: 600; letter-spacing: 1px;'>Loss Values:</p>", unsafe_allow_html=True)
                    for data in loss_data:
                        st.markdown(f"""
                        <div style='margin-bottom: 12px; padding: 15px; border-radius: 10px; background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.05) 100%); border: 1px solid rgba(233, 69, 96, 0.1); box-shadow: 0 5px 15px rgba(0,0,0,0.2); backdrop-filter: blur(5px);'>
                            <p style='margin: 0; color: #f0f0f0; font-weight: 600; letter-spacing: 0.5px;'>Step {data['Step']}</p>
                            <p style='margin: 0; color: #a0a0a0;'>Content Loss: {data['Content Loss']}</p>
                            <p style='margin: 0; color: #a0a0a0;'>Style Loss: {data['Style Loss']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<p style='color: #e94560; font-weight: 600; letter-spacing: 1px;'>Evolution:</p>", unsafe_allow_html=True)
                    st.markdown("<div class='evolution-images'>", unsafe_allow_html=True)
                    st.image(
                        display_frames,
                        caption=captions,
                        width=150,
                        clamp=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display loss chart in an enhanced expander
                with st.expander("📊 View Loss History Chart", expanded=False):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    fig.patch.set_facecolor('#000000')
                    ax.set_facecolor('#111111')
                    
                    ax.plot(st.session_state.loss_history["step"], st.session_state.loss_history["content"], label="Content Loss", color="#e94560", linewidth=2.5)
                    ax.plot(st.session_state.loss_history["step"], st.session_state.loss_history["style"], label="Style Loss", color="#4fc1ff", linewidth=2.5)
                    
                    ax.set_xlabel("Step", color="#f0f0f0", fontweight=600)
                    ax.set_ylabel("Loss", color="#f0f0f0", fontweight=600)
                    ax.set_title("Loss History", color="#e94560", fontsize=14, fontweight=700)
                    ax.legend(facecolor='#111111', edgecolor='#f0f0f0')
                    ax.grid(True, alpha=0.2, color="#f0f0f0", linestyle="--")
                    
                    ax.tick_params(axis='x', colors='#f0f0f0')
                    ax.tick_params(axis='y', colors='#f0f0f0')
                    for spine in ax.spines.values():
                        spine.set_color('#f0f0f0')
                        
                    st.pyplot(fig)
    
    # Process button logic
    if process_button:
        # Determine device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Using device: {device} - This may take a few minutes")
        
        # Display animated loading indicator
        loading_placeholder = st.empty()
        with loading_placeholder:
            if processing_lottie:
                st_lottie(processing_lottie, height=180, key="processing_animation", speed=1.2)
        
        # Move tensors to the device
        content_tensor = content_tensor.to(device)
        style_tensor = style_tensor.to(device)
        
        # Create a container for loss chart
        loss_chart_container = st.empty()
        
        try:
            # Run style transfer with enhanced implementation
            try:
                output_tensor, intermediate_results, loss_history = run_style_transfer(
                    content_tensor, 
                    style_tensor, 
                    num_steps=num_steps,
                    style_weight=style_weight,
                    content_weight=1,
                    device=device
                )
            except RuntimeError as e:
                if "graph a second time" in str(e):
                    st.error("Encountered a computation graph error. Trying a more efficient approach...")
                    
                    # Fallback to a more efficient implementation
                    def adaptive_style_transfer(content, style, steps=60):
                        # Create input tensor
                        input_img = content.clone().detach().requires_grad_(True)
                        
                        # Use a more efficient VGG model with fewer layers
                        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
                        
                        # Use a more efficient optimizer
                        optimizer = torch.optim.Adam([input_img], lr=0.05, betas=(0.9, 0.999))
                        
                        # Adaptive learning rate scheduler
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
                        
                        # Process
                        intermediate = []
                        history = {"content": [], "style": [], "total": [], "step": []}
                        last_loss = float('inf')
                        
                        # Progress bar and status
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for s in range(steps):
                            # Update progress
                            progress = (s + 1) / steps
                            progress_bar.progress(progress)
                            status_text.text(f"Adaptive Processing: Step {s+1}/{steps}")
                            
                            optimizer.zero_grad()
                            
                            # Get features
                            content_features = model(content)
                            style_features = model(style)
                            input_features = model(input_img)
                            
                            # Content loss
                            content_loss = F.mse_loss(input_features, content_features)
                            
                            # Enhanced style loss
                            style_gram = gram_matrix(style_features)
                            input_gram = gram_matrix(input_features)
                            style_loss = F.mse_loss(input_gram, style_gram)
                            
                            # Total loss with adaptive weighting
                            loss = content_loss + style_weight/10000 * style_loss
                            
                            # Record history
                            history["content"].append(content_loss.item())
                            history["style"].append(style_loss.item())
                            history["total"].append(loss.item())
                            history["step"].append(s)
                            
                            loss.backward()
                            optimizer.step()
                            
                            # Step the scheduler
                            scheduler.step(loss)
                            
                            # Clamp values
                            with torch.no_grad():
                                input_img.clamp_(0, 1)
                                
                            # Record intermediate
                            if s % max(1, steps // 5) == 0 or s == steps - 1:
                                intermediate.append(input_img.detach().clone())
                                
                            # Early stopping if loss stabilizes
                            if abs(loss.item() - last_loss) < 1e-5 and s > steps // 2:
                                remaining_steps = steps - s - 1
                                if remaining_steps > 5:
                                    st.write(f"Loss has stabilized at step {s}. Finishing early...")
                                    
                                    # Add final result
                                    intermediate.append(input_img.detach().clone())
                                    
                                    # Fill remaining history
                                    for i in range(s+1, steps):
                                        history["content"].append(content_loss.item())
                                        history["style"].append(style_loss.item())
                                        history["total"].append(loss.item())
                                        history["step"].append(i)
                                    
                                    # Update progress to complete
                                    progress_bar.progress(1.0)
                                    status_text.text(f"Processing completed early: Stable result achieved!")
                                    break
                            
                            last_loss = loss.item()
                            
                            # Report progress
                            if s % 5 == 0:
                                st.write(f"Adaptive Step {s}, Content Loss: {content_loss.item():.6f}, Style Loss: {style_loss.item():.6f}")
                        
                        return input_img.detach(), intermediate, history
                    
                    # Run adaptive version
                    output_tensor, intermediate_results, loss_history = adaptive_style_transfer(
                        content_tensor, style_tensor, steps=max(40, num_steps//2)
                    )
                else:
                    # If it's another type of error, re-raise it
                    raise
                    
            # Remove lottie animation after processing
            loading_placeholder.empty()
            
            # Convert result tensor to image
            output_image = tensor_to_image(output_tensor)
            output_pil = Image.fromarray((output_image * 255).astype(np.uint8))
            
            # Plot loss history
            if loss_history:
                # Store in session state for persistence
                st.session_state.loss_history = loss_history
                
                with st.expander("Loss History Chart", expanded=True):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    fig.patch.set_facecolor('#000000')
                    ax.set_facecolor('#111111')
                    
                    ax.plot(loss_history["step"], loss_history["content"], label="Content Loss", color="#e94560", linewidth=2.5)
                    ax.plot(loss_history["step"], loss_history["style"], label="Style Loss", color="#4fc1ff", linewidth=2.5)
                    
                    ax.set_xlabel("Step", color="#f0f0f0", fontweight=600)
                    ax.set_ylabel("Loss", color="#f0f0f0", fontweight=600)
                    ax.set_title("Loss History", color="#e94560", fontsize=14, fontweight=700)
                    ax.legend(facecolor='#111111', edgecolor='#f0f0f0')
                    ax.grid(True, alpha=0.2, color="#f0f0f0", linestyle="--")
                    
                    ax.tick_params(axis='x', colors='#f0f0f0')
                    ax.tick_params(axis='y', colors='#f0f0f0')
                    for spine in ax.spines.values():
                        spine.set_color('#f0f0f0')
                    
                    loss_chart_container.pyplot(fig)
                    
            # Store output image in session state
            st.session_state.output_image = output_pil
            st.session_state.processed = True
            
            # Store intermediate results
            pil_intermediates = []
            for result in intermediate_results:
                img = tensor_to_image(result)
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                pil_intermediates.append(img_pil)
                
            st.session_state.intermediate_results = pil_intermediates
            
            # Display the result with enhanced styling
            with result_container:
                st.markdown("<h3 style='text-align: center;'>Stylized Result</h3>", unsafe_allow_html=True)
                
                # Show transformation animation
                if transformation_lottie:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st_lottie(transformation_lottie, height=120, key="complete_animation", speed=1.0)
                
                # Display image with enhanced styling
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("<p class='result-title'>Stylized Masterpiece</p>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 3, 1])
                with col2:
                    st.image(output_pil, use_container_width=True, clamp=True)
                
                # Add a download button for the result
                buf = io.BytesIO()
                output_pil.save(buf, format="PNG")
                
                # Create columns for better button placement
                btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
                with btn_col2:
                    download_button = st.download_button(
                        label="💾 Download Masterpiece",
                        data=buf.getvalue(),
                        file_name="stylized_image.png",
                        mime="image/png",
                        key="download_button"
                    )
                
                st.markdown("</div></div>", unsafe_allow_html=True)
                
                # Show the progression steps and loss values in an enhanced container
                if len(intermediate_results) > 1:
                    st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: center;'>Style Transfer Evolution</h3>", unsafe_allow_html=True)
                    
                    # Create an enhanced container for the evolution display
                    st.markdown("<div class='evolution-container'>", unsafe_allow_html=True)
                    
                    # Display sample loss values in an enhanced format
                    loss_steps = [0, num_steps//4, num_steps//2, 3*num_steps//4, num_steps-1]
                    loss_data = []
                    
                    for step in loss_steps:
                        if step < len(loss_history["step"]):
                            idx = loss_history["step"].index(step)
                            loss_data.append({
                                "Step": step,
                                "Content Loss": f"{loss_history['content'][idx]:.6f}",
                                "Style Loss": f"{loss_history['style'][idx]:.6f}"
                            })
                    
                    # Explicitly set the frame size for the evolution frames
                    st.markdown("""
                    <style>
                    .evolution-images img {
                        max-width: 150px !important;
                        max-height: 150px !important;
                        object-fit: contain;
                        margin: 8px !important;
                        border-radius: 12px !important;
                        box-shadow: 0 8px 20px rgba(0,0,0,0.3) !important;
                        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
                        border: 1px solid rgba(233, 69, 96, 0.2) !important;
                    }
                    
                    .evolution-images img:hover {
                        transform: scale(1.15) translateY(-5px) !important;
                        box-shadow: 0 15px 30px rgba(0,0,0,0.4), 0 0 15px rgba(233, 69, 96, 0.3) !important;
                        z-index: 10 !important;
                        border: 1px solid rgba(233, 69, 96, 0.4) !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Convert intermediate results to images
                    evolution_frames = []
                    captions = []
                    
                    # Only include a subset of frames to keep it manageable
                    display_indices = list(range(0, len(intermediate_results), max(1, len(intermediate_results)//6)))
                    if len(intermediate_results)-1 not in display_indices:
                        display_indices.append(len(intermediate_results)-1)
                    
                    for i in display_indices:
                        img = tensor_to_image(intermediate_results[i])
                        img_pil = Image.fromarray((img * 255).astype(np.uint8))
                        evolution_frames.append(img_pil)
                        step_num = i * (num_steps // len(intermediate_results))
                        captions.append(f"Step {step_num}")
                    
                    # Display with enhanced styling in a sleeker container
                    col1, col2 = st.columns([3, 7])
                    
                    with col1:
                        st.markdown("<p style='color: #e94560; font-weight: 600; letter-spacing: 1px;'>Loss Values:</p>", unsafe_allow_html=True)
                        for data in loss_data:
                            st.markdown(f"""
                            <div style='margin-bottom: 12px; padding: 15px; border-radius: 10px; background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.05) 100%); border: 1px solid rgba(233, 69, 96, 0.1); box-shadow: 0 5px 15px rgba(0,0,0,0.2); backdrop-filter: blur(5px);'>
                                <p style='margin: 0; color: #f0f0f0; font-weight: 600; letter-spacing: 0.5px;'>Step {data['Step']}</p>
                                <p style='margin: 0; color: #a0a0a0;'>Content Loss: {data['Content Loss']}</p>
                                <p style='margin: 0; color: #a0a0a0;'>Style Loss: {data['Style Loss']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<p style='color: #e94560; font-weight: 600; letter-spacing: 1px;'>Evolution:</p>", unsafe_allow_html=True)
                        st.markdown("<div class='evolution-images'>", unsafe_allow_html=True)
                        st.image(
                            evolution_frames,
                            caption=captions,
                            width=150,
                            clamp=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display loss chart in an enhanced expander
                    with st.expander("📊 View Loss History Chart", expanded=False):
                        fig, ax = plt.subplots(figsize=(10, 4))
                        fig.patch.set_facecolor('#000000')
                        ax.set_facecolor('#111111')
                        
                        ax.plot(loss_history["step"], loss_history["content"], label="Content Loss", color="#e94560", linewidth=2.5)
                        ax.plot(loss_history["step"], loss_history["style"], label="Style Loss", color="#4fc1ff", linewidth=2.5)
                        
                        ax.set_xlabel("Step", color="#f0f0f0", fontweight=600)
                        ax.set_ylabel("Loss", color="#f0f0f0", fontweight=600)
                        ax.set_title("Loss History", color="#e94560", fontsize=14, fontweight=700)
                        ax.legend(facecolor='#111111', edgecolor='#f0f0f0')
                        ax.grid(True, alpha=0.2, color="#f0f0f0", linestyle="--")
                        
                        ax.tick_params(axis='x', colors='#f0f0f0')
                        ax.tick_params(axis='y', colors='#f0f0f0')
                        for spine in ax.spines.values():
                            spine.set_color('#f0f0f0')
                            
                        st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
else:
    # Display instructions with enhanced empty state
    st.markdown("""
    <div style='background: linear-gradient(135deg, rgba(15, 52, 96, 0.2) 0%, rgba(23, 37, 84, 0.2) 100%); padding: 3rem; border-radius: 16px; backdrop-filter: blur(10px); max-width: 900px; margin: 2rem auto; border: 1px solid rgba(233, 69, 96, 0.1); box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);'>
        <div class='lottie-container' style='display: flex; justify-content: center; margin-bottom: 2rem;'>
        </div>
        <p style='color: #f0f0f0; text-align: center; font-size: 1.5rem; font-weight: 600; letter-spacing: 1px; margin-bottom: 1.5rem;'>
            👆 Upload both a content image and a style image to begin your artistic journey
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show a different Lottie animation as placeholder
    if processing_lottie:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(processing_lottie, height=300, key="placeholder_animation", speed=1.2)

# Add a fixed footer at the bottom with the original "powered by" text
st.markdown("""
<div class='footer'>
    <p style="color: #f0f0f0; font-size: 0.9rem; margin: 0;">
        ✨ Powered by PyTorch and Neural Style Transfer | Created with Streamlit ✨
    </p>
    <p style="color: #a0a0a0; font-size: 0.7rem; margin: 0;">
        Transform your photos into artistic masterpieces with the power of AI
    </p>
</div>
""", unsafe_allow_html=True)