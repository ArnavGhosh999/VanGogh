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

# Load different lottie animations
artist_lottie = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_syqnfe7c.json")  # Artist palette animation
paintbrush_lottie = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_TmewUB.json")  # Paintbrush animation
processing_lottie = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_jAkGFa.json")  # Processing animation
magic_lottie = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_touohxv0.json")  # Magic wand animation

# Add custom CSS for styling with animations - using black background
st.markdown("""
<style>
    /* Background styling - black only */
    .stApp {
        background-color: #000000;
    }
    
    /* Title animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .main-title {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 1rem;
        color: #e94560;
        animation: fadeIn 1.5s ease-out;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .sub-title {
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 2rem;
        color: #f0f0f0;
        animation: fadeIn 2s ease-out;
    }
    
    /* Image containers */
    .images-row {
        display: flex;
        justify-content: space-around;
        margin-bottom: 2rem;
        animation: slideIn 1s ease-out;
    }
    
    .result-container {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
        animation: slideIn 1.5s ease-out;
    }
    
    /* Redesigned image cards - smaller and with transparent background */
    .image-card {
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        padding: 0.5rem;
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(5px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        max-height: 350px;
        overflow: hidden;
        width: 85%;
        margin: 0 auto;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .image-card img {
        max-height: 300px;
        object-fit: contain;
        margin: 0 auto;
        display: block;
        border-radius: 8px;
    }
    
    .image-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4);
    }
    
    .image-title {
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        color: #f0f0f0;
        font-weight: 600;
    }
    
    /* Result card - transparent instead of white */
    .result-card {
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        padding: 0.5rem;
        background-color: rgba(233, 69, 96, 0.1);
        backdrop-filter: blur(5px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        max-height: 450px;
        overflow: hidden;
        width: 75%;
        margin: 0 auto;
        border: 1px solid rgba(233, 69, 96, 0.3);
        animation: fadeIn 1s ease-out;
    }
    
    .result-card img {
        max-height: 380px;
        object-fit: contain;
        margin: 0 auto;
        display: block;
        border-radius: 8px;
    }
    
    .result-title {
        text-align: center;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
        color: #e94560;
        font-weight: 600;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #e94560;
        transition: width 0.3s ease;
    }
    
    /* Processing animation */
    .processing-animation {
        width: 100%;
        height: 4px;
        background: linear-gradient(to right, #e94560, #0f3460, #e94560);
        background-size: 200% 100%;
        animation: loading 2s infinite;
        margin-bottom: 1rem;
        border-radius: 2px;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    /* Process button */
    .btn-process {
        background-color: #e94560;
        color: white;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 5px;
        font-size: 1.1rem;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease;
        display: block;
        margin: 1rem auto;
        width: 80%;
        max-width: 300px;
    }
    
    .btn-process:hover {
        background-color: #c81e42;
        transform: scale(1.03);
    }
    
    /* General containers */
    .upload-section {
        animation: slideIn 1s ease-out;
    }
    
    .lottie-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    /* Evolution display */
    .evolution-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 15px;
        margin-top: 20px;
        background-color: rgba(15, 52, 96, 0.2);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(5px);
    }
    
    .evolution-image {
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 8px;
        padding: 5px;
        max-width: 150px;
        transition: transform 0.2s;
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    .evolution-image:hover {
        transform: scale(1.1);
    }
    
    /* Fixed sizing for images */
    .stImage img {
        max-height: 400px;
        object-fit: contain;
    }
    
    /* Loss history styling */
    .loss-history {
        margin-top: 30px;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        background-color: rgba(15, 52, 96, 0.2);
        backdrop-filter: blur(5px);
    }
    
    /* Download button styling */
    .stDownloadButton button {
        background-color: #e94560 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        margin: 1rem auto !important;
        display: block !important;
        width: 100% !important;
        max-width: 250px !important;
        transition: all 0.3s !important;
        font-weight: 600 !important;
    }
    
    .stDownloadButton button:hover {
        background-color: #c81e42 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(233, 69, 96, 0.4) !important;
    }
    
    /* Section headers */
    h3 {
        color: #f0f0f0 !important;
        text-align: center !important;
        margin-top: 2rem !important;
        margin-bottom: 1.5rem !important;
        font-weight: 600 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(233, 69, 96, 0.1) !important;
        border-radius: 8px !important;
        color: #f0f0f0 !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(15, 52, 96, 0.2) !important;
        border-radius: 0 0 8px 8px !important;
        padding: 10px !important;
    }
    
    /* Sidebar styling - dark */
    .css-6qob1r, .css-1544g2n {
        background-color: #111111 !important;
    }
    
    .css-6qob1r label, .css-6qob1r p, .css-1544g2n label, .css-1544g2n p {
        color: #f0f0f0 !important;
    }
    
    /* Tooltip styling */
    .stTooltip {
        background-color: rgba(26, 26, 46, 0.9) !important;
        border: 1px solid rgba(233, 69, 96, 0.5) !important;
        color: #f0f0f0 !important;
    }
    
    /* Footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        background-color: rgba(0, 0, 0, 0.7);
        z-index: 1000;
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
    """Neural style transfer implementation with simpler optimization to avoid graph errors"""
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
    optimizer = torch.optim.Adam([input_img], lr=0.03)
    
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
        if step % 5 == 0:
            st.write(f"Step {step}, Content Loss: {content_loss.item():.6f}, Style Loss: {style_loss.item():.6f}")
    
    # Clear progress display
    progress_bar.empty()
    status_text.text("Style transfer completed!")
    
    return input_img.detach(), intermediate_results, loss_history

# Page title and description with animation
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 class='main-title'>Van Gogh Style Transfer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Transform photos into artistic masterpieces</p>", unsafe_allow_html=True)
with col3:
    if artist_lottie:
        st_lottie(artist_lottie, height=150, key="artist_animation", speed=1.2)

# Create main upload section in the center
st.markdown("<div style='max-width: 800px; margin: 0 auto;'>", unsafe_allow_html=True)

# Upload sections side by side
col1, col2 = st.columns([1, 1])
with col1:
    if paintbrush_lottie:
        st_lottie(paintbrush_lottie, height=80, key="paintbrush_animation")
    st.markdown("<h4 style='text-align: center; color: #f0f0f0;'>Content Image</h4>", unsafe_allow_html=True)
    content_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="content_upload")

with col2:
    if magic_lottie:  # Using a different animation here
        st_lottie(magic_lottie, height=80, key="magic_animation")
    st.markdown("<h4 style='text-align: center; color: #f0f0f0;'>Style Image</h4>", unsafe_allow_html=True)
    style_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="style_upload")

st.markdown("</div>", unsafe_allow_html=True)

# Create a sidebar for parameters
with st.sidebar:
    st.markdown("<h3 style='color: #e94560; text-align: center;'>Style Transfer Controls</h3>", unsafe_allow_html=True)
    
    # Add some space and a divider
    st.markdown("<div style='margin: 1.5em 0; border-bottom: 1px solid rgba(255,255,255,0.2);'></div>", unsafe_allow_html=True)
    
    # Parameter sliders with better styling
    st.markdown("<p style='color: #f0f0f0; font-weight: 600;'>Processing Steps</p>", unsafe_allow_html=True)
    num_steps = st.slider("", min_value=20, max_value=100, value=40, step=5, 
                         help="More steps give better results but take longer to process")
    
    st.markdown("<p style='color: #f0f0f0; font-weight: 600; margin-top: 1.5em;'>Style Strength</p>", unsafe_allow_html=True)
    style_weight = st.slider("", min_value=10000, max_value=1000000, value=100000, step=10000, format="%e",
                            help="Higher values emphasize the style more")
    
    # Add some space
    st.markdown("<div style='margin: 2em 0;'></div>", unsafe_allow_html=True)
    
    # Create a styled button using HTML
    process_button = st.button("✨ Generate Masterpiece ✨", key="process_button", 
                              use_container_width=True,
                              help="Run the style transfer algorithm")

# Main content area for displaying images
if content_file is not None and style_file is not None:
    # Load the images
    content_tensor, content_img = load_image(content_file)
    style_tensor, style_img = load_image(style_file)
    
    # Display input images side by side with smaller cards
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
        # Display the result without white box
        with result_container:
            st.markdown("<h3 style='text-align: center;'>Stylized Result</h3>", unsafe_allow_html=True)
            
            # Display image with fixed dimensions
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
            
            # Show the progression steps and loss values in a more attractive container
            if len(st.session_state.intermediate_results) > 1:
                st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center;'>Style Transfer Evolution</h3>", unsafe_allow_html=True)
                
                # Create an attractive container for the evolution display
                st.markdown("<div class='evolution-container'>", unsafe_allow_html=True)
                
                # Display sample loss values in a cleaner format
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
                    max-width: 120px !important;
                    max-height: 120px !important;
                    object-fit: contain;
                    margin: 5px !important;
                    border-radius: 10px !important;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
                    transition: transform 0.3s ease !important;
                }
                
                .evolution-images img:hover {
                    transform: scale(1.1) !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Only include a subset of frames to keep it manageable
                display_indices = list(range(0, len(st.session_state.intermediate_results), max(1, len(st.session_state.intermediate_results)//6)))
                if len(st.session_state.intermediate_results)-1 not in display_indices:
                    display_indices.append(len(st.session_state.intermediate_results)-1)
                
                display_frames = [st.session_state.intermediate_results[i] for i in display_indices]
                captions = [f"Step {i * (len(st.session_state.loss_history['step']) // len(st.session_state.intermediate_results))}" for i in display_indices]
                
                # Display with fixed dimensions in a smaller container
                col1, col2 = st.columns([3, 7])
                
                with col1:
                    st.markdown("<p style='color: #e94560; font-weight: 600;'>Loss Values:</p>", unsafe_allow_html=True)
                    for data in loss_data:
                        st.markdown(f"""
                        <div style='margin-bottom: 10px; padding: 10px; border-radius: 8px; background-color: rgba(255,255,255,0.1);'>
                            <p style='margin: 0; color: #f0f0f0; font-weight: 600;'>Step {data['Step']}</p>
                            <p style='margin: 0; color: #a0a0a0;'>Content Loss: {data['Content Loss']}</p>
                            <p style='margin: 0; color: #a0a0a0;'>Style Loss: {data['Style Loss']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<p style='color: #e94560; font-weight: 600;'>Evolution:</p>", unsafe_allow_html=True)
                    st.markdown("<div class='evolution-images'>", unsafe_allow_html=True)
                    st.image(
                        display_frames,
                        caption=captions,
                        width=120,  # Fixed small width
                        clamp=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display loss chart in an expander
                with st.expander("📊 View Loss History Chart", expanded=False):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    fig.patch.set_facecolor('#000000')
                    ax.set_facecolor('#111111')
                    
                    ax.plot(st.session_state.loss_history["step"], st.session_state.loss_history["content"], label="Content Loss", color="#e94560", linewidth=2.5)
                    ax.plot(st.session_state.loss_history["step"], st.session_state.loss_history["style"], label="Style Loss", color="#3fc1c9", linewidth=2.5)
                    
                    ax.set_xlabel("Step", color="#f0f0f0")
                    ax.set_ylabel("Loss", color="#f0f0f0")
                    ax.set_title("Loss History", color="#e94560", fontsize=14)
                    ax.legend(facecolor='#111111', edgecolor='#f0f0f0')
                    ax.grid(True, alpha=0.2, color="#f0f0f0")
                    
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
        
        # Display animated loading indicator with processing_lottie
        loading_placeholder = st.empty()
        with loading_placeholder:
            if processing_lottie:
                st_lottie(processing_lottie, height=150, key="processing_animation")
        
        # Move tensors to the device
        content_tensor = content_tensor.to(device)
        style_tensor = style_tensor.to(device)
        
        # Create a container for loss chart
        loss_chart_container = st.empty()
        
        try:
            # Run style transfer with simpler implementation
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
                    st.error("Encountered a computation graph error. Trying a simpler approach...")
                    
                    # Fallback to a simpler implementation
                    def simple_style_transfer(content, style, steps=50):
                        # Create input tensor
                        input_img = content.clone().detach().requires_grad_(True)
                        
                        # Use a simpler VGG model with fewer layers
                        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
                        
                        # Optimizer
                        optimizer = torch.optim.Adam([input_img], lr=0.01)
                        
                        # Process
                        intermediate = []
                        history = {"content": [], "style": [], "total": [], "step": []}
                        
                        for s in range(steps):
                            optimizer.zero_grad()
                            
                            # Get features
                            content_features = model(content)
                            style_features = model(style)
                            input_features = model(input_img)
                            
                            # Simple content loss
                            content_loss = F.mse_loss(input_features, content_features)
                            
                            # Simple style loss (using mean and std)
                            mean_loss = F.mse_loss(
                                torch.mean(input_features, dim=[2, 3]), 
                                torch.mean(style_features, dim=[2, 3])
                            )
                            std_loss = F.mse_loss(
                                torch.std(input_features, dim=[2, 3]), 
                                torch.std(style_features, dim=[2, 3])
                            )
                            
                            style_loss = mean_loss + std_loss
                            loss = content_loss + style_weight/10000 * style_loss
                            
                            # Record history
                            history["content"].append(content_loss.item())
                            history["style"].append(style_loss.item())
                            history["total"].append(loss.item())
                            history["step"].append(s)
                            
                            loss.backward()
                            optimizer.step()
                            
                            # Clamp values
                            with torch.no_grad():
                                input_img.clamp_(0, 1)
                                
                            # Record intermediate
                            if s % max(1, steps // 5) == 0 or s == steps - 1:
                                intermediate.append(input_img.detach().clone())
                                
                            # Report progress
                            progress = (s + 1) / steps
                            progress_bar.progress(progress)
                            status_text.text(f"Processing: Step {s+1}/{steps}")
                            
                            if s % 5 == 0:
                                st.write(f"Simplified Step {s}, Content Loss: {content_loss.item():.6f}, Style Loss: {style_loss.item():.6f}")
                        
                        return input_img.detach(), intermediate, history
                    
                    # Run simplified version
                    output_tensor, intermediate_results, loss_history = simple_style_transfer(
                        content_tensor, style_tensor, steps=max(30, num_steps//2)
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
                    ax.plot(loss_history["step"], loss_history["style"], label="Style Loss", color="#3fc1c9", linewidth=2.5)
                    
                    ax.set_xlabel("Step", color="#f0f0f0")
                    ax.set_ylabel("Loss", color="#f0f0f0")
                    ax.set_title("Loss History", color="#e94560", fontsize=14)
                    ax.legend(facecolor='#111111', edgecolor='#f0f0f0')
                    ax.grid(True, alpha=0.2, color="#f0f0f0")
                    
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
            
            # Display the result without white box, using custom styling
            with result_container:
                st.markdown("<h3 style='text-align: center;'>Stylized Result</h3>", unsafe_allow_html=True)
                
                # Display image with fixed dimensions
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
                
                # Show the progression steps and loss values in a more attractive container
                if len(intermediate_results) > 1:
                    st.markdown("<div style='margin-top: 2.5rem;'></div>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: center;'>Style Transfer Evolution</h3>", unsafe_allow_html=True)
                    
                    # Create an attractive container for the evolution display
                    st.markdown("<div class='evolution-container'>", unsafe_allow_html=True)
                    
                    # Display sample loss values in a cleaner format
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
                        max-width: 120px !important;
                        max-height: 120px !important;
                        object-fit: contain;
                        margin: 5px !important;
                        border-radius: 10px !important;
                        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
                        transition: transform 0.3s ease !important;
                    }
                    
                    .evolution-images img:hover {
                        transform: scale(1.1) !important;
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
                    
                    # Display with fixed dimensions in a smaller container
                    col1, col2 = st.columns([3, 7])
                    
                    with col1:
                        st.markdown("<p style='color: #e94560; font-weight: 600;'>Loss Values:</p>", unsafe_allow_html=True)
                        for data in loss_data:
                            st.markdown(f"""
                            <div style='margin-bottom: 10px; padding: 10px; border-radius: 8px; background-color: rgba(255,255,255,0.1);'>
                                <p style='margin: 0; color: #f0f0f0; font-weight: 600;'>Step {data['Step']}</p>
                                <p style='margin: 0; color: #a0a0a0;'>Content Loss: {data['Content Loss']}</p>
                                <p style='margin: 0; color: #a0a0a0;'>Style Loss: {data['Style Loss']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("<p style='color: #e94560; font-weight: 600;'>Evolution:</p>", unsafe_allow_html=True)
                        st.markdown("<div class='evolution-images'>", unsafe_allow_html=True)
                        st.image(
                            evolution_frames,
                            caption=captions,
                            width=120,  # Fixed small width
                            clamp=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Display loss chart in an expander
                    with st.expander("📊 View Loss History Chart", expanded=False):
                        fig, ax = plt.subplots(figsize=(10, 4))
                        fig.patch.set_facecolor('#000000')
                        ax.set_facecolor('#111111')
                        
                        ax.plot(loss_history["step"], loss_history["content"], label="Content Loss", color="#e94560", linewidth=2.5)
                        ax.plot(loss_history["step"], loss_history["style"], label="Style Loss", color="#3fc1c9", linewidth=2.5)
                        
                        ax.set_xlabel("Step", color="#f0f0f0")
                        ax.set_ylabel("Loss", color="#f0f0f0")
                        ax.set_title("Loss History", color="#e94560", fontsize=14)
                        ax.legend(facecolor='#111111', edgecolor='#f0f0f0')
                        ax.grid(True, alpha=0.2, color="#f0f0f0")
                        
                        ax.tick_params(axis='x', colors='#f0f0f0')
                        ax.tick_params(axis='y', colors='#f0f0f0')
                        for spine in ax.spines.values():
                            spine.set_color('#f0f0f0')
                            
                        st.pyplot(fig)
            
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
else:
    # Display instructions when no images are uploaded
    st.markdown("""
    <div style='background-color: rgba(15, 52, 96, 0.3); padding: 2rem; border-radius: 15px; backdrop-filter: blur(5px); max-width: 800px; margin: 2rem auto;'>
        <p style='color: #f0f0f0; text-align: center; font-size: 1.2rem;'>
            👆 Please upload both a content image and a style image to begin
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show a different Lottie animation as placeholder
    if processing_lottie:  # Using different animation here
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(processing_lottie, height=300, key="placeholder_animation")
    
    # Placeholder with more attractive styling
    st.markdown("""
    <div style='background-color: rgba(233, 69, 96, 0.1); padding: 2rem; border-radius: 15px; backdrop-filter: blur(5px); max-width: 800px; margin: 2rem auto; border: 1px solid rgba(233, 69, 96, 0.2);'>
        <h3 style='text-align: center; color: #e94560; margin-bottom: 1.5rem;'>How It Works</h3>
        
        <div style='display: flex; margin-bottom: 1rem;'>
            <div style='background-color: #e94560; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 15px; flex-shrink: 0;'>1</div>
            <div>
                <p style='color: #f0f0f0; margin: 0;'><strong>Upload your photo</strong> - This is the content image you want to transform</p>
            </div>
        </div>
        
        <div style='display: flex; margin-bottom: 1rem;'>
            <div style='background-color: #e94560; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 15px; flex-shrink: 0;'>2</div>
            <div>
                <p style='color: #f0f0f0; margin: 0;'><strong>Upload a style reference</strong> - This is the artistic style you want to apply (e.g., a Van Gogh painting)</p>
            </div>
        </div>
        
        <div style='display: flex; margin-bottom: 1rem;'>
            <div style='background-color: #e94560; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 15px; flex-shrink: 0;'>3</div>
            <div>
                <p style='color: #f0f0f0; margin: 0;'><strong>Adjust parameters</strong> - Control how strongly the style is applied</p>
            </div>
        </div>
        
        <div style='display: flex; margin-bottom: 1rem;'>
            <div style='background-color: #e94560; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-right: 15px; flex-shrink: 0;'>4</div>
            <div>
                <p style='color: #f0f0f0; margin: 0;'><strong>Generate your masterpiece</strong> - Our AI will blend your photo with the artistic style</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Add a fixed footer at the bottom
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