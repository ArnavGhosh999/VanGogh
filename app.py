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
    page_title="Animated Neural Style Transfer App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load a lottie animation
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load lottie animations
artist_lottie = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_9iugmpgx.json")
paintbrush_lottie = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_rovf9gzu.json")
processing_lottie = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_AQEOul.json")

# Add custom CSS for styling with animations
st.markdown("""
<style>
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
        color: #1E88E5;
        animation: fadeIn 1.5s ease-out;
    }
    
    .sub-title {
        text-align: center;
        font-size: 1.5rem;
        margin-bottom: 2rem;
        color: #424242;
        animation: fadeIn 2s ease-out;
    }
    
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
    
    .image-card {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 1rem;
        background-color: white;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        max-height: 450px;
        overflow: hidden;
    }
    
    .image-card img {
        max-height: 350px;
        object-fit: contain;
        margin: 0 auto;
        display: block;
    }
    
    .image-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .image-title {
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        color: #212121;
    }
    
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
        transition: width 0.3s ease;
    }
    
    .processing-animation {
        width: 100%;
        height: 4px;
        background: linear-gradient(to right, #1E88E5, #64B5F6, #1E88E5);
        background-size: 200% 100%;
        animation: loading 2s infinite;
        margin-bottom: 1rem;
        border-radius: 2px;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    
    .btn-process {
        background-color: #1E88E5;
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
        background-color: #1565C0;
        transform: scale(1.03);
    }
    
    .upload-section {
        animation: slideIn 1s ease-out;
    }
    
    .lottie-container {
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    .result-image {
        max-height: 400px;
        width: auto;
        margin: 0 auto;
        display: block;
    }
    
    .evolution-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 20px;
    }
    
    .evolution-image {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
        max-width: 150px;
        transition: transform 0.2s;
    }
    
    .evolution-image:hover {
        transform: scale(1.05);
    }
    
    /* Fixed sizing for images to prevent overflow */
    .stImage img {
        max-height: 400px;
        object-fit: contain;
    }
    
    /* Keep loss history visible */
    .loss-history {
        margin-top: 30px;
        margin-bottom: 30px;
        border: 1px solid #eee;
        border-radius: 10px;
        padding: 15px;
        background-color: #f9f9f9;
    }
    
    /* Better styling for download button */
    .stDownloadButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        margin: 1rem auto;
        display: block;
        width: 100%;
        max-width: 250px;
        transition: background-color 0.3s;
    }
    
    .stDownloadButton button:hover {
        background-color: #1565C0;
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

# Page title and description with animation - with Lottie animation beside title
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 class='main-title'>Van Gogh Style Transfer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Transform your photos into artistic masterpieces with neural style transfer</p>", unsafe_allow_html=True)
with col3:
    if artist_lottie:
        st_lottie(artist_lottie, height=150, key="artist_animation")

# Create main upload section in the center
st.header("Upload Images")

# First upload section - Content Image
col1, col2 = st.columns([1, 1])
with col1:
    if paintbrush_lottie:
        st_lottie(paintbrush_lottie, height=100, key="paintbrush_animation")
    st.subheader("Content Image")
    content_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"], key="content_upload")

# Second upload section - Style Image
with col2:
    if paintbrush_lottie:
        st_lottie(paintbrush_lottie, height=100, key="paintbrush_animation2")
    st.subheader("Style Image")
    style_file = st.file_uploader("Upload style reference", type=["jpg", "jpeg", "png"], key="style_upload")

# Create a sidebar for parameters
with st.sidebar:
    st.header("Style Transfer Parameters")
    num_steps = st.slider("Processing Steps", min_value=20, max_value=200, value=50, step=10, 
                         help="More steps give better results but take longer to process")
    style_weight = st.slider("Style Strength", min_value=10000, max_value=1000000, value=100000, step=10000, format="%e",
                            help="Higher values emphasize the style more")
    
    # Create a styled button using HTML
    process_button_placeholder = st.empty()
    process_button = process_button_placeholder.button("Generate Masterpiece", key="process_button", 
                                                      use_container_width=True,
                                                      help="Run the style transfer algorithm")

# Main content area for displaying images
if content_file is not None and style_file is not None:
    # Load the images
    content_tensor, content_img = load_image(content_file)
    style_tensor, style_img = load_image(style_file)
    
    # Display input images side by side
    st.markdown("<h3 style='text-align: center;'>Input Images</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
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
        # Display the existing result
        with result_container:
            st.markdown("<h3 style='text-align: center;'>Stylized Result</h3>", unsafe_allow_html=True)
            
            # Display image with fixed dimensions
            st.markdown("<div class='result-container'>", unsafe_allow_html=True)
            st.markdown("<div class='image-card' style='max-width: 600px; margin: 0 auto;'>", unsafe_allow_html=True)
            st.markdown("<p class='image-title'>Stylized Masterpiece</p>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                st.image(st.session_state.output_image, use_container_width=True, clamp=True)
            
            # Add a download button for the result
            buf = io.BytesIO()
            st.session_state.output_image.save(buf, format="PNG")
            
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
            with btn_col2:
                download_button = st.download_button(
                    label="Download Masterpiece",
                    data=buf.getvalue(),
                    file_name="stylized_image.png",
                    mime="image/png",
                    key="download_button"
                )
            
            st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Show loss history if available
            if st.session_state.loss_history:
                with st.expander("Loss History Chart", expanded=True):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(st.session_state.loss_history["step"], st.session_state.loss_history["content"], label="Content Loss", color="blue")
                    ax.plot(st.session_state.loss_history["step"], st.session_state.loss_history["style"], label="Style Loss", color="red")
                    ax.set_xlabel("Step")
                    ax.set_ylabel("Loss")
                    ax.set_title("Loss History")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
            
            # Show the progression steps
            if st.session_state.intermediate_results:
                st.subheader("Style Transfer Evolution")
                
                # Display sample loss values if available
                if st.session_state.loss_history:
                    loss_steps = [0, len(st.session_state.loss_history["step"])//4, len(st.session_state.loss_history["step"])//2, 
                                  3*len(st.session_state.loss_history["step"])//4, len(st.session_state.loss_history["step"])-1]
                    for step in loss_steps:
                        if step < len(st.session_state.loss_history["step"]):
                            st.write(f"Step {st.session_state.loss_history['step'][step]}, Content Loss: {st.session_state.loss_history['content'][step]:.6f}, Style Loss: {st.session_state.loss_history['style'][step]:.6f}")
                
                # Explicitly set the frame size for the evolution frames
                st.markdown("""
                <style>
                .evolution-container img {
                    max-width: 150px !important;
                    max-height: 150px !important;
                    object-fit: contain;
                    margin: 5px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown("<div class='evolution-container'>", unsafe_allow_html=True)
                
                # Only include a subset of frames to keep it manageable
                display_indices = list(range(0, len(st.session_state.intermediate_results), max(1, len(st.session_state.intermediate_results)//6)))
                if len(st.session_state.intermediate_results)-1 not in display_indices:
                    display_indices.append(len(st.session_state.intermediate_results)-1)
                
                display_frames = [st.session_state.intermediate_results[i] for i in display_indices]
                captions = [f"Step {i * (len(st.session_state.loss_history['step']) // len(st.session_state.intermediate_results))}" for i in display_indices]
                
                # Display with fixed dimensions
                col1, col2, col3 = st.columns([1, 10, 1])
                with col2:
                    st.image(
                        display_frames,
                        caption=captions,
                        width=120,  # Fixed small width
                        clamp=True
                    )
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Process button logic
    if process_button:
        # Determine device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Using device: {device} - This may take a few minutes")
        
        # Display animated loading indicator
        if processing_lottie:
            lottie_placeholder = st.empty()
            with lottie_placeholder:
                st_lottie(processing_lottie, height=150, key="processing_animation")
        
        # Move tensors to the device
        content_tensor = content_tensor.to(device)
        style_tensor = style_tensor.to(device)
        
        # Create a container for loss chart
        loss_chart_container = st.empty()
        
        # Create a placeholder for the result
        result_container = st.container()
        
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
            lottie_placeholder.empty()
            
            # Convert result tensor to image
            output_image = tensor_to_image(output_tensor)
            output_pil = Image.fromarray((output_image * 255).astype(np.uint8))
            
            # Plot loss history
            if loss_history:
                # Store in session state for persistence
                st.session_state.loss_history = loss_history
                
                with st.expander("Loss History Chart", expanded=True):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(loss_history["step"], loss_history["content"], label="Content Loss", color="blue")
                    ax.plot(loss_history["step"], loss_history["style"], label="Style Loss", color="red")
                    ax.set_xlabel("Step")
                    ax.set_ylabel("Loss")
                    ax.set_title("Loss History")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
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
            
            # Display the result in the center with controlled size
            with result_container:
                st.markdown("<h3 style='text-align: center;'>Stylized Result</h3>", unsafe_allow_html=True)
                
                # Create a persistent variable to store the result image in session state
                if 'output_image' not in st.session_state:
                    st.session_state.output_image = output_pil
                
                # Display image with fixed dimensions to prevent oversized display
                st.markdown("<div class='result-container'>", unsafe_allow_html=True)
                st.markdown("<div class='image-card' style='max-width: 600px; margin: 0 auto;'>", unsafe_allow_html=True)
                st.markdown("<p class='image-title'>Stylized Masterpiece</p>", unsafe_allow_html=True)
                
                # Display with fixed height
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
                        label="Download Masterpiece",
                        data=buf.getvalue(),
                        file_name="stylized_image.png",
                        mime="image/png",
                        key="download_button"
                    )
                
                st.markdown("</div></div>", unsafe_allow_html=True)
                
                # Show the progression steps and loss values
                if len(intermediate_results) > 1:
                    st.subheader("Style Transfer Evolution")
                    
                    # Display sample loss values
                    loss_steps = [0, num_steps//4, num_steps//2, 3*num_steps//4, num_steps-1]
                    for step in loss_steps:
                        if step < len(loss_history["step"]):
                            idx = loss_history["step"].index(step)
                            st.write(f"Step {step}, Content Loss: {loss_history['content'][idx]:.6f}, Style Loss: {loss_history['style'][idx]:.6f}")
                    
                    # Explicitly set the frame size for the evolution frames
                    st.markdown("""
                    <style>
                    .evolution-container img {
                        max-width: 150px !important;
                        max-height: 150px !important;
                        object-fit: contain;
                        margin: 5px;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<div class='evolution-container'>", unsafe_allow_html=True)
                    
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
                    
                    # Display with fixed dimensions
                    col1, col2, col3 = st.columns([1, 10, 1])
                    with col2:
                        st.image(
                            evolution_frames,
                            caption=captions,
                            width=120,  # Fixed small width
                            clamp=True
                        )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
else:
    # Display instructions when no images are uploaded
    st.info("👆 Please upload both a content image and a style image to begin")
    
    # Show a Lottie animation as placeholder
    if artist_lottie:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st_lottie(artist_lottie, height=300, key="placeholder_animation")
    
    # Placeholder for sample images with information
    st.markdown("""
    ### How It Works
    
    1. **Upload your photo** - This is the content image you want to transform
    2. **Upload a style reference** - This is the artistic style you want to apply (e.g., a Van Gogh painting)
    3. **Adjust parameters** - Control how strongly the style is applied
    4. **Generate your masterpiece** - Our AI will blend your photo with the artistic style
    
    The neural style transfer algorithm extracts the content from your photo and the style patterns from the reference image, then combines them into a new artistic creation.
    """)

# Add a footer
st.markdown("""
---
<p style="text-align: center; color: #666; font-size: 0.8rem;">
    Powered by PyTorch and Neural Style Transfer | Created with Streamlit
</p>
""", unsafe_allow_html=True)