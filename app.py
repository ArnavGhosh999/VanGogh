import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, ctx
import plotly.graph_objects as go
import base64
import io
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import time
import matplotlib.pyplot as plt

# Initialize app
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP], 
                suppress_callback_exceptions=True,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])

server = app.server

# Add custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Van Gogh Style Transfer</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background: linear-gradient(135deg, #FFCBA4 0%, #FFDAB9 50%, #FFE4C4 100%);
                background-attachment: fixed;
                color: #CC0000;
                font-family: 'Arial', sans-serif;
            }
            
            @keyframes float {
                0% { transform: translateY(0px); }
                50% { transform: translateY(-10px); }
                100% { transform: translateY(0px); }
            }
            
            @keyframes rotate {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            @keyframes paintStroke {
                0% { transform: rotate(-15deg) translateX(0) translateY(0); }
                25% { transform: rotate(10deg) translateX(5px) translateY(5px); }
                50% { transform: rotate(-5deg) translateX(10px) translateY(0); }
                75% { transform: rotate(15deg) translateX(5px) translateY(-5px); }
                100% { transform: rotate(-15deg) translateX(0) translateY(0); }
            }
            
            .main-title {
                font-size: 2.5rem;
                font-weight: bold;
                text-align: center;
                color: #CC0000;
                margin-top: 20px;
                margin-bottom: 20px;
                text-shadow: 2px 2px 4px rgba(204, 0, 0, 0.2);
                animation: float 5s ease-in-out infinite;
            }
            
            .sub-title {
                font-size: 1.2rem;
                text-align: center;
                color: #CC0000;
                margin-bottom: 30px;
                opacity: 0.9;
            }
            
            .card {
                border-radius: 15px;
                box-shadow: 0 10px 25px rgba(204, 0, 0, 0.2);
                border: 2px solid rgba(204, 0, 0, 0.3);
                background-color: rgba(255, 255, 255, 0.7);
                transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                margin-bottom: 20px;
                overflow: hidden;
            }
            
            .card:hover {
                transform: translateY(-8px);
                box-shadow: 0 15px 35px rgba(204, 0, 0, 0.3);
                border: 2px solid rgba(204, 0, 0, 0.6);
            }
            
            .card-title {
                color: #CC0000;
                font-weight: bold;
                text-align: center;
                margin-bottom: 15px;
                font-size: 1.3rem;
            }
            
            .result-card {
                border-radius: 15px;
                box-shadow: 0 15px 35px rgba(204, 0, 0, 0.25);
                border: 3px solid rgba(204, 0, 0, 0.3);
                background: linear-gradient(135deg, rgba(255, 235, 205, 0.9) 0%, rgba(255, 218, 185, 0.9) 100%);
                padding: 20px;
                margin-bottom: 30px;
                animation: float 6s ease-in-out infinite;
            }
            
            .result-card:hover {
                transform: translateY(-10px);
                box-shadow: 0 20px 40px rgba(204, 0, 0, 0.3);
                border: 3px solid rgba(204, 0, 0, 0.5);
            }
            
            .brush-icon {
                font-size: 32px;
                display: inline-block;
                animation: paintStroke 3s ease-in-out infinite;
                margin: 0 10px;
            }
            
            .floating-icon {
                animation: float 3s ease-in-out infinite;
                display: inline-block;
            }
            
            .btn-primary {
                background: linear-gradient(90deg, #CC0000, #FF3333);
                border: none;
                box-shadow: 0 5px 15px rgba(204, 0, 0, 0.3);
                transition: all 0.3s ease;
                font-weight: bold;
                letter-spacing: 1px;
                border-radius: 10px;
                padding: 10px 20px;
            }
            
            .btn-primary:hover {
                background: linear-gradient(90deg, #AA0000, #CC0000);
                transform: translateY(-3px) scale(1.03);
                box-shadow: 0 10px 25px rgba(204, 0, 0, 0.4);
            }
            
            .form-control {
                border: 2px solid rgba(204, 0, 0, 0.3);
                border-radius: 10px;
                transition: all 0.3s ease;
                background-color: rgba(255, 255, 255, 0.7);
            }
            
            .form-control:focus {
                border-color: rgba(204, 0, 0, 0.7);
                box-shadow: 0 5px 15px rgba(204, 0, 0, 0.1);
            }
            
            .upload-box {
                border: 3px dashed rgba(204, 0, 0, 0.4);
                border-radius: 15px;
                padding: 30px;
                text-align: center;
                transition: all 0.4s ease;
                background-color: rgba(255, 255, 255, 0.5);
                margin-bottom: 20px;
                cursor: pointer;
            }
            
            .upload-box:hover {
                border-color: rgba(204, 0, 0, 0.7);
                background-color: rgba(255, 235, 205, 0.8);
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(204, 0, 0, 0.2);
            }
            
            .evolution-container {
                background: linear-gradient(135deg, rgba(255, 224, 189, 0.85) 0%, rgba(255, 203, 164, 0.85) 100%);
                padding: 20px;
                border-radius: 15px;
                border: 2px solid rgba(204, 0, 0, 0.3);
                box-shadow: 0 15px 35px rgba(204, 0, 0, 0.15);
                margin-top: 30px;
                position: relative;
            }
            
            .evolution-image {
                border: 2px solid rgba(204, 0, 0, 0.3);
                border-radius: 10px;
                padding: 5px;
                margin: 5px;
                transition: all 0.3s ease;
                background-color: rgba(255, 255, 255, 0.7);
            }
            
            .evolution-image:hover {
                transform: scale(1.1) translateY(-5px);
                box-shadow: 0 10px 20px rgba(204, 0, 0, 0.2);
                border-color: rgba(204, 0, 0, 0.5);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <div style="text-align: center; padding: 10px; background: linear-gradient(90deg, rgba(255, 203, 164, 0.9), rgba(255, 218, 185, 0.9)); margin-top: 40px; border-top: 2px solid rgba(204, 0, 0, 0.3);">
            <p style="color: #CC0000; margin: 0;">
                ✨ Powered by PyTorch and Neural Style Transfer ✨
            </p>
            <p style="color: #CC0000; font-size: 0.8rem; margin: 0; opacity: 0.8;">
                Transform your photos into artistic masterpieces with the power of AI
            </p>
        </div>
    </body>
</html>
'''

# Helper functions for style transfer
def load_image(image_bytes, max_size=512):
    """Load and preprocess an uploaded image"""
    if image_bytes is not None:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
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
    """Neural style transfer implementation"""
    # Start with a copy of the content image
    input_img = content_img.clone().detach().requires_grad_(True)
    
    # Load VGG19 model with newer weights parameter
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
    
    # Use Adam optimizer
    optimizer = torch.optim.Adam([input_img], lr=0.05)
    
    # Store intermediate results
    intermediate_results = []
    loss_history = {"content": [], "style": [], "total": [], "step": []}
    
    # Optimization loop
    for step in range(num_steps):
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
        
        # Apply weight to style loss
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
    
    return input_img.detach(), intermediate_results, loss_history

def plot_loss_chart(loss_history):
    """Create a loss history chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=loss_history["step"], 
        y=loss_history["content"],
        mode='lines',
        name='Content Loss',
        line=dict(color='#CC0000', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=loss_history["step"], 
        y=loss_history["style"],
        mode='lines',
        name='Style Loss',
        line=dict(color='#800000', width=3)
    ))
    
    fig.update_layout(
        title={
            'text': 'Loss History',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color='#CC0000')
        },
        xaxis_title="Step",
        yaxis_title="Loss",
        plot_bgcolor='rgba(255,228,196,0.7)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(204,0,0,0.1)',
            title_font=dict(size=18, color='#CC0000'),
            tickfont=dict(size=14, color='#CC0000')
        ),
        yaxis=dict(
            gridcolor='rgba(204,0,0,0.1)',
            title_font=dict(size=18, color='#CC0000'),
            tickfont=dict(size=14, color='#CC0000')
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,235,205,0.7)',
            bordercolor='rgba(204,0,0,0.3)',
            font=dict(color='#CC0000')
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        height=500
    )
    
    return fig

def generate_evolution_images(intermediate_results):
    """Generate HTML for displaying the evolution images"""
    evolution_imgs = []
    for i, tensor in enumerate(intermediate_results):
        step_num = i * (100 // len(intermediate_results))
        img = tensor_to_image(tensor)
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        
        # Save to bytes
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        img_html = html.Div([
            html.Img(src=f'data:image/png;base64,{encoded_img}', 
                    className='evolution-image',
                    style={'height': '150px'}),
            html.Div(f"Step {step_num}", 
                    style={'textAlign': 'center', 'color': '#CC0000', 'fontSize': '0.9rem'})
        ], style={'display': 'inline-block', 'margin': '10px'})
        
        evolution_imgs.append(img_html)
    
    return evolution_imgs

# App layout
app.layout = html.Div([
    # Title section
    html.H1("Van Gogh Style Transfer", className="main-title"),
    html.P("Transform photos into artistic masterpieces", className="sub-title"),
    
    # Icon decorations
    html.Div([
        html.Span("🖌️", className="brush-icon"),
        html.Span("🎨", className="brush-icon", style={"animation-delay": "0.5s"}),
        html.Span("🖌️", className="brush-icon", style={"animation-delay": "1s"})
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # Main container
    dbc.Container([
        dbc.Row([
            # Left column - Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Style Transfer Controls", className="card-title"),
                        
                        html.P("Processing Steps", style={'fontWeight': 'bold', 'color': '#CC0000', 'marginTop': '20px'}),
                        dcc.Slider(
                            id='num-steps-slider',
                            min=20,
                            max=150,
                            step=5,
                            value=50,
                            marks={20: '20', 80: '80', 150: '150'},
                        ),
                        
                        html.P("Style Strength", style={'fontWeight': 'bold', 'color': '#CC0000', 'marginTop': '20px'}),
                        dcc.Slider(
                            id='style-weight-slider',
                            min=1,
                            max=10,
                            step=0.5,
                            value=5,
                            marks={1: 'Low', 5: 'Med', 10: 'High'},
                        ),
                        
                        html.Div([
                            dbc.Button(
                                [
                                    html.Span("✨", className="floating-icon"),
                                    " Generate Masterpiece ",
                                    html.Span("✨", className="floating-icon")
                                ],
                                id="process-button", 
                                color="primary", 
                                className="mt-4 mb-2 w-100"
                            )
                        ], style={'textAlign': 'center', 'marginTop': '30px'})
                    ])
                ]),
                
                # Processing status
                html.Div(id='processing-status', className='mt-3'),
                
                # Add decorative elements
                html.Div([
                    html.Div("🖌️", className="brush-icon", style={"fontSize": "32px"}),
                    html.Div("🎨", className="brush-icon", style={"fontSize": "36px", "animationDelay": "0.7s"}),
                    html.Div("🖌️", className="brush-icon", style={"fontSize": "32px", "animationDelay": "1.4s"})
                ], style={'textAlign': 'center', 'marginTop': '40px'})
            ], md=4),
            
            # Right column - Upload and Results
            dbc.Col([
                # Upload section
                dbc.Row([
                    dbc.Col([
                        html.H5("Content Image", style={'textAlign': 'center', 'color': '#CC0000', 'fontWeight': 'bold'}),
                        dcc.Upload(
                            id='upload-content',
                            children=html.Div([
                                '📷 Drag and Drop or ',
                                html.A('Select Content Image', style={'color': '#CC0000', 'fontWeight': 'bold'})
                            ]),
                            className='upload-box',
                            multiple=False
                        ),
                        html.Div(id='content-image-display')
                    ], md=6),
                    
                    dbc.Col([
                        html.H5("Style Image", style={'textAlign': 'center', 'color': '#CC0000', 'fontWeight': 'bold'}),
                        dcc.Upload(
                            id='upload-style',
                            children=html.Div([
                                '🖼️ Drag and Drop or ',
                                html.A('Select Style Image', style={'color': '#CC0000', 'fontWeight': 'bold'})
                            ]),
                            className='upload-box',
                            multiple=False
                        ),
                        html.Div(id='style-image-display')
                    ], md=6)
                ]),
                
                # Result section
                html.Div(id='result-display'),
                
                # Loss chart
                html.Div(id='loss-chart-container'),
                
                # Evolution display
                html.Div(id='evolution-display')
            ], md=8)
        ])
    ], fluid=True),
    
    # Store components for saving state
    dcc.Store(id='content-image-store'),
    dcc.Store(id='style-image-store'),
    dcc.Store(id='result-image-store'),
    dcc.Store(id='evolution-store'),
    dcc.Store(id='loss-history-store')
])

# Callbacks for image uploads
@callback(
    Output('content-image-display', 'children'),
    Output('content-image-store', 'data'),
    Input('upload-content', 'contents'),
    State('upload-content', 'filename')
)
def update_content_image(contents, filename):
    if contents is None:
        return html.Div(), None
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    return html.Div([
        html.Img(src=contents, style={'width': '100%', 'borderRadius': '10px', 'marginTop': '10px'}),
        html.P(filename, style={'textAlign': 'center', 'marginTop': '5px'})
    ]), content_string
    
@callback(
    Output('style-image-display', 'children'),
    Output('style-image-store', 'data'),
    Input('upload-style', 'contents'),
    State('upload-style', 'filename')
)
def update_style_image(contents, filename):
    if contents is None:
        return html.Div(), None
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    return html.Div([
        html.Img(src=contents, style={'width': '100%', 'borderRadius': '10px', 'marginTop': '10px'}),
        html.P(filename, style={'textAlign': 'center', 'marginTop': '5px'})
    ]), content_string

# Callback for style transfer processing
@callback(
    Output('processing-status', 'children'),
    Output('result-image-store', 'data'),
    Output('evolution-store', 'data'),
    Output('loss-history-store', 'data'),
    Input('process-button', 'n_clicks'),
    State('content-image-store', 'data'),
    State('style-image-store', 'data'),
    State('num-steps-slider', 'value'),
    State('style-weight-slider', 'value'),
    prevent_initial_call=True
)
def process_style_transfer(n_clicks, content_data, style_data, num_steps, style_weight_slider):
    if not n_clicks or content_data is None or style_data is None:
        return html.Div(), None, None, None
    
    # Show processing message
    processing_status = html.Div([
        html.Div([
            html.Div("Creating Your Masterpiece...", 
                    style={'color': '#CC0000', 'fontWeight': 'bold', 'fontSize': '1.2rem', 'textAlign': 'center', 'marginBottom': '15px'}),
            html.Div(style={'textAlign': 'center'}, children=[
                html.Span("🖌️", className="brush-icon"),
                " Processing... ",
                html.Span("🎨", className="floating-icon")
            ]),
            dbc.Progress(value=100, striped=True, animated=True, 
                        style={'height': '10px', 'borderRadius': '5px', 'marginTop': '10px'})
        ], className="card p-3")
    ])
    
    # Convert style_weight_slider (1-10) to actual style weight (10,000 - 2,000,000)
    style_weight = int(style_weight_slider * 200000)
    
    try:
        # Convert base64 data to tensors
        content_decoded = base64.b64decode(content_data)
        style_decoded = base64.b64decode(style_data)
        
        content_tensor, content_img = load_image(content_decoded)
        style_tensor, style_img = load_image(style_decoded)
        
        # Determine device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move tensors to device
        content_tensor = content_tensor.to(device)
        style_tensor = style_tensor.to(device)
        
        # Run style transfer
        output_tensor, intermediate_results, loss_history = run_style_transfer(
            content_tensor,
            style_tensor,
            num_steps=num_steps,
            style_weight=style_weight,
            device=device
        )
        
        # Convert result to image
        output_image = tensor_to_image(output_tensor)
        output_pil = Image.fromarray((output_image * 255).astype(np.uint8))
        
        # Convert to base64 string
        buffer = io.BytesIO()
        output_pil.save(buffer, format="PNG")
        encoded_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Store intermediate results
        intermediate_encoded = []
        for result in intermediate_results:
            img = tensor_to_image(result)
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            
            buffer = io.BytesIO()
            img_pil.save(buffer, format="PNG")
            intermediate_encoded.append(base64.b64encode(buffer.getvalue()).decode('utf-8'))
        
        # Success message
        processing_status = html.Div([
            dbc.Alert("Style transfer completed successfully! ✨", color="success", style={'textAlign': 'center'})
        ])
        
        return processing_status, encoded_img, intermediate_encoded, loss_history
    
    except Exception as e:
        # Error message
        processing_status = html.Div([
            dbc.Alert(f"Error during processing: {str(e)}", color="danger")
        ])
        
        return processing_status, None, None, None

# Callback to display results
@callback(
    Output('result-display', 'children'),
    Output('loss-chart-container', 'children'),
    Output('evolution-display', 'children'),
    Input('result-image-store', 'data'),
    Input('loss-history-store', 'data'),
    Input('evolution-store', 'data'),
    prevent_initial_call=True
)
def display_results(result_data, loss_history, evolution_data):
    if result_data is None:
        return html.Div(), html.Div(), html.Div()
    
    # Result display
    result_display = html.Div([
        html.H3("Stylized Result", style={'textAlign': 'center', 'color': '#CC0000', 'marginTop': '30px', 'marginBottom': '20px'}),
        
        # Artistic decoration
        html.Div([
            html.Span("🖌️", className="brush-icon"),
            html.Span("✨", className="floating-icon"),
            html.Span("🎨", className="brush-icon", style={"animation-delay": "0.5s"})
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Result card
        html.Div([
            html.H4("Stylized Masterpiece", style={'textAlign': 'center', 'color': '#CC0000', 'fontWeight': 'bold', 'marginBottom': '20px'}),
            html.Img(src=f'data:image/png;base64,{result_data}', 
                   style={'maxWidth': '100%', 'borderRadius': '10px', 'display': 'block', 'margin': '0 auto', 
                          'boxShadow': '0 10px 30px rgba(204, 0, 0, 0.2)', 'border': '2px solid rgba(204, 0, 0, 0.2)'}),
            
            # Download button
            html.Div([
                html.A(
                    dbc.Button("💾 Download Masterpiece", color="primary", className="mt-3"),
                    href=f'data:image/png;base64,{result_data}',
                    download='stylized_image.png'
                )
            ], style={'textAlign': 'center', 'marginTop': '20px'})
        ], className="result-card")
    ])
    
    # Loss chart
    if loss_history:
        loss_chart = html.Div([
            html.H3("Loss History", style={'textAlign': 'center', 'color': '#CC0000', 'marginTop': '30px', 'marginBottom': '20px'}),
            dcc.Graph(figure=plot_loss_chart(loss_history))
        ])
    else:
        loss_chart = html.Div()
    
    # Evolution display
    if evolution_data:
        evolution_images = []
        for i, img_data in enumerate(evolution_data):
            step_num = i * (100 // len(evolution_data))
            img_div = html.Div([
                html.Img(
                    src=f'data:image/png;base64,{img_data}', 
                    className='evolution-image',
                    style={'height': '150px'}
                ),
                html.Div(
                    f"Step {step_num}", 
                    style={'textAlign': 'center', 'color': '#CC0000', 'fontSize': '0.9rem', 'marginTop': '5px'}
                )
            ], style={'display': 'inline-block', 'margin': '10px'})
            
            evolution_images.append(img_div)
        
        evolution_display = html.Div([
            html.H3("Style Transfer Evolution", style={'textAlign': 'center', 'color': '#CC0000', 'marginTop': '30px', 'marginBottom': '20px'}),
            html.Div(evolution_images, className="evolution-container")
        ])
    else:
        evolution_display = html.Div()
    
    return result_display, loss_chart, evolution_display

# Run the app
if __name__ == '__main__':
    app.run(debug=True)