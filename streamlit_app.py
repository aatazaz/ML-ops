
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import pandas as pd
import plotly.graph_objects as go
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
from transformers import ViTModel, ViTImageProcessor

# --- Model Architectures (Copied from previous cells) ---
class BaselineRNN(nn.Module):
    """Simple RNN for sequence classification"""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5):
        super().__init__()
        rnn_kwargs = {
            'input_size': input_dim,
            'hidden_size': hidden_dim,
            'num_layers': num_layers,
            'batch_first': True,
        }
        if num_layers > 1:
            rnn_kwargs['dropout'] = dropout
        self.rnn = nn.RNN(**rnn_kwargs)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        output, hidden = self.rnn(x)
        # Use last time step
        last_output = output[:, -1, :]
        return self.classifier(last_output)

class BaselineLSTM(nn.Module):
    """LSTM for sequence classification"""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5):
        super().__init__()
        lstm_kwargs = {
            'input_size': input_dim,
            'hidden_size': hidden_dim,
            'num_layers': num_layers,
            'batch_first': True,
            'bidirectional': False
        }
        if num_layers > 1:
            lstm_kwargs['dropout'] = dropout
        self.lstm = nn.LSTM(**lstm_kwargs)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        last_output = output[:, -1, :]
        return self.classifier(last_output)

class BaselineGRU(nn.Module):
    """GRU for sequence classification"""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5):
        super().__init__()
        gru_kwargs = {
            'input_size': input_dim,
            'hidden_size': hidden_dim,
            'num_layers': num_layers,
            'batch_first': True,
            'bidirectional': False
        }
        if num_layers > 1:
            gru_kwargs['dropout'] = dropout
        self.gru = nn.GRU(**gru_kwargs)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        output, hidden = self.gru(x)
        last_output = output[:, -1, :]
        return self.classifier(last_output)

class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM captures context from both directions"""
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim # Store hidden_dim as an instance variable
        lstm_kwargs = {
            'input_size': input_dim,
            'hidden_size': hidden_dim,
            'num_layers': num_layers,
            'batch_first': True,
            'bidirectional': True
        }
        if num_layers > 1:
            lstm_kwargs['dropout'] = dropout
        self.lstm = nn.LSTM(**lstm_kwargs)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        )

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        # Concatenate forward and backward final states
        forward_last = output[:, -1, :self.hidden_dim] # Use self.hidden_dim
        backward_last = output[:, 0, self.hidden_dim:] # Use self.hidden_dim
        combined = torch.cat([forward_last, backward_last], dim=1)
        return self.classifier(combined)

class StackedLSTM(nn.Module):
    """Deep LSTM with 3 layers"""
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        # Stacked LSTMs always have num_layers > 1 implicitly
        # However, for consistency, we pass dropout only if it's > 0 (or num_layers > 1 is explicit)
        lstm_kwargs = {
            'input_size': input_dim,
            'hidden_size': hidden_dim,
            'batch_first': True,
        }

        # These are always single layer LSTMs in this stacked setup, so dropout will be 0.
        # Explicitly not passing dropout if num_layers is 1 to avoid TypeError
        # This class effectively uses num_layers=1 for each individual LSTM, then stacks them.
        # If the dropout argument is intended for the _LSTM_ constructor itself
        # then the dropout should not be passed to single-layer LSTMs. 
        # However, the previous implementation did pass it with dropout=0.5 for dropout1, dropout2.
        # Reverting to original logic for consistency with previous definition, but still wrapping in kwargs.
        # The dropout in __init__ for StackedLSTM was not for the individual LSTMs but for the sequential layers of Dropout
        
        # No, the StackedLSTM was designed to apply dropout between layers, so each LSTM layer is individual without dropout arg
        # The dropout here refers to the nn.Dropout layers, not the internal LSTM dropout
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        last_output = out[:, -1, :]
        return self.classifier(last_output)

class VideoTransformer(nn.Module):
    """Transformer-based model for video sequences"""
    def __init__(self, input_dim, num_classes, num_heads=8,
                 num_layers=4, dropout=0.1, max_seq_len=100):
        super().__init__()

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification token (like [CLS] in BERT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding (truncate if needed)
        x = x + self.pos_encoding[:, :seq_len+1]
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Use classification token for prediction
        cls_output = x[:, 0, :]

        return self.classifier(cls_output)


# --- Feature Extractor Class (Copied from previous cells) ---
class VideoFeatureExtractor:
    """ Extract spatial features from video frames using pre-trained models """

    def __init__(self, model_name='resnet50', device='cpu',
                 feature_layer='avgpool'):
        """
        Args:
            model_name: 'resnet18', 'resnet50', 'resnet101', 'efficientnet', 'vit'
            device: torch device
            feature_layer: Which layer to extract features from
        """
        self.device = device
        self.model_name = model_name
        self.feature_layer = feature_layer

        self.model, self.feature_dim = self.load_pretrained_model()
        self.model = self.model.to(device)
        self.model.eval()

    def load_pretrained_model(self):
        """Load pre-trained model"""
        if 'resnet' in self.model_name:
            if self.model_name == 'resnet18':
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                feature_dim = 512
            elif self.model_name == 'resnet50':
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                feature_dim = 2048
            elif self.model_name == 'resnet101':
                model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
                feature_dim = 2048

            # Remove final classification layer
            model = nn.Sequential(*list(model.children())[:-1])

        elif 'efficientnet' in self.model_name:
            if self.model_name == 'efficientnet_b0':
                model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                feature_dim = 1280
            elif self.model_name == 'efficientnet_b4':
                model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
                feature_dim = 1792

            # Remove classification head
            model.classifier = nn.Identity()

        elif self.model_name == 'vit':
            # from transformers import ViTModel, ViTImageProcessor # Already imported above

            self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            model = ViTModel.from_pretrained('google/vit-base-patch16-224')
            feature_dim = 768

        else:
            raise ValueError(f"Model {self.model_name} not supported")

        return model, feature_dim

    def extract_features(self, video_batch):
        """
        Extract features from batch of videos
        Args:
            video_batch: Tensor of shape [B, T, C, H, W]
        Returns:
            features: Tensor of shape [B, T, feature_dim]
        """
        batch_size, seq_len = video_batch.shape[:2]

        with torch.no_grad():
            if self.model_name == 'vit':
                # ViT needs special processing
                features = self.extract_vit_features(video_batch)
            else:
                # Reshape for batch processing: [B*T, C, H, W]
                frames_flat = video_batch.view(-1, *video_batch.shape[2:])
                frames_flat = frames_flat.to(self.device)

                # Extract features
                features_flat = self.model(frames_flat)

                # Reshape features
                if isinstance(features_flat, tuple):
                    features_flat = features_flat[0]

                # Flatten spatial dimensions
                features_flat = features_flat.view(features_flat.size(0), -1)

                # Reshape back to [B, T, feature_dim]
                features = features_flat.view(batch_size, seq_len, -1)

        return features.cpu()

    def extract_vit_features(self, video_batch):
        """Extract features using Vision Transformer"""
        batch_size, seq_len = video_batch.shape[:2]

        # Convert to PIL images for ViT processor
        frames_pil = []
        for i in range(video_batch.shape[0] * video_batch.shape[1]):
            frame = video_batch.view(-1, *video_batch.shape[2:])[i]
            # Denormalize for PIL conversion
            frame = frame * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            frame = torch.clip(frame, 0, 1)
            frame = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frames_pil.append(Image.fromarray(frame))

        # Process through ViT
        inputs = self.processor(frames_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            features_flat = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Reshape to [B, T, feature_dim]
        features = features_flat.view(batch_size, seq_len, -1)

        return features.cpu()

# --- Streamlit App Functions ---

def create_streamlit_app(best_model_path, best_model_name, feature_dim, feature_extractor, class_names):
    """
    Create Streamlit web interface for video action recognition
    """

    st.set_page_config(
        page_title="Video Action Recognition",
        page_icon="🎬",
        layout="wide"
    )

    # Title and description
    st.title("🎬 Video Action Recognition System")
    st.markdown("""
    Upload a short video and the AI model will recognize the activity with confidence scores.
    Built with PyTorch and Streamlit.
    """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Model Information")

        # Load best model
        @st.cache_resource
        def load_model(model_path, model_name, input_dim, num_classes):
            # Instantiate the correct model class based on best_model_name
            # Ensure model_name is stripped of any whitespace
            model_name_stripped = model_name.strip()
            if model_name_stripped == 'RNN':
                model_instance = BaselineRNN(input_dim=input_dim, hidden_dim=256, num_classes=num_classes)
            elif model_name_stripped == 'LSTM':
                model_instance = BaselineLSTM(input_dim=input_dim, hidden_dim=256, num_classes=num_classes)
            elif model_name_stripped == 'GRU':
                model_instance = BaselineGRU(input_dim=input_dim, hidden_dim=256, num_classes=num_classes)
            elif model_name_stripped == 'Bidirectional LSTM':
                model_instance = BidirectionalLSTM(input_dim=input_dim, hidden_dim=256, num_classes=num_classes)
            elif model_name_stripped == 'Stacked LSTM':
                model_instance = StackedLSTM(input_dim=input_dim, hidden_dim=256, num_classes=num_classes)
            elif model_name_stripped == 'Transformer':
                model_instance = VideoTransformer(input_dim=input_dim, num_classes=num_classes)
            else:
                raise ValueError(f"Unknown model name: {model_name}")

            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model_instance.load_state_dict(checkpoint['model_state_dict'])
            model_instance.eval()
            return model_instance, checkpoint

        model, checkpoint = load_model(best_model_path, best_model_name, feature_dim, len(class_names))
        st.success(f"✅ Model loaded: {checkpoint.get('model_name', best_model_name)}")

        # Model stats
        total_params = sum(p.numel() for p in model.parameters())
        st.metric("Model Parameters", f"{total_params:,}")
        st.metric("Number of Classes", len(class_names))

        st.header("📊 Performance")
        st.metric("Top-1 Accuracy", f"{checkpoint.get('val_acc', 'N/A'):.1f}%")

        st.header("ℹ️ Instructions")
        st.info("""
        1. Upload a short video (5-10 seconds)
        2. Click 'Analyze Video'
        3. View predictions and confidence scores
        """)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📤 Upload Video")

        # Video upload
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a short video for action recognition"
        )

        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            # Display video
            st.video(uploaded_file)

            # Analyze button
            if st.button("🔍 Analyze Video", type="primary"):
                with st.spinner("Processing video..."):
                    # Process video
                    predictions = process_video(video_path, model, feature_extractor, class_names)

                    # Display results
                    st.header("🎯 Prediction Results")

                    # Top prediction
                    top_pred = predictions[0]
                    st.success(f"**Predicted Action:** {top_pred['class']}")
                    st.metric("Confidence", f"{top_pred['confidence']:.1%}")

                    # All predictions
                    st.subheader("📊 All Predictions")

                    # Create bar chart
                    classes_to_display = [p['class'] for p in predictions[:10]]
                    confidences = [p['confidence'] for p in predictions[:10]]

                    fig = go.Figure(data=[
                        go.Bar(
                            x=confidences,
                            y=classes_to_display,
                            orientation='h',
                            marker_color='steelblue',
                            text=[f'{c:.1%}' for c in confidences],
                            textposition='outside'
                        )
                    ])

                    fig.update_layout(
                        title="Top 10 Predictions",
                        xaxis_title="Confidence",
                        yaxis_title="Action Class",
                        height=400,
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Detailed table
                    st.subheader("📋 Detailed Scores")

                    df_predictions = pd.DataFrame(
                        [{
                            'Class': p['class'],
                            'Confidence': p['confidence']
                        }]
                         for p in predictions[:10]
                    )
                    st.dataframe(
                        df_predictions.style.format({'Confidence': '{:.2%}'}),
                        use_container_width=True
                    )

            # Clean up
            os.unlink(video_path)

    with col2:
        st.header("📈 Model Comparison")

        # Show model comparison chart
        try:
            comparison_img = Image.open('model_comparison.png')
            st.image(comparison_img, caption="Model Performance Comparison", use_column_width=True)
        except FileNotFoundError:
            st.info("Comparison chart will appear after training all models and the 'model_comparison.png' file is generated.")

        st.header("🎯 Sample Predictions")

        # Example predictions - These should be actual videos or representative images
        # For now, using placeholders, as actual video files might not be readily available in the environment
        examples = [
            {"title": "Example 1: Archery", "description": "A person shooting an arrow.", "video_url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4"},
            {"title": "Example 2: Basketball", "description": "A basketball player making a shot.", "video_url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4"},
            {"title": "Example 3: Jumping Jack", "description": "Someone performing jumping jacks.", "video_url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4"},
        ]

        for ex in examples:
            with st.expander(ex['title']):
                st.write(ex['description'])
                st.video(ex['video_url'])

def process_video(video_path, model, feature_extractor, class_names):
    """
    Process video and make predictions
    """
    # Extract frames from video
    frames = extract_frames_from_video(video_path, num_frames=16)

    # Preprocess frames
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    frames_processed = [transform(frame) for frame in frames]
    video_tensor = torch.stack(frames_processed, dim=0).unsqueeze(0)  # [1, T, C, H, W]

    # Set feature extractor to evaluation mode and move to cpu (streamlit usually runs on cpu)
    feature_extractor.model.eval()
    feature_extractor.device = 'cpu'
    feature_extractor.model.to('cpu')

    # Extract features
    with torch.no_grad():
        features = feature_extractor.extract_features(video_tensor)

    # Set model to evaluation mode and move to cpu
    model.eval()
    model.to('cpu')

    # Make prediction
    with torch.no_grad():
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1)

    # Get top predictions
    top_probs, top_indices = torch.topk(probs[0], k=len(class_names))

    # Format results
    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        predictions.append({
            'class': class_names[idx.item()],
            'confidence': prob.item(),
            'index': idx.item()
        })

    return predictions

def extract_frames_from_video(video_path, num_frames=16):
    """
    Extract frames from video
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
        frame_indices = (frame_indices * (num_frames // len(frame_indices) + 1))[:num_frames]
    else:
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            # Fallback: black frame
            frames.append(Image.new('RGB', (256, 256), color='black'))

    cap.release()
    return frames

# --- Main application entry point ---
if __name__ == '__main__':
    # A dummy global device for feature extractor. Streamlit app will force CPU.
    device_global = 'cpu'

    # Instantiate feature extractor for use in the Streamlit app
    @st.cache_resource
    def get_feature_extractor():
        # Assuming 'resnet50' was used, modify if different
        return VideoFeatureExtractor(model_name='resnet50', device=device_global)

    feature_extractor_app = get_feature_extractor()

    # Dynamic values from the notebook execution
    best_model_path_app = 'best_RNN.pth'
    best_model_name_app = 'RNN'
    feature_dim_app = 2048
    class_names_app = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth']

    create_streamlit_app(best_model_path_app, best_model_name_app, feature_dim_app, feature_extractor_app, class_names_app)
