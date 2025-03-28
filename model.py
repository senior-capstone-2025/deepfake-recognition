import os
import cv2
import numpy as np
import torch
import mediapipe as mp
import librosa
import noisereduce as nr
import subprocess
import pickle
from tqdm import tqdm
import torchvision.models as models
from torch import nn
import webrtcvad
import wave
import struct
from collections import deque
import torch.nn.functional as F

# Import StyleGRU from the codebase
from StyleGRU.model.StyleGRU import StyleGRU
from StyleGRU.dataloader.triplet_clip_loader import get_diff



class StyleAttention(nn.Module):
    """Style Attention module to fuse StyleGRU features with ResNet features"""
    def __init__(self, style_dim, content_dim, output_dim):
        super(StyleAttention, self).__init__()
        self.style_projection = nn.Linear(style_dim, output_dim)
        self.content_projection = nn.Linear(content_dim, output_dim)
        self.attention = nn.MultiheadAttention(output_dim, num_heads=8)
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, style_features, content_features):
        # Project both feature types to the same dimension
        style_proj = self.style_projection(style_features)
        content_proj = self.content_projection(content_features)
        
        # Use attention to fuse the features
        attn_output, _ = self.attention(
            query=content_proj.unsqueeze(0),
            key=style_proj.unsqueeze(0),
            value=style_proj.unsqueeze(0)
        )
        
        # Concatenate attention output with content features
        fused_features = torch.cat([content_proj, attn_output.squeeze(0)], dim=1)
        
        # Final fusion
        output = self.fusion(fused_features)
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=16):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1)]


class TemporalTransformerEncoder(nn.Module):
    """Temporal transformer encoder for sequence modeling"""
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, dropout=0.1):
        super(TemporalTransformerEncoder, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x, mask=None):
        # Project input to transformer dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer encoder
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        return output


class AudioClassifier(nn.Module):
    """Audio classifier for deepfake detection"""
    def __init__(self, input_dim, hidden_dim=512):
        super(AudioClassifier, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # Apply classifier and sigmoid activation
        return torch.sigmoid(self.model(x))


class MultimodalDeepfakeDetector(nn.Module):
    """Complete multimodal deepfake detection model with style attention and transformer"""
    def __init__(self, 
                 style_dim=8192, 
                 content_dim=512, 
                 fusion_dim=1024,
                 audio_dim=120,
                 transformer_dim=512,
                 num_heads=8,
                 num_layers=4,
                 sequence_length=16):
        super(MultimodalDeepfakeDetector, self).__init__()
        
        # Video processing components
        self.style_gru = StyleGRU(feature_size=9216)
        self.style_attention = StyleAttention(
            style_dim=style_dim,
            content_dim=content_dim,
            output_dim=fusion_dim
        )
        
        # Temporal transformer for sequence modeling
        self.temporal_transformer = TemporalTransformerEncoder(
            input_dim=fusion_dim,
            d_model=transformer_dim,
            nhead=num_heads,
            num_layers=num_layers
        )
        
        # Video classifier
        self.video_classifier = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(transformer_dim // 2, 1)
        )
        
        # Audio classifier
        self.audio_classifier = AudioClassifier(audio_dim)
        
        # Final fusion layer for multimodal decision
        self.fusion_classifier = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        
        # Sequence length for temporal processing
        self.sequence_length = sequence_length
        
    def forward(self, style_features, content_features, audio_features=None):
        # Process through StyleGRU
        _, style_hidden = self.style_gru(style_features)
        
        # Fuse features with style attention
        fused_features = self.style_attention(style_hidden, content_features)
        
        # Reshape for transformer if needed
        if len(fused_features.shape) == 1:
            # If we have a single feature vector, expand to sequence
            fused_features = fused_features.unsqueeze(0).unsqueeze(0)
            fused_features = fused_features.expand(-1, self.sequence_length, -1)
        elif len(fused_features.shape) == 2 and fused_features.shape[0] == 1:
            # If we have a batch of 1, expand to sequence
            fused_features = fused_features.unsqueeze(1)
            fused_features = fused_features.expand(-1, self.sequence_length, -1)
        
        # Process through temporal transformer
        transformer_output = self.temporal_transformer(fused_features)
        
        # Global average pooling over sequence dimension
        pooled_output = torch.mean(transformer_output, dim=1)
        
        # Video classification
        video_pred = torch.sigmoid(self.video_classifier(pooled_output))
        
        # Audio classification if audio features are provided
        if audio_features is not None:
            audio_pred = self.audio_classifier(audio_features)
            
            # Combine video and audio predictions
            combined_preds = torch.cat([video_pred, audio_pred], dim=1)
            final_pred = self.fusion_classifier(combined_preds)
            
            return final_pred, video_pred, audio_pred
        else:
            # Return only video prediction if no audio
            return video_pred, video_pred, None
    
    def predict_video(self, video_path, processor):
        """Make prediction on a video file"""
        # Process video
        features = processor.process_video(video_path)
        
        if features is None:
            return None
        
        # Prepare inputs
        style_features = torch.FloatTensor(features['style_features']).unsqueeze(0)
        content_features = torch.FloatTensor(features['resnet_features'][0]).unsqueeze(0)
        
        if len(features['audio_features']) > 0:
            # Average audio features across time
            audio_features = torch.FloatTensor(np.mean(features['audio_features'], axis=0)).unsqueeze(0)
        else:
            audio_features = None
        
        # Move to GPU if available
        if torch.cuda.is_available():
            style_features = style_features.cuda()
            content_features = content_features.cuda()
            if audio_features is not None:
                audio_features = audio_features.cuda()
        
        # Make prediction
        with torch.no_grad():
            final_pred, video_pred, audio_pred = self.forward(
                style_features, content_features, audio_features
            )
        
        # Return predictions
        result = {
            'final_prediction': final_pred.cpu().numpy()[0][0],
            'video_prediction': video_pred.cpu().numpy()[0][0],
            'audio_prediction': audio_pred.cpu().numpy()[0][0] if audio_pred is not None else None,
            'is_fake': final_pred.cpu().numpy()[0][0] > 0.5
        }
        
        return result




