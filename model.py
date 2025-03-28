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

class VideoAudioProcessor:
    def __init__(self, 
                 output_dir='processed_data',
                 frame_rate=1,
                 face_confidence=0.5,
                 audio_sample_rate=16000,
                 style_gru_model_path=None,
                 style_feature_size=9216,
                 sequence_length=32):
        
        self.output_dir = output_dir
        self.frame_rate = frame_rate
        self.face_confidence = face_confidence
        self.audio_sample_rate = audio_sample_rate
        self.sequence_length = sequence_length
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'video_features'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'audio_features'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'combined_features'), exist_ok=True)
        
        # Initialize MediaPipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=face_confidence,
            min_tracking_confidence=face_confidence
        )
        
        # Initialize ResNet model for feature extraction
        self.resnet = models.video.r3d_18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove classification layer
        self.resnet.eval()
        if torch.cuda.is_available():
            self.resnet = self.resnet.cuda()
        
        # Initialize StyleGRU model if path is provided
        self.style_gru = None
        if style_gru_model_path and os.path.exists(style_gru_model_path):
            self.style_gru = StyleGRU(feature_size=style_feature_size)
            self.style_gru.load_state_dict(torch.load(style_gru_model_path))
            self.style_gru.eval()
            if torch.cuda.is_available():
                self.style_gru = self.style_gru.cuda()
        
        # Initialize VAD for voice activity detection
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (highest)
    
    def extract_frames(self, video_path):
        """Extract frames from video at specified frame rate"""
        frames = []
        timestamps = []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Calculate frame interval based on desired frame rate
        frame_interval = int(fps / self.frame_rate)
        
        current_frame = 0
        while cap.isOpened():
            if current_frame >= total_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Convert to RGB (MediaPipe uses RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            timestamps.append(current_frame / fps)
            
            current_frame += frame_interval
        
        cap.release()
        return frames, timestamps, duration
    
    def detect_and_crop_face(self, frame):
        """Detect facial landmarks and crop face region"""
        h, w = frame.shape[:2]
        results = self.face_mesh.process(frame)
        
        if not results.multi_face_landmarks:
            return None, None
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract landmark coordinates
        landmarks = []
        for landmark in face_landmarks.landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            landmarks.append((x, y))
        
        # Calculate bounding box with margin
        landmarks = np.array(landmarks)
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        
        # Add margin (20% of face size)
        margin_x = int((x_max - x_min) * 0.2)
        margin_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)
        
        # Crop face
        face_crop = frame[y_min:y_max, x_min:x_max]
        
        # Normalize landmarks to the cropped face coordinates
        norm_landmarks = landmarks.copy()
        norm_landmarks[:, 0] = norm_landmarks[:, 0] - x_min
        norm_landmarks[:, 1] = norm_landmarks[:, 1] - y_min
        
        return face_crop, norm_landmarks
    
    def extract_resnet_features(self, face_frames):
        """Extract features using 3D ResNet"""
        if len(face_frames) < 16:  # ResNet3D typically needs at least 16 frames
            # Pad with duplicates of the last frame if needed
            face_frames.extend([face_frames[-1]] * (16 - len(face_frames)))
        
        # Prepare frames for ResNet (normalize, resize, etc.)
        processed_frames = []
        for frame in face_frames:
            # Resize to ResNet input size
            frame = cv2.resize(frame, (112, 112))
            # Convert to float and normalize
            frame = frame.astype(np.float32) / 255.0
            # Normalize using ImageNet mean and std
            frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            # Change to channel-first format
            frame = np.transpose(frame, (2, 0, 1))
            processed_frames.append(frame)
        
        # Convert to tensor
        clip = torch.FloatTensor(np.array(processed_frames))
        # Add batch dimension
        clip = clip.unsqueeze(0)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            clip = clip.cuda()
        
        # Extract features
        with torch.no_grad():
            features = self.resnet(clip)
        
        return features.cpu().numpy().flatten()
    
    def extract_style_features(self, features_sequence):
        """Extract temporal style features using StyleGRU"""
        if self.style_gru is None:
            return None
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(np.array(features_sequence))
        # Add batch dimension
        features_tensor = features_tensor.unsqueeze(0)
        
        # Calculate temporal differences
        features_diff = get_diff(features_tensor)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            features_diff = features_diff.cuda()
        
        # Extract features
        with torch.no_grad():
            _, style_features = self.style_gru(features_diff)
        
        return style_features.cpu().numpy().flatten()
    
    def extract_audio(self, video_path, output_audio_path=None):
        """Extract audio from video using ffmpeg"""
        if output_audio_path is None:
            output_audio_path = os.path.splitext(video_path)[0] + '.wav'
        
        command = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(self.audio_sample_rate), '-ac', '1', output_audio_path,
            '-y'  # Overwrite if exists
        ]
        
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_audio_path
    
    def reduce_noise(self, audio_data, sr):
        """Apply noise reduction to audio"""
        # Reduce noise
        reduced_noise = nr.reduce_noise(
            y=audio_data,
            sr=sr,
            prop_decrease=0.75,
            stationary=True
        )
        return reduced_noise
    
    def detect_voice_activity(self, audio_path):
        """Detect voice activity in audio using WebRTC VAD"""
        # Open the wave file
        wf = wave.open(audio_path, 'rb')
        
        # Check if the audio format is compatible with VAD
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 48000]:
            raise ValueError("Audio file must be mono, 16-bit, and 8000/16000/32000/48000 Hz")
        
        # Process audio in 30ms chunks
        frame_duration_ms = 30
        frame_size = int(wf.getframerate() * frame_duration_ms / 1000)
        
        # Store chunks with voice activity
        voice_chunks = []
        
        while True:
            chunk = wf.readframes(frame_size)
            if len(chunk) < 2 * frame_size:  # End of file or incomplete chunk
                break
                
            # Check if chunk contains voice
            is_speech = self.vad.is_speech(chunk, wf.getframerate())
            if is_speech:
                voice_chunks.append(chunk)
        
        wf.close()
        
        # Combine voice chunks into a single audio stream
        if voice_chunks:
            voice_audio = b''.join(voice_chunks)
            # Convert bytes to audio samples
            voice_samples = np.array(struct.unpack(f'{len(voice_audio)//2}h', voice_audio), dtype=np.float32) / 32768.0
            return voice_samples
        else:
            return np.array([])
    
    def extract_audio_features(self, audio_data, sr):
        """Extract MFCC features from audio"""
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio_data,
            sr=sr,
            n_mfcc=40,
            hop_length=512,
            n_fft=2048
        )
        
        # Add delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Combine features
        audio_features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs], axis=0)
        
        # Transpose to get time as first dimension
        audio_features = audio_features.T
        
        return audio_features
    
    def process_video(self, video_path, label=None):
        """Process video file and extract features"""
        video_name = os.path.basename(video_path).split('.')[0]
        print(f"Processing video: {video_name}")
        
        # Extract frames
        frames, timestamps, duration = self.extract_frames(video_path)
        print(f"Extracted {len(frames)} frames")
        
        # Process frames
        face_frames = []
        landmarks_list = []
        valid_frames = []
        
        for i, frame in enumerate(frames):
            face_crop, landmarks = self.detect_and_crop_face(frame)
            if face_crop is not None and landmarks is not None:
                face_frames.append(face_crop)
                landmarks_list.append(landmarks)
                valid_frames.append(i)
        
        print(f"Detected faces in {len(face_frames)} frames")
        
        if len(face_frames) == 0:
            print(f"No faces detected in {video_name}")
            return None
        
        # Extract ResNet features for each face frame
        resnet_features = []
        
        # Process in sliding windows of 16 frames with overlap
        window_size = 16
        stride = 8
        
        for i in range(0, len(face_frames) - window_size + 1, stride):
            window_frames = face_frames[i:i+window_size]
            features = self.extract_resnet_features(window_frames)
            resnet_features.append(features)
        
        print(f"Extracted ResNet features for {len(resnet_features)} windows")
        
        # Extract style features if we have enough frames
        style_features = None
        if len(resnet_features) >= 2 and self.style_gru is not None:
            # Ensure we have the right sequence length for StyleGRU
            if len(resnet_features) > self.sequence_length:
                resnet_features = resnet_features[:self.sequence_length]
            elif len(resnet_features) < self.sequence_length:
                # Pad with zeros
                padding = [np.zeros_like(resnet_features[0]) for _ in range(self.sequence_length - len(resnet_features))]
                resnet_features.extend(padding)
            
            style_features = self.extract_style_features(resnet_features)
            print("Extracted style features")
        
        # Extract audio and process
        audio_path = self.extract_audio(video_path)
        audio_data, sr = librosa.load(audio_path, sr=self.audio_sample_rate)
        
        # Reduce noise
        audio_data = self.reduce_noise(audio_data, sr)
        
        # Detect voice activity
        voice_data = self.detect_voice_activity(audio_path)
        if len(voice_data) > 0:
            # Extract audio features
            audio_features = self.extract_audio_features(voice_data, sr)
            print(f"Extracted audio features: {audio_features.shape}")
        else:
            audio_features = np.array([])
            print("No voice activity detected")
        
        # Clean up temporary audio file
        if os.path.exists(audio_path) and audio_path != video_path:
            os.remove(audio_path)
        
        # Save features
        output = {
            'video_name': video_name,
            'duration': duration,
            'timestamps': [timestamps[i] for i in valid_frames],
            'landmarks': landmarks_list,
            'resnet_features': resnet_features,
            'style_features': style_features,
            'audio_features': audio_features,
            'label': label
        }
        
        # Save features
        output_path = os.path.join(self.output_dir, 'combined_features', f"{video_name}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(output, f)
        
        return output
    
    def process_dataset(self, video_dir, label_map=None):
        """Process all videos in a directory"""
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        results = []
        
        for video_file in tqdm(video_files, desc="Processing videos"):
            video_path = os.path.join(video_dir, video_file)
            
            # Determine label if label_map is provided
            label = None
            if label_map is not None:
                for key, value in label_map.items():
                    if key in video_file:
                        label = value
                        break
            
            result = self.process_video(video_path, label)
            if result is not None:
                results.append(result)
        
        return results
    
    def prepare_for_stylegru(self, features_list, output_path):
        """Prepare features for StyleGRU training"""
        real_samples = []
        fake_samples = []
        
        for features in features_list:
            if features['label'] == 0:  # Real
                real_samples.append(features)
            elif features['label'] == 1:  # Fake
                fake_samples.append(features)
        
        # Save prepared data
        prepared_data = {
            'real_samples': real_samples,
            'fake_samples': fake_samples
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(prepared_data, f)
        
        print(f"Prepared {len(real_samples)} real samples and {len(fake_samples)} fake samples")
        return prepared_data
    
    def create_triplets_for_training(self, prepared_data, output_path, num_triplets=10000):
        """Create triplets for contrastive learning"""
        real_samples = prepared_data['real_samples']
        fake_samples = prepared_data['fake_samples']
        
        triplets = []
        
        # Ensure we have enough samples
        if len(real_samples) < 2 or len(fake_samples) < 1:
            print("Not enough samples to create triplets")
            return []
        
        for _ in tqdm(range(num_triplets), desc="Creating triplets"):
            # Randomly select two real samples and one fake sample
            pos_anchor_idx = np.random.randint(0, len(real_samples))
            pos_idx = np.random.randint(0, len(real_samples))
            neg_idx = np.random.randint(0, len(fake_samples))
            
            # Ensure positive anchor and positive are different samples
            while pos_anchor_idx == pos_idx and len(real_samples) > 1:
                pos_idx = np.random.randint(0, len(real_samples))
            
            pos_anchor = real_samples[pos_anchor_idx]
            pos = real_samples[pos_idx]
            neg = fake_samples[neg_idx]
            
            # Create triplet
            triplet = {
                'pos_anchor': pos_anchor['resnet_features'],
                'pos': pos['resnet_features'],
                'neg': neg['resnet_features'],
                'labels': [0, 0, 1]  # [pos_anchor_label, pos_label, neg_label]
            }
            
            triplets.append(triplet)
        
        # Save triplets
        with open(output_path, 'wb') as f:
            pickle.dump(triplets, f)
        
        print(f"Created {len(triplets)} triplets for training")
        return triplets


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


def calculate_accuracy(predictions, targets):
    """Calculate accuracy for binary classification"""
    pred_labels = (predictions > 0.5).float()
    correct = (pred_labels == targets).float().sum()
    return correct / targets.size(0)


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """Train the deepfake detection model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for style_features, content_features, audio_features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            # Move data to device
            style_features = style_features.to(device)
            content_features = content_features.to(device)
            audio_features = audio_features.to(device) if audio_features is not None else None
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            final_pred, video_pred, audio_pred = model(style_features, content_features, audio_features)
            
            # Calculate losses
            final_loss = criterion(final_pred, labels)
            video_loss = criterion(video_pred, labels)
            
            if audio_pred is not None:
                audio_loss = criterion(audio_pred, labels)
                total_loss = final_loss + 0.5 * (video_loss + audio_loss)
            else:
                total_loss = final_loss + 0.5 * video_loss
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            # Update statistics
            train_loss += total_loss.item()
            train_acc += calculate_accuracy(final_pred, labels)
        
        # Calculate average training metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for style_features, content_features, audio_features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Move data to device
                style_features = style_features.to(device)
                content_features = content_features.to(device)
                audio_features = audio_features.to(device) if audio_features is not None else None
                labels = labels.to(device)
                
                # Forward pass
                final_pred, video_pred, audio_pred = model(style_features, content_features, audio_features)
                
                # Calculate losses
                final_loss = criterion(final_pred, labels)
                video_loss = criterion(video_pred, labels)
                
                if audio_pred is not None:
                    audio_loss = criterion(audio_pred, labels)
                    total_loss = final_loss + 0.5 * (video_loss + audio_loss)
                else:
                    total_loss = final_loss + 0.5 * video_loss
                
                # Update statistics
                val_loss += total_loss.item()
                val_acc += calculate_accuracy(final_pred, labels)
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    return model


def evaluate_model(model, test_loader):
    """Evaluate the deepfake detection model on test data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for style_features, content_features, audio_features, labels in tqdm(test_loader, desc="Testing"):
            # Move data to device
            style_features = style_features.to(device)
            content_features = content_features.to(device)
            audio_features = audio_features.to(device) if audio_features is not None else None
            labels = labels.to(device)
            
            # Forward pass
            final_pred, _, _ = model(style_features, content_features, audio_features)
            
            # Store predictions and labels
            all_preds.extend(final_pred.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = np.mean((all_preds > 0.5) == all_labels)
    
    # Calculate AUC-ROC if sklearn is available
    try:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
        
        auc_roc = roc_auc_score(all_labels, all_preds)
        average_precision = average_precision_score(all_labels, all_preds)
        
        # Calculate precision, recall, and thresholds
        precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
        
        # Find the threshold that gives the best F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"Average Precision: {average_precision:.4f}")
        print(f"Best Threshold: {best_threshold:.4f}")
        
        return {
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'average_precision': average_precision,
            'best_threshold': best_threshold
        }
    
    except ImportError:
        print(f"Test Accuracy: {accuracy:.4f}")
        return {'accuracy': accuracy}


def merge_video_predictions(predictions, sample_to_video, video_counts):
    """Merge predictions from multiple clips of the same video"""
    video_predictions = {}
    
    for pred, video_id in zip(predictions, sample_to_video):
        if video_id not in video_predictions:
            video_predictions[video_id] = []
        
        video_predictions[video_id].append(pred)
    
    # Average predictions for each video
    final_predictions = {}
    for video_id, preds in video_predictions.items():
        final_predictions[video_id] = sum(preds) / len(preds)
    
    return final_predictions


def main():
    # Example usage
    processor = VideoAudioProcessor(
        output_dir='processed_data',
        frame_rate=1,
        face_confidence=0.5,
        audio_sample_rate=16000,
        style_gru_model_path='path/to/pretrained/stylegru.pth',
        style_feature_size=9216,
        sequence_length=32
    )
    
    # Process a dataset
    label_map = {
        'real': 0,
        'fake': 1,
        'deepfake': 1,
        'faceswap': 1
    }
    
    # Process dataset and prepare features
    results = processor.process_dataset('path/to/videos', label_map)
    prepared_data = processor.prepare_for_stylegru(results, 'processed_data/prepared_data.pkl')
    
    # Create model
    model = MultimodalDeepfakeDetector(
        style_dim=8192,
        content_dim=512,
        fusion_dim=1024,
        audio_dim=120,
        transformer_dim=512,
        num_heads=8,
        num_layers=4,
        sequence_length=16
    )
    
    # Create data loaders (this would need to be implemented based on your data format)
    # train_loader, val_loader, test_loader = create_data_loaders(prepared_data)
    
    # Train model
    # trained_model = train_model(model, train_loader, val_loader, num_epochs=10)
    
    # Evaluate model
    # metrics = evaluate_model(trained_model, test_loader)
    
    # Make prediction on a single video
    # result = model.predict_video('path/to/test_video.mp4', processor)
    # print(f"Prediction: {result['final_prediction']:.4f}, Is Fake: {result['is_fake']}")
    
    print("Processing complete!")


if __name__ == "__main__":
    main()

