import numpy as np
import torch
from tqdm import tqdm
from model import MultimodalDeepfakeDetector
from preprocessor import VideoAudioProcessor

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

