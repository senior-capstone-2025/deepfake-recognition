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


