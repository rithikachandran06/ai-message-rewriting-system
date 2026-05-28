"""
Model Manager for saving and loading trained models
Handles model persistence and versioning
"""

import torch
import os
import json
import shutil
from datetime import datetime
from typing import Dict, Optional, List
import pickle

class ModelManager:
    """
    Manages model saving, loading, and versioning
    """
    def __init__(self, base_path: str = "./models"):
        """
        Initialize model manager
        
        Args:
            base_path: Base directory for model storage
        """
        self.base_path = base_path
        self.ensure_directory_exists()
    
    def ensure_directory_exists(self):
        """Ensure the model directory exists"""
        os.makedirs(self.base_path, exist_ok=True)
    
    def save_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        training_stats: Dict,
        model_name: str = "text_rewriter",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save model with metadata
        
        Args:
            model: The model to save
            optimizer: The optimizer state
            training_stats: Training statistics
            model_name: Name for the model
            metadata: Additional metadata
            
        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.base_path, f"{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(model_dir, "model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_stats': training_stats,
            'timestamp': timestamp
        }, model_path)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'model_name': model_name,
            'timestamp': timestamp,
            'model_path': model_path,
            'training_stats': training_stats
        })
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save training stats separately for easy access
        stats_path = os.path.join(model_dir, "training_stats.pkl")
        with open(stats_path, 'wb') as f:
            pickle.dump(training_stats, f)
        
        print(f"Model saved to: {model_dir}")
        return model_dir
    
    def load_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        model_path: str
    ) -> Dict:
        """
        Load model from path
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            model_path: Path to model file
            
        Returns:
            Dictionary with loaded data
        """
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            training_stats = checkpoint['training_stats']
            
            print(f"Model loaded from: {model_path}")
            return {
                'training_stats': training_stats,
                'timestamp': checkpoint.get('timestamp', 'unknown')
            }
        except Exception as e:
            print(f"Error loading model: {e}")
            return {}
    
    def list_models(self) -> List[Dict]:
        """
        List all available models
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        if not os.path.exists(self.base_path):
            return models
        
        for item in os.listdir(self.base_path):
            model_dir = os.path.join(self.base_path, item)
            if os.path.isdir(model_dir):
                metadata_path = os.path.join(model_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        models.append(metadata)
                    except Exception as e:
                        print(f"Error reading metadata for {item}: {e}")
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return models
    
    def get_latest_model(self, model_name: str = "text_rewriter") -> Optional[str]:
        """
        Get the path to the latest model
        
        Args:
            model_name: Name of the model to find
            
        Returns:
            Path to latest model or None
        """
        models = self.list_models()
        
        for model in models:
            if model.get('model_name') == model_name:
                return model.get('model_path')
        
        return None
    
    def delete_model(self, model_path: str) -> bool:
        """
        Delete a model
        
        Args:
            model_path: Path to model directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                print(f"Model deleted: {model_path}")
                return True
            else:
                print(f"Model not found: {model_path}")
                return False
        except Exception as e:
            print(f"Error deleting model: {e}")
            return False
    
    def cleanup_old_models(self, keep_count: int = 5) -> int:
        """
        Clean up old models, keeping only the most recent ones
        
        Args:
            keep_count: Number of recent models to keep
            
        Returns:
            Number of models deleted
        """
        models = self.list_models()
        
        if len(models) <= keep_count:
            return 0
        
        deleted_count = 0
        models_to_delete = models[keep_count:]
        
        for model in models_to_delete:
            model_path = os.path.dirname(model.get('model_path', ''))
            if self.delete_model(model_path):
                deleted_count += 1
        
        return deleted_count
    
    def get_model_info(self, model_path: str) -> Dict:
        """
        Get information about a specific model
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Dictionary with model information
        """
        metadata_path = os.path.join(model_path, "metadata.json")
        
        if not os.path.exists(metadata_path):
            return {}
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"Error reading model info: {e}")
            return {}
    
    def export_model(self, model_path: str, export_path: str) -> bool:
        """
        Export model to a different location
        
        Args:
            model_path: Source model path
            export_path: Destination path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(model_path):
                shutil.copytree(model_path, export_path)
                print(f"Model exported to: {export_path}")
                return True
            else:
                print(f"Source model not found: {model_path}")
                return False
        except Exception as e:
            print(f"Error exporting model: {e}")
            return False
    
    def create_backup(self, model_path: str) -> str:
        """
        Create a backup of a model
        
        Args:
            model_path: Path to model to backup
            
        Returns:
            Path to backup or empty string if failed
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{model_path}_backup_{timestamp}"
        
        if self.export_model(model_path, backup_path):
            return backup_path
        else:
            return ""
    
    def get_model_size(self, model_path: str) -> int:
        """
        Get the size of a model in bytes
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Size in bytes
        """
        total_size = 0
        
        if not os.path.exists(model_path):
            return 0
        
        for dirpath, dirnames, filenames in os.walk(model_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        
        return total_size
    
    def get_model_summary(self) -> str:
        """
        Get a summary of all models
        
        Returns:
            String summary
        """
        models = self.list_models()
        
        if not models:
            return "No models found."
        
        summary = f"Found {len(models)} models:\n\n"
        
        for i, model in enumerate(models, 1):
            model_path = os.path.dirname(model.get('model_path', ''))
            size = self.get_model_size(model_path)
            size_mb = size / (1024 * 1024)
            
            summary += f"{i}. {model.get('model_name', 'Unknown')}\n"
            summary += f"   Timestamp: {model.get('timestamp', 'Unknown')}\n"
            summary += f"   Size: {size_mb:.2f} MB\n"
            summary += f"   Path: {model_path}\n\n"
        
        return summary
