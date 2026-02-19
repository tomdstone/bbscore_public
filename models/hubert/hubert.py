import numpy as np
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor


class HuBERT:
    def __init__(self):
        self.model_mappings = {
            "HuBERT-Base": "facebook/hubert-base-ls960",
            "HuBERT-Large": "facebook/hubert-large-ls960-ft",
            "HuBERT-XL": "facebook/hubert-xlarge-ls960-ft",
        }
        self.processor = None
        self.static = True

    def get_model(self, identifier):
        hf_name = self.model_mappings[identifier]
        # HuBERT uses Wav2Vec2FeatureExtractor (same architecture family)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(hf_name)
        model = HubertModel.from_pretrained(hf_name)
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"Warning: torch.compile failed for HuBERT: {e}")
        return model

    def preprocess_fn(self, input_data):
        """
        Args:
            input_data: np.ndarray of shape (num_samples,) at 16kHz
        Returns:
            Tensor of shape (1, num_samples)
        """
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.numpy()
        inputs = self.processor(
            input_data, sampling_rate=16000,
            return_tensors="pt", padding=True
        )
        return inputs.input_values

    def postprocess_fn(self, features):
        """Mean-pool encoder output (B, seq_len, hidden_dim) -> (B, hidden_dim)."""
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        if features.ndim == 3:
            return features.mean(axis=1)
        return features.reshape(features.shape[0], -1)
