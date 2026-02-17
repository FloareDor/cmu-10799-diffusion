
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class AudioEncoder(nn.Module):
    """
    Wav2Vec2 wrapper for audio encoding.
    """
    def __init__(self, model_name="facebook/wav2vec2-base-960h", freeze=True):
        super().__init__()
        print(f"Loading Audio Encoder: {model_name}...")
        self.backbone = Wav2Vec2Model.from_pretrained(model_name)
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            
        self.output_dim = self.backbone.config.hidden_size # 768 for base

    def forward(self, audio):
        # audio: (B, T) raw waveform
        # Wav2Vec2 expects (B, T)
        
        # If frozen, run in no_grad
        # If frozen, run in no_grad
        # Check first parameter to see if gradients are required
        req_grad = next(self.backbone.parameters()).requires_grad
        
        with torch.set_grad_enabled(req_grad): 
             outputs = self.backbone(audio)
        
        # outputs.last_hidden_state: (B, SequenceLength, 768)
        # For baseline, we can Mean Pool over time to get a global audio vector
        # (B, 768)
        
        return outputs.last_hidden_state.mean(dim=1)
