import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-chinese"):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)


    # Fine-tuning control: freeze by default, then optionally unfreeze only the top layers.
    def freeze_backbone(self):
        """Freeze all pretrained text-encoder parameters."""
        for p in self.model.parameters():
            p.requires_grad_(False)

    def unfreeze_top_k_layers(self, k=2, unfreeze_pooler=True):
        """
        Freeze the full backbone, then unfreeze only the last k encoder layers
        and optionally the pooler for low-learning-rate second-stage tuning.
        """
        # Ensure no stale trainable parameters are left from a previous phase.
        self.freeze_backbone()

        if k <= 0:
            return

        if not (hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer")):
            raise ValueError("The current model has no recognizable encoder.layer structure.")

        layers = self.model.encoder.layer
        total_layers = len(layers)
        k = min(k, total_layers)

        for layer in layers[total_layers - k:]:
            for p in layer.parameters():
                p.requires_grad_(True)

        if unfreeze_pooler and hasattr(self.model, "pooler") and self.model.pooler is not None:
            for p in self.model.pooler.parameters():
                p.requires_grad_(True)
    # ===== [MODIFIED END] =====
        
    # Tokenization is handled by the caller; forward only encodes token IDs.
    def forward(self, input_ids, attention_mask):
        
        outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

        # Sequence-level embeddings.
        last_hidden_state = outputs.last_hidden_state  
        # shape: [B, Lt, D]

        return last_hidden_state, attention_mask

''' Example smoke test.
# ===== test =====
if __name__ == "__main__":
    encoder = TextEncoder()
    texts = [
        "我今天真的很开心",
        "不过下午有点焦虑"
    ]
    T, mask = encoder(texts)

    print("shape:", T.shape)   # [B, Lt, D]
    '''
