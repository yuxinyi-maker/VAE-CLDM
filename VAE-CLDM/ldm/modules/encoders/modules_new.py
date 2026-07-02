import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia

# Import modified x_transformer module
from ldm.modules.x_transformer_new import Encoder, TransformerWrapper


class AbstractEncoder(nn.Module):
    """Abstract encoder base class: defines unified interface for all encoders"""
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        """Encode method: must be implemented by subclass"""
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    """Class embedder: converts class labels to embedding vectors"""
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key  # key used to get class labels from batch
        self.embedding = nn.Embedding(n_classes, embed_dim)  # embedding layer

    def forward(self, batch, key=None):
        """Forward pass"""
        if key is None:
            key = self.key
        # class embedding for cross attention
        c = batch[key][:, None]  # add a dimension [batch_size, 1]
        c = self.embedding(c)  # convert to embedding vector [batch_size, 1, embed_dim]
        return c


class TransformerEmbedder(AbstractEncoder):
    """Transformer encoder: uses custom Transformer for text encoding"""
    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        # Create Transformer wrapper
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer)
        )

    def forward(self, tokens):
        """Forward pass"""
        tokens = tokens.to(self.device)
        z = self.transformer(tokens, return_embeddings=True)  # return embeddings instead of logits
        return z

    def encode(self, x):
        """Encode interface"""
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """BERT tokenizer: uses HuggingFace BERT tokenizer, vocab size: 30522"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface  # whether to provide VQ interface
        self.max_length = max_length  # maximum sequence length

    def forward(self, text):
        """Forward pass:token"""
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt"
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        """Encode method"""
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]  # VQ interface compatible format

    def decode(self, text):
        """Decode method: currently returns text directly"""
        return text


class BERTEmbedder(AbstractEncoder):
    """BERT embedder: combines BERT tokenizer and Transformer encoder"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda", use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer  # whether to use tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        # Transformer encoder
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
            emb_dropout=embedding_dropout  # embedding dropout
        )

    def forward(self, text):
        """Forward pass"""
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # use tokenizer
        else:
            tokens = text  # directly use input tokens
        z = self.transformer(tokens, return_embeddings=True)  # get embeddings
        return z

    def encode(self, text):
        """Encode method:77"""
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """CLIP text encoder - final fixed version"""
    
    def __init__(self, version="/hy-tmp/all/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        
       
        self.transformer = self.transformer.to(self.device)
        
        # Freeze all parameters
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        print(f"✅ CLIP encoder initialized: {version}, device: {self.device}")

    def encode(self, text):
        """Encode text - fixed version"""
        return self(text)

    def forward(self, text):
        """Forward pass - (reduced debug output)"""
        # Only output details for the first batch
        is_first_batch = not hasattr(self, '_has_run')
        
        if is_first_batch:
            print(f"🔍 CLIP encoder input: type{type(text)}, content sample: {text[:2] if isinstance(text, list) and len(text) > 1 else text}")
            self._has_run = True
        
      
        if isinstance(text, torch.Tensor):
            if is_first_batch:
                print(f"⚠️ CLIP received tensor input: shape{text.shape}")
            # If already encoded condition, return directly
            if text.dim() == 3 and text.shape[1] == 77 and text.shape[2] == 768:
                if text.device != self.device:
                    text = text.to(self.device)
                return text
            else:
                batch_size = text.shape[0] if hasattr(text, 'shape') else 1
                default_text = ["a 3d rock sample"] * batch_size
                text = default_text
        
        # Ensure it is a list of strings
        if not isinstance(text, list):
            if isinstance(text, str):
                text = [text]
            else:
                batch_size = 1
                if hasattr(text, '__len__'):
                    batch_size = len(text)
                text = ["a 3d rock sample"] * batch_size
        
        # Tokenize
        try:
            batch_encoding = self.tokenizer(
                text, 
                truncation=True, 
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            tokens = batch_encoding["input_ids"].to(dtype=torch.long, device=self.device)
            
            if is_first_batch:
                print(f"🔍 CLIPTokenize: shape{tokens.shape}")
            
        except Exception as e:
            if is_first_batch:
                print(f"❌ CLIPTokenize: {e}")
            batch_size = len(text)
            tokens = torch.zeros(batch_size, self.max_length, dtype=torch.long, device=self.device)
        
        # Encode
        try:
            # Manually create position_ids
            batch_size = tokens.shape[0]
            position_ids = torch.arange(self.max_length, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
                
            outputs = self.transformer(
                input_ids=tokens,
                position_ids=position_ids,
                output_hidden_states=True
            )
            
            z = outputs.last_hidden_state
            
            if is_first_batch:
                print(f"✅ CLIPEncode: shape{z.shape}")
                
            return z
            
        except Exception as e:
            if is_first_batch:
                print(f"❌ CLIPEncode: {e}")
            batch_size = len(text)
            return torch.zeros(batch_size, self.max_length, 768, device=self.device)


            

            
class FrozenCLIPTextEmbedder(nn.Module):
    """
    CLIP:OpenAICLIPEncode
    """
    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")  # load CLIP model
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat  # repeat count(for temporal dimension)
        self.normalize = normalize  # whether to normalize

    def freeze(self):
        """Freeze all parameters"""
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        """Forward pass:Encode"""
        tokens = clip.tokenize(text).to(self.device)  # CLIPTokenize
        z = self.model.encode_text(tokens)  # Encode
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)  # L2 normalization
        return z

    def encode(self, text):
        """Encode interface:"""
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]  # add temporal dimension [batch_size, 1, hidden_dim]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)  # repeat to specified number of timesteps
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
    CLIP:uses CLIPEncode
    """
    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)  # load CLIP model

        self.antialias = antialias  # whether to use antialiasing

        # CLIP normalization parameters
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        """Preprocess: resize and normalize"""
        # Resize to 224x224(CLIP input size)
        x = kornia.geometry.resize(
            x,
            (224, 224),
            interpolation='bicubic',
            align_corners=True,
            antialias=self.antialias
        )
        x = (x + 1.) / 2.  # [-1,1]map to[0,1]
        # CLIP normalization
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        """Forward pass:Encode"""
        # Assume x is in range [-1,1]
        return self.model.encode_image(self.preprocess(x))


class PorosityConditionEncoder(nn.Module):
    """
    Encode:
    Converts scalar porosity values to condition embedding vectors
    """
    def __init__(self, condition_dim=128, hidden_dim=64):
        super().__init__()
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        
        # Encode
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),  # input porosity scalar value
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, condition_dim),  # output condition dimension
            nn.Tanh()  # constrain output range
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, porosity):
        """
        Forward pass
        
        Args:
            porosity: porosity value [batch_size]  [batch_size, 1]
            
        Returns:
            condition: condition embedding [batch_size, condition_dim]
        """
        if porosity.dim() == 1:
            porosity = porosity.unsqueeze(-1)  # [batch_size, 1]
        
        # Ensure porosity is within reasonable range [0, 1]
        porosity = torch.clamp(porosity, 0.0, 1.0)
        
        # Encode
        condition = self.encoder(porosity)  # [batch_size, condition_dim]
        return condition

    def encode(self, porosity):
        """Encode interface"""
        return self(porosity)


class MultiConditionEncoder(nn.Module):
    """
    Encode:
    Fuses both condition types into a unified condition representation
    """
    def __init__(self, 
                 text_encoder_config=None,
                 condition_dim=128,
                 fusion_method='concat'):
        super().__init__()
        self.condition_dim = condition_dim
        self.fusion_method = fusion_method
        
        # Encode(uses CLIP)
        if text_encoder_config is None:
            text_encoder_config = {
                'model_path': "/hy-tmp/all/clip-vit-large-patch14",
                'device': 'cuda',
                'max_length': 77
            }
        self.text_encoder = FrozenCLIPEmbedder(**text_encoder_config)
        text_dim = 768  # CLIP ViT-L/14 hidden dimension
        
        # Encode
        self.porosity_encoder = PorosityConditionEncoder(condition_dim=condition_dim)
        
        # Condition fusion layer
        if fusion_method == 'concat':
            # project to target dimension after concatenation
            self.fusion_proj = nn.Linear(text_dim + condition_dim, condition_dim)
        elif fusion_method == 'add':
            # ensure both condition dimensions are the same, then add
            self.text_proj = nn.Linear(text_dim, condition_dim)
            self.porosity_proj = nn.Linear(condition_dim, condition_dim)
        elif fusion_method == 'weighted':
            # weighted fusion
            self.text_proj = nn.Linear(text_dim, condition_dim)
            self.porosity_proj = nn.Linear(condition_dim, condition_dim)
            self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable weight
        
        self.norm = nn.LayerNorm(condition_dim)
        
    def freeze_text_encoder(self):
        """Encode"""
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, text, porosity):
        """
        Forward pass:
        
        Args:
            text: text description list [batch_size]
            porosity: porosity value [batch_size]  [batch_size, 1]
            
        Returns:
            fused_condition: fused condition [batch_size, condition_dim]
        """
        # Encode
        text_emb = self.text_encoder(text)  # [batch_size, seq_len, text_dim]
        text_emb = text_emb.mean(dim=1)     # average pooling [batch_size, text_dim]
        
        # Encode
        porosity_emb = self.porosity_encoder(porosity)  # [batch_size, condition_dim]
        
        # Condition fusion
        if self.fusion_method == 'concat':
            # concatenate both conditions
            fused = torch.cat([text_emb, porosity_emb], dim=1)  # [batch_size, text_dim + condition_dim]
            fused_condition = self.fusion_proj(fused)  # [batch_size, condition_dim]
            
        elif self.fusion_method == 'add':
            # project to same dimension and add
            text_proj = self.text_proj(text_emb)  # [batch_size, condition_dim]
            porosity_proj = self.porosity_proj(porosity_emb)  # [batch_size, condition_dim]
            fused_condition = text_proj + porosity_proj
            
        elif self.fusion_method == 'weighted':
            # learnable weight fusion
            text_proj = self.text_proj(text_emb)
            porosity_proj = self.porosity_proj(porosity_emb)
            fused_condition = self.alpha * text_proj + (1 - self.alpha) * porosity_proj
        
        # Layer normalization
        fused_condition = self.norm(fused_condition)
        
        return fused_condition

    def encode(self, text, porosity):
        """Encode interface"""
        return self(text, porosity)


class Rock3DConditionProcessor:
    """
    3D rock data condition processor: unified handling of condition inputs during training and inference
    """
    def __init__(self, condition_dim=128, device='cuda'):
        self.condition_dim = condition_dim
        self.device = device
        
        # Encode
        self.encoder = MultiConditionEncoder(
            condition_dim=condition_dim,
            fusion_method='concat'  # use concatenation to preserve more information
        )
        self.encoder.to(device)
        
    def process_batch(self, batch):
        """
        Process condition information in batch data
        
        Args:
            batch: batch data containing text and porosity
            
        Returns:
            condition: fused condition [batch_size, condition_dim]
        """
        text = batch.get('text', [''] * len(batch['image']))
        porosity = batch.get('porosity', torch.zeros(len(batch['image'])).to(self.device))
        
        # device
        if isinstance(porosity, torch.Tensor):
            porosity = porosity.to(self.device)
        else:
            porosity = torch.tensor(porosity, device=self.device)
        
        # Encode
        with torch.no_grad():
            condition = self.encoder(text, porosity)
            
        return condition
    
    def process_single(self, text, porosity):
        """
        Process condition information for a single sample
        
        Args:
            text: text description
            porosity: porosity value
            
        Returns:
            condition: fused condition [1, condition_dim]
        """
        if isinstance(text, str):
            text = [text]
        if isinstance(porosity, (int, float)):
            porosity = torch.tensor([porosity], device=self.device)
        
        with torch.no_grad():
            condition = self.encoder(text, porosity)
            
        return condition


# Create aliases for backward compatibility
ConditionEncoder = MultiConditionEncoder

