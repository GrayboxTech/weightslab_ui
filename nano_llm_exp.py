import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import threading
from board import Dash


from weightslab.experiment import Experiment
from weightslab.model_with_ops import NetworkWithOps
from weightslab.model_with_ops import DepType
from weightslab.modules_with_ops import Conv2dWithNeuronOps
from weightslab.modules_with_ops import LinearWithNeuronOps
from weightslab.modules_with_ops import BatchNorm2dWithNeuronOps



# Custom Multi-Head Attention Block
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        # Query, Key, Value projections
        self.query = LinearWithNeuronOps(d_model, d_model)
        self.key = LinearWithNeuronOps(d_model, d_model)
        self.value = LinearWithNeuronOps(d_model, d_model)

        # Output projection
        self.out_proj = LinearWithNeuronOps(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        # Project and reshape to [batch_size, seq_length, num_heads, head_dim]
        # import pdb; pdb.set_trace()
        q = self.query(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_length, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.out_proj(attn_output)

        return output

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Feed-forward network
        self.ff = nn.Sequential(
            LinearWithNeuronOps(d_model, d_ff),
            nn.GELU(),
            LinearWithNeuronOps(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection and layer norm
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Feed-forward with residual connection and layer norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Full Transformer Language Model
class TransformerLM(NetworkWithOps, nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6, d_ff=1024, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.fc_out = LinearWithNeuronOps(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def generate_causal_mask(self, seq_len):
        # Create a causal attention mask (lower triangular)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return ~mask  # Invert to get the causal mask (1s for positions to attend)

    def forward(self, x):
        seq_len = x.size(1)

        # Get token embeddings and add positional embeddings
        token_embeddings = self.embedding(x)
        position_embeddings = self.pos_embedding[:, :seq_len, :]
        x = self.dropout(token_embeddings + position_embeddings)

        # Generate causal mask (to prevent looking ahead)
        causal_mask = self.generate_causal_mask(seq_len).to(x.device)

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, causal_mask)

        x = self.norm(x)
        logits = self.fc_out(x)

        return logits



class TokenizedTextDataset(Dataset):
    def __init__(self, dataset, tokenizer, block_size=128):
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Process texts individually
        for text in dataset["text"]:
            if text and not text.isspace():
                # Tokenize each text separately
                tokenized = self.tokenizer(
                    text, 
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt"
                )["input_ids"][0]
                
                # Only create examples if we have enough tokens
                if len(tokenized) > 1:
                    # Create blocks from this single text
                    for i in range(0, len(tokenized) - block_size, block_size // 2):
                        if i + block_size <= len(tokenized):
                            self.examples.append(tokenized[i:i + block_size])
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        x = self.examples[idx]
        # Input: all tokens except the last one
        # Target: all tokens except the first one (shifted by one position)
        return x[:-1], x[1:]


def get_exp():
    # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load a small portion of the dataset
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train[:2%]")
    print(f"Dataset size: {len(dataset)} entries")
    
    # Initialize tokenizer (GPT-2 tokenizer)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Hyperparameters
    vocab_size = tokenizer.vocab_size
    d_model = 256       # Embedding dimension
    num_heads = 8       # Number of attention heads
    num_layers = 6      # Number of transformer layers
    d_ff = 1024         # Feed-forward dimension
    block_size = 128    # Sequence length
    batch_size = 8      # Batch size
    lr = 3e-4           # Learning rate
    training_steps = 1000  # Number of training steps
    
    # Create train and eval datasets
    train_dataset = TokenizedTextDataset(dataset, tokenizer, block_size=block_size)
    eval_dataset = TokenizedTextDataset(dataset, tokenizer, block_size=block_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True
    )
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True
    )
    
    # Initialize model
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=block_size,
        dropout=0.1
    )
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Custom loss function (Cross Entropy with reduction='none')
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Initialize optimizer
    optimizer_class = torch.optim.AdamW
    
    # Create Experiment object
    experiment = Experiment(
        model=model,
        optimizer_class=optimizer_class,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        device=device,
        learning_rate=lr,
        batch_size=batch_size,
        training_steps_to_do=training_steps,
        name="nano-llm",
        logger=Dash("nano-llm"),
        criterion=criterion
    )

    def print_prompt_results():
        def generate_text(prompt, max_length=50, temperature=1.0):
            model.eval()
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                for _ in range(max_length):
                    outputs = model(input_ids)
                    # import pdb; pdb.set_trace()
                    next_token_logits = outputs[:, -1, :] / temperature

                    # Apply softmax to get probabilities
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_token], dim=1)

                    if next_token.item() == tokenizer.eos_token_id:
                        break

                return tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
        # Test generation with a few prompts
        test_prompts = [
            "The purpose of language models is to",
            "In the field of artificial intelligence",
            "Machine learning algorithms can"
        ]
        
        print("\nText Generation Examples:")
        for prompt in test_prompts:
            generated = generate_text(prompt)
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}")
            print("-" * 50)

    # Optional: Register callbacks if needed
    def print_step_callback():
        print(f"Current step: {experiment.performed_train_steps()}")

    experiment.register_train_loop_callback(print_step_callback)
    experiment.register_train_loop_callback(print_prompt_results)

    return experiment

def main():
    experiment = get_exp()
    # import pdb; pdb.set_trace()
    # Train the model using Experiment's methods
    experiment.toggle_training_status()
    try:
        # Train for specified number of steps with periodic full evaluation
        for _ in range(2000):
            experiment.train_steps_or_eval_llms()

    except Exception as e:  
        print(f"Training interrupted: {e}")
    finally:
        # Save the final model checkpoint
        experiment.dump()
        print("Training completed. Model checkpoint saved.")

if __name__ == "__main__":
    main()