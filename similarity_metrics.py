"""
Utilities for model weight loading, vocabulary handling, and similarity calculation.
"""

import torch
import torch.nn.functional as F
import os
import collections
import re
import json
import base64
from safetensors.torch import load_file
import numpy as np
from scipy.optimize import linear_sum_assignment


def load_all_weights_from_dir(directory_path):
    """
    Loads all .safetensors and .bin files from a specified directory and merges them into a single state_dict.
    Prefers .safetensors files if both types are present.
    """
    print(f"\nLoading all weights from directory: {directory_path}")
    merged_state_dict = {}
    
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found {directory_path}")
        return None

    safetensors_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.safetensors')])
    bin_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.bin')])
    
    if safetensors_files:
        files = safetensors_files
        print(f"Found {len(safetensors_files)} .safetensors files, loading these preferentially.")
        if bin_files:
            print(f"Ignoring {len(bin_files)} .bin files.")
    elif bin_files:
        files = bin_files
        print(f"Found {len(bin_files)} .bin files.")
    else:
        print(f"Warning: No .safetensors or .bin files found in {directory_path}")
        return None

    for filename in files:
        file_path = os.path.join(directory_path, filename)
        try:
            print(f"  Loading: {filename}")
            if filename.endswith('.safetensors'):
                state_dict = load_file(file_path, device='cpu')
            elif filename.endswith('.bin'):
                state_dict = torch.load(file_path, map_location='cpu')
            merged_state_dict.update(state_dict)
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
            
    print(f"Successfully loaded and merged {len(files)} files.")
    return merged_state_dict


def get_word_embedding_weight(state_dict):
    """
    Extracts word embedding weights from a state_dict.
    """
    embedding_keys = [
        'model.decoder.embed_tokens.weight',
        'gpt_neox.embed_in.weight',
        'model.embed_tokens.weight',
        'transformer.embedding.word_embeddings.weight',
        'transformer.word_embeddings.weight',
        'word_embeddings.weight',
        'decoder.embed_tokens.weight',
        'transformer.wte.weight',
        'wte.weight',
        'language_model.model.embed_tokens.weight',
        'llm.model.embed_tokens.weight',
        'model.transformer.wte.weight',
        'internlm_model.model.embed_tokens.weight'
    ]
    
    for key, tensor in state_dict.items():
        if key in embedding_keys:
            print(f"Found word embedding weights: {key}")
            return tensor
    
    print("Warning: Could not find recognizable word embedding weights in the state_dict.")
    return None


def load_vocab_from_dir(directory_path):
    """
    Loads a vocabulary (token -> id mapping) from a directory.
    Supports tokenizer.json, tokenizer.model, and qwen.tiktoken files.
    """
    json_path = os.path.join(directory_path, 'tokenizer.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
            vocab = tokenizer_data.get('model', {}).get('vocab')
            if vocab and isinstance(vocab, dict):
                print(f"Successfully loaded a vocabulary of {len(vocab)} tokens from {json_path}.")
                return vocab
            else:
                print(f"Error: Could not extract vocab from {json_path} or format is incorrect.")
        except Exception as e:
            print(f"Error loading or parsing {json_path}: {e}")
    
    model_path = os.path.join(directory_path, 'tokenizer.model')
    if os.path.exists(model_path):
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(model_path)
            vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
            print(f"Successfully loaded a vocabulary of {len(vocab)} tokens from {model_path}.")
            return vocab
        except ImportError:
            print("Error: The 'sentencepiece' library is required to handle tokenizer.model files.")
            print("Please run: pip install sentencepiece")
        except Exception as e:
            print(f"Error loading or parsing {model_path}: {e}")

    ice_model_path = os.path.join(directory_path, 'ice_text.model')
    if os.path.exists(ice_model_path):
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(ice_model_path)
            vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
            print(f"Successfully loaded a vocabulary of {len(vocab)} tokens from {ice_model_path}.")
            return vocab
        except ImportError:
            print("Error: The 'sentencepiece' library is required to handle .model files.")
            print("Please run: pip install sentencepiece")
        except Exception as e:
            print(f"Error loading or parsing {ice_model_path}: {e}")

    tiktoken_path = os.path.join(directory_path, 'qwen.tiktoken')
    if os.path.exists(tiktoken_path):
        try:
            vocab = {}
            with open(tiktoken_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        token_b64, token_id_str = parts
                        try:
                            token_bytes = base64.b64decode(token_b64)
                            token_str = token_bytes.decode('utf-8', errors='replace')
                            vocab[token_str] = int(token_id_str)
                        except (ValueError, TypeError):
                            continue
            if vocab:
                print(f"Successfully loaded a vocabulary of {len(vocab)} tokens from {tiktoken_path}.")
                return vocab
            else:
                print(f"Error: {tiktoken_path} is empty or has an incorrect format.")
        except Exception as e:
            print(f"Error loading or parsing {tiktoken_path}: {e}")

    print(f"Warning: Could not find tokenizer.json, tokenizer.model, or qwen.tiktoken in {directory_path}")
    return None


def find_overlapping_vocab(vocab1, vocab2):
    """
    Finds overlapping tokens between two vocabularies (token -> id dicts) and returns their indices.
    """
    if not vocab1 or not vocab2:
        return [], [], []

    tokens1 = set(vocab1.keys())
    tokens2 = set(vocab2.keys())
    
    overlapping_tokens = sorted(list(tokens1.intersection(tokens2)))
    
    indices1 = [vocab1[token] for token in overlapping_tokens]
    indices2 = [vocab2[token] for token in overlapping_tokens]
    
    print(f"Found {len(overlapping_tokens)} overlapping tokens.")
    
    return overlapping_tokens, indices1, indices2

def generate_negative_sample_embedding(reference_embedding):
    """
    Generates a set of random embeddings as a negative sample based on the statistics of a reference embedding.
    """
    print("\n--- Generating random negative sample for embeddings ---")
    if reference_embedding is None:
        print("Error: Reference embedding is None, cannot generate negative sample.")
        return None
    
    mean = reference_embedding.mean()
    std = reference_embedding.std()
    shape = reference_embedding.shape
    random_tensor = torch.normal(mean.item(), std.item(), size=shape, dtype=reference_embedding.dtype)
    
    print("Negative sample embedding generated.")
    return random_tensor

def get_attention_weights(state_dict):
    """
    Extracts q, k, v, o attention weights from the state_dict.
    Returns a dictionary of weights organized by layer number.
    """
    layer_weights = collections.defaultdict(dict)

    for key, tensor in state_dict.items():
        layer_num = -1
        
        # Match layer number for LLaMA-like, Qwen-like, MPT-like etc. models
        # e.g., model.layers.0.self_attn... or transformer.blocks.0.attn...
        match = re.search(r'\.(layers|h|blocks)\.(\d+)\.', key)
        if match:
            layer_num = int(match.group(2))
        else:
            continue

        # Case 1: LLaMA-style / Qwen2-style (q,k,v,o are separate)
        if 'self_attn.q_proj.weight' in key and 'model.layers' in key:
            layer_weights[layer_num]['q'] = tensor
        elif 'self_attn.k_proj.weight' in key and 'model.layers' in key:
            layer_weights[layer_num]['k'] = tensor
        elif 'self_attn.v_proj.weight' in key and 'model.layers' in key:
            layer_weights[layer_num]['v'] = tensor
        elif 'self_attn.o_proj.weight' in key and 'model.layers' in key:
            layer_weights[layer_num]['o'] = tensor
     
        # Case 2: Older Qwen-style, MPT-style (q,k,v are packed)
        elif 'attn.c_attn.weight' in key or 'attn.Wqkv.weight' in key or 'W_pack.weight' in key:
            qkv = tensor
            if qkv.dim() == 2 and qkv.shape[1] == 3 * qkv.shape[0]: # (hidden_size, 3 * hidden_size)
                hidden_size = qkv.shape[0]
                q, k, v = torch.split(qkv.T, hidden_size, dim=0)
            elif qkv.dim() == 2 and qkv.shape[0] == 3 * qkv.shape[1]: # (3 * hidden_size, hidden_size)
                hidden_size = qkv.shape[1]
                q, k, v = torch.split(qkv, hidden_size, dim=0)
            else:
                 print(f"Warning: Could not split q,k,v from tensor in layer {layer_num} with shape {qkv.shape}.")
                 continue
            layer_weights[layer_num]['q'] = q
            layer_weights[layer_num]['k'] = k
            layer_weights[layer_num]['v'] = v
        elif 'attn.c_proj.weight' in key or 'attn.out_proj.weight' in key or 'dense.weight' in key and 'attention' in key:
            layer_weights[layer_num]['o'] = tensor

        # Other packed formats
        elif 'attention.wqkv.weight' in key:
            qkv = tensor
            len_mini_kv=(qkv.shape[0]-qkv.shape[1])//2
            mini_k = qkv[qkv.shape[1]:qkv.shape[1]+len_mini_kv]
            mini_v = qkv[qkv.shape[1]+len_mini_kv:]
            repeat_times=1
            q = qkv[:qkv.shape[1]]
            k = torch.cat([mini_k for i in range(repeat_times)],dim=0)
            v = torch.cat([mini_v for i in range(repeat_times)],dim=0)
            layer_weights[layer_num]['q'] = q
            layer_weights[layer_num]['k'] = k
            layer_weights[layer_num]['v'] = v
        elif 'attention.wo.weight' in key:
            layer_weights[layer_num]['o'] = tensor
        elif (('query_key_value.weight' in key) or ('att_proj.weight' in key)):
            qkv = tensor
            if qkv.dim() == 2 and qkv.shape[1] == 3 * qkv.shape[0]:
                hidden_size = qkv.shape[0]
                q, k, v = torch.split(qkv.T, hidden_size, dim=0)
            elif qkv.dim() == 2 and qkv.shape[0] == 3 * qkv.shape[1]:
                hidden_size = qkv.shape[1]
                q, k, v = torch.split(qkv, hidden_size, dim=0)
            else:
                print(f"Warning: Could not split q,k,v from tensor in layer {layer_num} with shape {qkv.shape}.")
                continue
            layer_weights[layer_num]['q'] = q
            layer_weights[layer_num]['k'] = k
            layer_weights[layer_num]['v'] = v
        elif 'attn_out.weight' in key:
            layer_weights[layer_num]['o'] = tensor

    if not layer_weights:
        print("Warning: No recognizable attention weights found in the state_dict.")
    
    return layer_weights

def generate_negative_sample(reference_weights):
    """
    Generates a set of random weights based on the statistics of reference weights.
    """
    print("\n--- Generating random negative sample based on reference weights ---")
    negative_sample_weights = collections.defaultdict(dict)
    for layer_num, layer_data in reference_weights.items():
        for weight_key in ['q', 'k', 'v', 'o']:
            if weight_key in layer_data:
                tensor = layer_data[weight_key]
                mean = tensor.mean()
                std = tensor.std()
                shape = tensor.shape
                random_tensor = torch.normal(mean.item(), std.item(), size=shape, dtype=tensor.dtype)
                negative_sample_weights[layer_num][weight_key] = random_tensor
    print("Negative sample weights generated.")
    return negative_sample_weights

def direct_layer_matcher(weights1, weights2, metric_calculator):
    """
    Performs direct layer matching (i vs i) for common layers.
    """
    common_layers = sorted(list(set(weights1.keys()) & set(weights2.keys())))
    
    direct_similarities = {}
    print(f"  Performing direct matching on {len(common_layers)} common layers.")

    for l_num in common_layers:
        layer1_data = weights1[l_num]
        layer2_data = weights2[l_num]
        
        try:
            sim = metric_calculator(layer1_data, layer2_data)
            if sim is not None:
                direct_similarities[l_num] = sim
        except Exception as e:
            print(f"Error calculating similarity for layer {l_num}: {e}")
            continue
            
    return direct_similarities

# --- CKA Similarity Calculation Functions ---

def _gram_linear(x):
    """Computes the Gram matrix for a linear kernel."""
    return x @ x.T

def _gram_rbf(x, sigma=None):
    """
    Computes the Gram matrix for an RBF kernel.
    
    Args:
        x: A feature matrix of shape (n, d).
        sigma: Bandwidth for the RBF kernel. If None, it's estimated as the median of pairwise distances.
        
    Returns:
        A Gram matrix of shape (n, n).
    """
    if sigma is None:
        sq_dists = torch.pdist(x, p=2).pow(2)
        if sq_dists.numel() > 0:
            sigma = torch.sqrt(torch.median(sq_dists))
        else:
            sigma = torch.tensor(1.0, device=x.device)

    sum_sq = torch.sum(x * x, dim=1, keepdim=True)
    sq_dists = torch.clamp(sum_sq + sum_sq.T - 2 * (x @ x.T), min=0)
    
    gamma = 1.0 / (2 * sigma**2)
    return torch.exp(-gamma * sq_dists)

def _center_gram(gram: torch.Tensor, unbiased: bool = False) -> torch.Tensor:
    """
    Centers a symmetric Gram matrix.
    
    Args:
        gram: A symmetric Gram matrix of shape (n, n).
        unbiased: Whether to use the unbiased U-statistic formula.
    
    Returns:
        A centered symmetric matrix.
    """
    if not torch.allclose(gram, gram.t()):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.clone()
    n = gram.shape[0]
    device = gram.device
    dtype = gram.dtype

    if unbiased:
        idx = torch.arange(n, device=device)
        gram[idx, idx] = 0
        means = gram.sum(dim=0, dtype=torch.float64) / (n - 2)
        means = means - means.sum() / (2 * (n - 1))
        means = means.to(dtype=dtype, device=device)
        gram = gram - means.unsqueeze(1) - means.unsqueeze(0)
        gram[idx, idx] = 0
    else:
        means = gram.mean(dim=0, dtype=torch.float64)
        means = means - means.mean() / 2
        means = means.to(dtype=dtype, device=device)
        gram = gram - means.unsqueeze(1) - means.unsqueeze(0)
    
    return gram

def cka_from_features(X, Y, kernel="linear", sigma=None, unbiased=True, device='cpu'):
    """
    Computes the CKA score between two feature representations X and Y.
    
    Args:
        X: Feature matrix 1, shape=(n, d1)
        Y: Feature matrix 2, shape=(n, d2)
        kernel: Kernel type, 'linear' or 'rbf'.
        sigma: Bandwidth for RBF kernel. If None, it's auto-estimated.
        unbiased: Whether to use the unbiased estimator.
        device: 'cpu' or 'cuda'.
        
    Returns:
        CKA score (float).
    """
    X = X.to(device)
    Y = Y.to(device)
    
    if kernel.lower() == 'linear':
        K = _gram_linear(X)
        L = _gram_linear(Y)
    elif kernel.lower() == 'rbf':
        K = _gram_rbf(X, sigma=sigma)
        L = _gram_rbf(Y, sigma=sigma)
    else:
        raise ValueError(f"Unknown kernel: {kernel}. Must be 'linear' or 'rbf'.")

    K_c = _center_gram(K, unbiased=unbiased)
    L_c = _center_gram(L, unbiased=unbiased)

    hsic = torch.sum(K_c * L_c)
    var1 = torch.sum(K_c * K_c)
    var2 = torch.sum(L_c * L_c)
    
    # Avoid division by zero
    if var1 < 1e-8 or var2 < 1e-8:
        return 0.0

    cka = hsic / torch.sqrt(var1 * var2)
    return cka.item()


def calculate_attention_cka_similarities(layer_weights1, layer_weights2, device='cpu', 
                                         subselect_indices=None, subselect_signs=None, 
                                         base_model_is_first=None):
    """
    Calculates CKA similarity for Q and K weights.
    Uses a unified alignment logic for pruning and permutation.
    Uses LAP for layer matching when layer counts differ.
    """
    def attention_cka_metric_calculator(layer1_data, layer2_data):
        scores = {}
        if 'q' in layer1_data and 'k' in layer1_data and 'q' in layer2_data and 'k' in layer2_data:
            q1, k1 = layer1_data['q'].to(torch.float32).to(device), layer1_data['k'].to(torch.float32).to(device)
            q2, k2 = layer2_data['q'].to(torch.float32).to(device), layer2_data['k'].to(torch.float32).to(device)

            # --- Unified Alignment Logic ---
            if subselect_indices is not None and base_model_is_first is not None:
                indices_t = torch.tensor(subselect_indices, dtype=torch.long, device=device)
                
                def apply_signs(weight_matrix, signs_vec):
                    if signs_vec is not None:
                        s_tensor = torch.tensor(signs_vec, device=device, dtype=torch.float32)
                        return weight_matrix @ torch.diag(s_tensor)
                    return weight_matrix

                if base_model_is_first:
                    base_q, base_k, target_q, target_k = q1, k1, q2, k2
                else:
                    base_q, base_k, target_q, target_k = q2, k2, q1, k1
                
                # Use one-hot selection matrix for GPU acceleration
                selection_matrix = F.one_hot(indices_t, num_classes=base_q.shape[1]).to(base_q.dtype).T

                base_q_sub = base_q @ selection_matrix
                base_k_sub = base_k @ selection_matrix
                
                base_q_final = apply_signs(base_q_sub, subselect_signs)
                base_k_final = apply_signs(base_k_sub, subselect_signs)

                # CKA input is always (aligned base model, target model)
                scores['Wq_weights'] = cka_from_features(base_q_final.T, target_q.T, device=device)
                scores['Wk_weights'] = cka_from_features(base_k_final.T, target_k.T, device=device)
            
            # --- Direct comparison as fallback ---
            else:
                if q1.shape == q2.shape: # Only meaningful if shapes match
                    scores['Wq_weights'] = cka_from_features(q1.T, q2.T, device=device)
                    scores['Wk_weights'] = cka_from_features(k1.T, k2.T, device=device)

            
            return scores if scores else None
        return None

    layers1 = sorted(layer_weights1.keys())
    layers2 = sorted(layer_weights2.keys())
    layer_scores_dict = {}

    # Use LAP for layer matching only if layer counts differ and both models have layers
    if len(layers1) != len(layers2) and layers1 and layers2:
        strategy_name = "LAP Layer Matching"
        print(f"\n--- Model layer counts differ ({len(layers1)} vs {len(layers2)}), using {strategy_name} ---")
        
        num_l1, num_l2 = len(layers1), len(layers2)
        cost_matrix = np.zeros((num_l1, num_l2))
        all_scores_cache = {}

        print("  Calculating all layer-pair QK CKA similarities for matching...")
        for i, l1 in enumerate(layers1):
            for j, l2 in enumerate(layers2):
                scores = attention_cka_metric_calculator(layer_weights1[l1], layer_weights2[l2])
                all_scores_cache[(l1, l2)] = scores
                
                sims = []
                if scores:
                    q_sim = scores.get('Wq_weights')
                    k_sim = scores.get('Wk_weights')
                    if q_sim is not None: sims.append(q_sim)
                    if k_sim is not None: sims.append(k_sim)
                
                if sims:
                    avg_sim = sum(sims) / len(sims)
                    cost_matrix[i, j] = -avg_sim # LAP minimizes cost, so negate similarity
        
        print("  Performing LAP layer matching based on average QK CKA similarity...")
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        print("  --- LAP Layer Matching Results ---")
        for r, c in zip(row_ind, col_ind):
            l1, l2 = layers1[r], layers2[c]
            layer_scores_dict[l1] = all_scores_cache[(l1, l2)] # Use layer numbers from model 1
            sim_score = -cost_matrix[r, c]
            print(f"    Model1-L{l1} <-> Model2-L{l2} (Avg. Similarity: {sim_score:.4f})")
    
    else: # Same number of layers or cannot align, use direct matching
        layer_scores_dict = direct_layer_matcher(layer_weights1, layer_weights2, attention_cka_metric_calculator)

    if not layer_scores_dict:
        return {}
    
    # Restructure results, grouping by metric
    results = { 'averages': {}, 'layers': {} }
    all_metrics = [f'{key}_{suffix}' for key in ['Wq','Wk'] for suffix in ['weights']]
    
    valid_metrics = [k for k in all_metrics if any(
        res and res.get(k) is not None for res in layer_scores_dict.values()
    )]

    for metric in valid_metrics:
        metric_layer_scores = {
            layer_num: scores[metric] 
            for layer_num, scores in layer_scores_dict.items() 
            if scores and scores.get(metric) is not None
        }
        
        if metric_layer_scores:
            results['layers'][metric] = metric_layer_scores
            avg_score = sum(metric_layer_scores.values()) / len(metric_layer_scores)
            results['averages'][metric] = avg_score

    # Calculate total averages
    for suffix in ['weights']:
        avg_scores = [v for k, v in results['averages'].items() if k.endswith(suffix)]
        if avg_scores:
            total_avg = sum(avg_scores) / len(avg_scores)
            avg_key = f'Wq_Wk_{suffix}'
            results['averages'][avg_key] = total_avg

    results['layers'] = {k: v for k, v in results['layers'].items() if v}
    
    return results 