from typing import Tuple, List
import sys
import torch
import torch.nn as nn
sys.path.append(r"F:\work\ieeetgrs\chromaformer\Swin-Transformer-V2")
from swin_transformer_v2.model_parts import PatchEmbedding, SwinTransformerStage

__all__: List[str] = ["ChromaFormer", "chromaformer_t", "chromaformer_s", "chromaformer_b", "chromaformer_l", "chromaformer_h"]

class SpectralDependencyModule(nn.Module):
    """
    Implements the Spectral Dependency Module (SDM) for learning multi-spectral correlations.
    """
    def __init__(self, in_channels: int, embed_dim: int):
        super(SpectralDependencyModule, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        Q = self.query(x)
        K = self.key(x).transpose(-2, -1)  
        V = self.value(x)
        
        attention = self.softmax(Q @ K / (K.shape[-1] ** 0.5))
        output = attention @ V
        return output.view(B, C, H, W)  


class ChromaFormer(nn.Module):
    """
    Implements the ChromaFormer: a Swin-based transformer with Spectral Dependency Module (SDM).
    """
    def __init__(self, 
                 in_channels: int,
                 embedding_channels: int,
                 depths: Tuple[int, ...],
                 input_resolution: Tuple[int, int],
                 number_of_heads: Tuple[int, ...],
                 num_classes: int = 10,  
                 window_size: int = 7,
                 patch_size: int = 4,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: float = 0.2,
                 use_checkpoint: bool = False,
                 sequential_self_attention: bool = False,
                 use_deformable_block: bool = False) -> None:
        super(ChromaFormer, self).__init__()
        self.patch_size: int = patch_size
        
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, out_channels=embedding_channels, patch_size=patch_size)

        patch_resolution: Tuple[int, int] = (input_resolution[0] // patch_size, input_resolution[1] // patch_size)
        
        dropout_path = torch.linspace(0., dropout_path, sum(depths)).tolist()
        
        self.stages = nn.ModuleList()
        self.spectral_module = SpectralDependencyModule(in_channels=embedding_channels, embed_dim=embedding_channels)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        num_downscales = len(depths) - 2  
        final_channels = embedding_channels * (2 ** num_downscales)
        self.fc = nn.Linear(final_channels, num_classes)
        
        for index, (depth, number_of_head) in enumerate(zip(depths, number_of_heads)):
            self.stages.append(
                SwinTransformerStage(
                    in_channels=embedding_channels * (2 ** max(index - 1, 0)),
                    depth=depth,
                    downscale=not (index == 0),
                    input_resolution=(patch_resolution[0] // (2 ** max(index - 1, 0)),
                                      patch_resolution[1] // (2 ** max(index - 1, 0))),
                    number_of_heads=number_of_head,
                    window_size=window_size,
                    ff_feature_ratio=ff_feature_ratio,
                    dropout=dropout,
                    dropout_attention=dropout_attention,
                    dropout_path=dropout_path[sum(depths[:index]):sum(depths[:index + 1])],
                    use_checkpoint=use_checkpoint,
                    sequential_self_attention=sequential_self_attention,
                    use_deformable_block=use_deformable_block and (index > 0)
                ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        x = self.patch_embedding(x)
        x = self.spectral_module(x)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.global_avg_pool(x)  
        x = x.flatten(1)                 
        x = self.fc(x)                   
        return x


def chromaformer_t(input_resolution: Tuple[int, int], **kwargs) -> ChromaFormer:
    return ChromaFormer(input_resolution=input_resolution, embedding_channels=96, depths=(2, 2, 6, 2), number_of_heads=(3, 6, 12, 24), **kwargs)

def chromaformer_s(input_resolution: Tuple[int, int], **kwargs) -> ChromaFormer:
    return ChromaFormer(input_resolution=input_resolution, embedding_channels=96, depths=(2, 2, 18, 2), number_of_heads=(3, 6, 12, 24), **kwargs)

def chromaformer_b(input_resolution: Tuple[int, int], **kwargs) -> ChromaFormer:
    return ChromaFormer(input_resolution=input_resolution, embedding_channels=128, depths=(2, 2, 18, 2), number_of_heads=(4, 8, 16, 32), **kwargs)

def chromaformer_l(input_resolution: Tuple[int, int], **kwargs) -> ChromaFormer:
    return ChromaFormer(input_resolution=input_resolution, embedding_channels=192, depths=(2, 2, 18, 2), number_of_heads=(6, 12, 24, 48), **kwargs)

def chromaformer_h(input_resolution: Tuple[int, int], **kwargs) -> ChromaFormer:
    return ChromaFormer(input_resolution=input_resolution, embedding_channels=352, depths=(2, 2, 18, 2), number_of_heads=(11, 22, 44, 88), **kwargs)
