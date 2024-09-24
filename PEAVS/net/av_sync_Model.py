import torch
from torch import nn
import torch.nn.functional as F
from modules.transformer import TransformerEncoder



class SyncMetric(nn.Module):
    def __init__(self, model_args):
        """
        """
        super(SyncMetric, self).__init__()
        
        # Model Hyperparameters
        self.num_heads = model_args.num_heads
        self.layers = model_args.layers
        self.attn_mask = model_args.attn_mask
        output_dim = model_args.output_dim
        self.a_dim, self.v_dim = 128, 1024
        self.attn_dropout = model_args.attn_dropout
        self.relu_dropout = model_args.relu_dropout
        self.res_dropout = model_args.res_dropout
        self.out_dropout = model_args.out_dropout
        self.embed_dropout = model_args.embed_dropout
        self.d_v = 128
        
        combined_dim = 2*self.d_v

        # 1D convolutional projection layers
        self.conv_1d_v = nn.Conv1d(self.v_dim, self.d_v, kernel_size=1, padding=0, bias=False)

        # Self Attentions 
        self.trans_a_mem = self.transformer_arch(self_type='audio_self')
        self.trans_v_mem = self.transformer_arch(self_type='visual_self')
        
        # Cross-modal 
        self.trans_v_with_a = self.transformer_arch(self_type='visual/audio')
        self.trans_a_with_v = self.transformer_arch(self_type='audio/visual')
        
        # Linear layers
        self.proj1 = nn.Linear(combined_dim, combined_dim//2)
        self.proj2 = nn.Linear(combined_dim//2, combined_dim//4)
        self.proj3 = nn.Linear(combined_dim//4, combined_dim//4)  # adding one more hidden layer
        self.out_layer = nn.Linear(combined_dim//4, output_dim)


    def transformer_arch(self, self_type='audio/visual', layers = 3):
        if self_type == 'visual/audio':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout  
        elif self_type == 'audio/visual':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout  
        elif self_type == 'audio_self':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout    
        elif self_type == 'visual_self':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Not a valid network")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    
    
    def forward(self, x_aud, x_vid, mask_a, mask_v):
        """
        audio, and vision should have dimension [batch_size, seq_len, n_features]

        """    
        x_aud = x_aud.transpose(1, 2)
        x_vid = x_vid.transpose(1, 2)
       
        # 1-D Convolution visual/audio features
        proj_x_v = x_vid if self.v_dim == self.d_v else self.conv_1d_v(x_vid)
        proj_x_a = x_aud.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
  
        # Audio/Visual
        h_av = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v, encoder_padding_mask=mask_v)
        h_as = self.trans_a_mem(h_av, encoder_padding_mask = mask_v)
        mask_v_expanded = mask_v.unsqueeze(-1).expand(-1, -1, h_as.size(-1))  # Expand to [batch_size, seq_len, embed_dim]
        h_as_masked = h_as.permute(1, 0, 2) * mask_v_expanded  # Align mask and apply to h_as
        valid_token_counts = mask_v.sum(dim=1, keepdim=True).float()  # Get the number of valid tokens for each batch
        representation_visual = h_as_masked.sum(dim=1) / valid_token_counts

        # Visual/Audio
        h_va = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a,encoder_padding_mask = mask_a)
        h_vs = self.trans_v_mem(h_va, encoder_padding_mask = mask_a)
        seq_len_vs = h_vs.size(0)  # Get the sequence length of h_vs
        mask_a_expanded = mask_a[:, :seq_len_vs].unsqueeze(-1).expand(-1, -1, h_vs.size(-1))  # [batch_size, seq_len, embed_dim]
        h_vs_masked = h_vs.permute(1, 0, 2) * mask_a_expanded  # [batch_size, seq_len, embed_dim]
        valid_token_counts = mask_a.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
        representation_audio = h_vs_masked.sum(dim=1) / valid_token_counts  # [batch_size, embed_dim]
    
        # Concatenating audiovisual representations
        av_h_rep = torch.cat([representation_visual, representation_audio], dim=1)
        
        # Pass the representation through three fully connected layers
        av_h_rep = F.relu(self.proj1(av_h_rep))
        av_h_rep = F.relu(self.proj2(av_h_rep))
        av_h_rep = F.relu(self.proj3(av_h_rep))

        # Main network output
        output = self.out_layer(F.dropout(av_h_rep, p=self.out_dropout, training=self.training))
        
        return output