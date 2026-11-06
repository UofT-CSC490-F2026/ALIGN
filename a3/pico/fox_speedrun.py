"""Native July GPT with memory-efficient, differentiable FoX attention and KV caching."""
import torch
from torch import nn
from torch.nn import functional as F
from nanochat.gpt import GPT, Linear, norm
from nanochat.engine import KVCache
from torch.nn.attention.flex_attention import flex_attention

flex = torch.compile(flex_attention, dynamic=True)


def forgetting_attention(q,k,v,cumulative,window):
    # PyTorch 2.9 needs distinct captured buffers for distinct gradient indices.
    query_cumulative=cumulative.clone()
    def mod(score,b,h,i,j):
        allowed=(i>=j)
        if window>=0:
            allowed=allowed & (i-j<=window)
        return torch.where(allowed,score+query_cumulative[b,h,i]-cumulative[b,h,j],-float('inf'))
    return flex(q,k,v,score_mod=mod)


class FoxCache(KVCache):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.forget_cache=torch.zeros(self.n_layers,self.batch_size,self.max_seq_len,
                                      self.n_heads,device=self.k_cache.device,dtype=torch.float32)

    def prefill(self,other):
        super().prefill(other)
        pos=other.get_pos()
        self.forget_cache[:,:,:pos]=other.forget_cache[:,:,:pos]

    def reset(self):
        super().reset()
        self.forget_cache.zero_()


class FoxAttention(nn.Module):
    def __init__(self,source,config):
        super().__init__()
        for name in ('c_q','c_k','c_v','c_proj','ve_gate','ve_gate_channels','layer_idx','n_head','head_dim'):
            setattr(self,name,getattr(source,name))
        self.forget=Linear(config.n_embd+1,config.n_head,bias=False)

    def forward(self,x,ve,cos_sin,window_size,kv_cache=None):
        b,t,c=x.shape
        shape=(b,t,self.n_head,self.head_dim)
        q=norm(self.c_q(x).view(shape))*1.2
        k=norm(self.c_k(x).view(shape))*1.2
        v=self.c_v(x).view(shape)
        if ve is not None:
            gate=3*self.ve_gate(x[...,:self.ve_gate_channels]).sigmoid()
            v=v+gate.unsqueeze(-1)*ve.view(shape)
        gates=F.logsigmoid(self.forget(torch.cat((x,torch.ones_like(x[...,:1])),dim=-1)).float())
        cumulative=gates.cumsum(1)
        if kv_cache is None:
            y=forgetting_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),
                                  cumulative.transpose(1,2).contiguous(),window_size[0]).transpose(1,2)
        else:
            start=kv_cache.get_pos()
            stop=start+t
            if stop>kv_cache.max_seq_len:
                raise ValueError('Cache capacity exceeded')
            gc=kv_cache.forget_cache[self.layer_idx]
            if start:
                cumulative=cumulative+gc[:,start-1:start]
            kc,vc=kv_cache.get_layer_cache(self.layer_idx)
            kc[:,start:stop]=k
            vc[:,start:stop]=v
            gc[:,start:stop]=cumulative
            scores=q.transpose(1,2).float()@kc[:,:stop].transpose(1,2).float().transpose(-1,-2)
            scores=scores*self.head_dim**-.5+cumulative.transpose(1,2).unsqueeze(-1)-gc[:,:stop].transpose(1,2).unsqueeze(-2)
            delta=torch.arange(start,stop,device=x.device)[:,None]-torch.arange(stop,device=x.device)[None,:]
            mask=delta>=0
            if window_size[0]>=0:
                mask=mask & (delta<=window_size[0])
            probs=scores.masked_fill(~mask,-float('inf')).softmax(-1)
            y=(probs.to(v.dtype)@vc[:,:stop].transpose(1,2)).transpose(1,2)
            if self.layer_idx==kv_cache.n_layers-1:
                kv_cache.advance(t)
        return self.c_proj(y.contiguous().view(b,t,c))


class FoxGPT(GPT):
    def __init__(self,config):
        super().__init__(config)
        for block in self.transformer.h:
            block.attn=FoxAttention(block.attn,config)

    def init_weights(self):
        super().init_weights()
        for block in self.transformer.h:
            nn.init.zeros_(block.attn.forget.weight)
            with torch.no_grad():
                block.attn.forget.weight[:,-1].fill_(3.)


def install_cache():
    import nanochat.engine as engine
    engine.KVCache=FoxCache
