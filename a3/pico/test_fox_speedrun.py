"""Numerical gate gradients, causality, and native cached-generation correctness."""
def run():
    import torch
    from nanochat.common import compute_init
    from nanochat.gpt import GPTConfig
    from pico.fox_speedrun import FoxGPT,FoxCache,forgetting_attention,install_cache
    from pico.variants import PilotGPT
    compute_init('cuda')
    torch.set_float32_matmul_precision('highest')
    for window in (-1,16):
        tensors=[torch.randn(2,2,64,64,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
        g=torch.randn(2,2,64,device='cuda',requires_grad=True)
        cumulative=g.logsigmoid().cumsum(-1) if hasattr(g,'logsigmoid') else torch.nn.functional.logsigmoid(g).cumsum(-1)
        y=forgetting_attention(*tensors,cumulative,window)
        reference_tensors=[t.detach().clone().requires_grad_() for t in tensors]
        reference_g=g.detach().clone().requires_grad_()
        reference_cumulative=torch.nn.functional.logsigmoid(reference_g).cumsum(-1)
        q,k,v=reference_tensors
        scores=q.float()@k.float().transpose(-1,-2)/8+reference_cumulative.unsqueeze(-1)-reference_cumulative.unsqueeze(-2)
        d=torch.arange(64,device='cuda')[:,None]-torch.arange(64,device='cuda')[None,:]
        mask=(d>=0) if window<0 else ((d>=0)&(d<=window))
        ref=scores.masked_fill(~mask,-float('inf')).softmax(-1).to(v.dtype)@v
        torch.testing.assert_close(y,ref,atol=.025,rtol=.025)
        dy=torch.randn_like(y)
        a=torch.autograd.grad(y,(*tensors,g),dy)
        b=torch.autograd.grad(ref,(*reference_tensors,reference_g),dy)
        for ga,gb in zip(a,b):
            torch.testing.assert_close(ga,gb,atol=.06,rtol=.08)
        assert a[-1].abs().sum()>0
    config=GPTConfig(n_layer=2,n_embd=128,n_head=1,n_kv_head=1,sequence_len=64,vocab_size=256)
    model=FoxGPT(config).to('cuda')
    model.init_weights()
    for block in model.transformer.h:
        torch.nn.init.normal_(block.attn.c_proj.weight,std=.02)
        torch.nn.init.normal_(block.mlp.c_proj.weight,std=.02)
    reference=PilotGPT(config).to('cuda')
    reference.init_weights()
    reference.install_variant('fox')
    reference.load_state_dict(model.state_dict(),strict=True)
    x=torch.randint(0,256,(1,24),device='cuda')
    with torch.no_grad():
        full=model(x)
        torch.testing.assert_close(full,reference(x),atol=.04,rtol=.04)
        cache=FoxCache(1,1,32,128,2,'cuda',torch.bfloat16)
        first=model(x[:,:16],kv_cache=cache)
        expanded=FoxCache(2,1,32,128,2,'cuda',torch.bfloat16)
        expanded.prefill(cache)
        outputs=[first]
        for i in range(16,24):
            outputs.append(model(x[:,i:i+1],kv_cache=cache))
        torch.testing.assert_close(torch.cat(outputs,dim=1),full,atol=.06,rtol=.06)
        copy_out=model(x[:,16:17].expand(2,-1),kv_cache=expanded)
        torch.testing.assert_close(copy_out[0],outputs[1][0],atol=.06,rtol=.06)
    install_cache()
    return dict(status='passed',torch_version=str(torch.__version__),checks=['forward and Q/K/V/gate gradients vs dense reference','native FoX vs pilot FoX','cached prefill/decode vs full sequence','cache replication'])
