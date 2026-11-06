"""Part 3 model-FLOP convention and hardware provenance (not hardware counters)."""
import torch
from nanochat import flash_attention

SPECS = {
    'A10': (125e12, .000306),
    'A100-80GB': (312e12, .000694),
    'H100': (989e12, .001097),
}


def describe(model, variant, hardware):
    name = torch.cuda.get_device_name()
    required = {'A10': 'a10', 'A100-80GB': 'a100', 'H100': 'h100'}[hardware]
    assert required in name.lower(), (hardware, name)
    if hardware == 'A10':
        assert 'a100' not in name.lower()
    if hardware == 'H100':
        assert 'hbm3' in name.lower() or 'sxm' in name.lower(), name
    assert torch.cuda.is_bf16_supported()
    peak, price = SPECS[hardware]
    # Modal's A10 SKU may deliver A10G. Use its actual vendor BF16 peak.
    # Historical hardware-v1 logs are preserved and corrected in reporting.
    if 'a10g' in name.lower():
        peak = 70e12
    props = torch.cuda.get_device_properties(0)
    backend = ('dense-reference-fp32-QK-bf16-PV' if variant == 'fox' else
               ('FA3:' + str(getattr(flash_attention._fa3, '__file__', 'unknown'))
                if flash_attention.USE_FA3 else 'torch-SDPA-auto'))
    return dict(hardware=hardware, gpu_memory_bytes=props.total_memory,
                compute_capability=list(torch.cuda.get_device_capability()),
                cuda_version=torch.version.cuda, attention_backend=backend,
                model_flops_per_token=model.estimate_flops(),
                dense_bf16_peak_flops=peak, gpu_dollars_per_second=price,
                flops_convention='upstream: 6*Linear-weight-count + 12*heads*head_dim*sum(min(context,window)); excludes elementwise, optimizer, and redundant dense masked work',
                mfu_caveat='estimated useful-model FLOPs / dense BF16 peak; mixed FP32 attention and dense reference overhead make this NOT hardware utilization',
                warmup_steps_excluded=5)
