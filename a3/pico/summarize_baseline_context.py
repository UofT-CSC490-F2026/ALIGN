"""Validate and summarize the baseline-only Part 4 correction."""
import json
import statistics
from pathlib import Path
import random

def paired_delta(before,after):
    a,b=before['documents'],after['documents']
    if [r['document'] for r in a]!=[r['document'] for r in b]:
        raise ValueError('Document mismatch')
    if [r['bytes'] for r in a]!=[r['bytes'] for r in b]:
        raise ValueError('Target byte mismatch')
    deltas=[y['bpb']-x['bpb'] for x,y in zip(a,b)]
    rng=random.Random(20261104)
    bootstrap=sorted(statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(2000))
    return {'mean_document_delta_bpb':statistics.mean(deltas),
            'bootstrap_95_percent_interval':[bootstrap[50],bootstrap[1949]],
            'limitations':'Document bootstrap conditional on this checkpoint; not training-seed uncertainty.'}




def summarize(folder):
    folder=Path(folder)
    data=json.loads((folder/'comparison.json').read_text())
    rows=data['results']
    assert len(rows)==4
    assert {(r['seed'],r['training_context']) for r in rows}=={(s,c) for s in (42,43) for c in (512,2048)}
    assert all(r['status']=='complete' and r['variant']=='baseline' and r['native_baseline']
               and r['tokens_seen']==33554432 and r['seed_training_tokens_before']==16384000 for r in rows)
    assert len({r['tokenizer_sha256'] for r in rows})==1
    assert all(r['eval_suite_sha256']==data['eval_suite_sha256'] for r in rows)
    assert len({r['parameters'] for r in rows})==1
    assert all(r['source_sha256']==data['source_sha256'][str(r['seed'])] for r in rows)
    assert all(r['model']['n_layer']==6 and r['model']['n_embd']==384
               and r['model']['n_head']==6 and r['training_shards']==4 for r in rows)
    def metric(seed,trained,tested):
        if trained==0:
            return next(r['suffix_eval'] for r in data['originals'] if r['seed']==seed and r['context']==tested)
        return next(r['suffix_eval'] for r in data['cross_eval'] if r['seed']==seed and r['training_context']==trained and r['evaluation_context']==tested)
    means={str(t):{str(e):statistics.mean(metric(s,t,e)['bpb'] for s in (42,43))
                   for e in (512,2048)} for t in (0,512,2048)}
    paired={str(s):paired_delta(metric(s,512,2048),metric(s,2048,2048)) for s in (42,43)}
    aggregate_deltas={str(s):metric(s,2048,2048)['bpb']-metric(s,512,2048)['bpb'] for s in (42,43)}
    summary=dict(means=means,paired_document_stats=paired,aggregate_deltas=aggregate_deltas,
                 continuation_runtime_cost_estimate=sum(r['estimated_runtime_cost_usd'] for r in rows))
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Part 4: native baseline context extension','',
           'Original Part 2 baseline weights; upstream GPT with RoPE. '
           'Two seeds; same original tokenizer and four ClimbMix training shards. '
           'Each continuation adds 33.554432M tokens to the initial 16.384M.', '',
           '| Checkpoint | Eval 512 BPB | Eval 2048 BPB |','|---|---:|---:|']
    for trained,label in ((0,'Original 512'),(512,'Continued 512'),(2048,'Continued 2048')):
        m=means[str(trained)]
        lines.append(f"| {label} | {m['512']:.5f} | {m['2048']:.5f} |")
    lines += ['', 'Both evaluation windows score identical final 256-token suffixes of 48 held-out documents. '
              'Optimizer and data order restart in both arms; RoPE retained without interpolation. '
              'SSSL windows increase with configured context. Two seeds do not establish generality.', '',
              '## Paired uncertainty', '', '```json',json.dumps(summary,indent=2),'```','',
              '## Verified W&B runs','']
    for row in rows:
        audit=next(a for a in data['wandb_audit'] if a['name']==row['name'])
        assert audit['state']=='finished' and audit['loss_points']>0 and audit['validation_points']>0
        lines.append(f"- [{row['name']}]({row['wandb_url']}): {audit['loss_points']} loss points, {audit['validation_points']} validation points.")
    lines+=['','Runtime estimate excludes original-checkpoint evaluation, cross-evaluation, startup, and other overhead; not an invoice.','']
    text='\n'.join(lines)
    (folder/'REPORT.md').write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':
    summarize(Path(__file__).resolve().parents[1]/'baseline_context_results')
