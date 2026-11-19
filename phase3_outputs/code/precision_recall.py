import pickle, numpy as np, csv, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from opencood.utils.eval_utils import voc_ap
files = ['/home/hassan/cnet/out/main2/main2_0_1_2_3_4_5_6_7.pkl','/home/hassan/cnet/out/main2/main2_8_9_10_11_12_13_14_15.pkl']
res = []
for f in files: res += pickle.load(open(f,'rb'))['results']
methods = ['B0','B1','B2','B3','ours','U0']
THR_SCORE = 0.5            # fixed in advance, not tuned
agg = {}
for r in res:
    if r['D'] != 100.0 or r['method'] not in methods: continue
    key = (r['cond'], r['method'])
    a = agg.setdefault(key, {t: [[],[],0] for t in (0.5,0.7)})
    for t in (0.5,0.7):
        tp, sc, gt = r['st'][t][0], r['st'][t][1], r['st'][t][2]
        a[t][0].append(np.asarray(tp)); a[t][1].append(np.asarray(sc)); a[t][2] += gt
def prcurve(tp, sc, gt):
    o = np.argsort(-sc, kind='stable'); tp = tp[o]
    ctp = np.cumsum(tp); cfp = np.cumsum(1-tp)
    return ctp/gt, ctp/np.maximum(ctp+cfp,1)
rows = []; curves = {}
print('score range sample', np.percentile(np.concatenate(agg[('C3','ours')][0.7][1]), [1,25,50,75,99]))
for (cond, m), a in sorted(agg.items()):
    for t in (0.5,0.7):
        tp = np.concatenate(a[t][0]); sc = np.concatenate(a[t][1]); gt = a[t][2]
        rec, prec = prcurve(tp, sc, gt)
        ap = voc_ap(rec.tolist(), prec.tolist())[0]
        keep = sc >= THR_SCORE
        ntp = tp[keep].sum(); ndet = keep.sum()
        P = ntp/ndet if ndet else 0.0; R = ntp/gt; F = 2*P*R/(P+R) if P+R else 0.0
        rows.append([cond, m, t, round(ap,4), round(P,4), round(R,4), round(F,4), int(ndet), int(ntp), int(gt)])
        curves[(cond,m,t)] = (rec, prec)
out = '/mnt/c/Users/aliih/OneDrive/Desktop/opv2v_zips/phase3_outputs/data/main/precision_recall.csv'
with open(out,'w',newline='') as f:
    w = csv.writer(f); w.writerow(['cond','method','IoU','AP_recomputed','precision@score>=0.5','recall@score>=0.5','F1@score>=0.5','n_detections','n_true_pos','n_gt'])
    w.writerows(rows)
for r in rows:
    if r[0]=='C3' or r[1] in ('B3','ours'): print(r)
colors = {'B0':'#888','B1':'#6baed6','B2':'#c9a227','B3':'#d95f02','ours':'#1b9e77','U0':'#7570b3'}
fig, axs = plt.subplots(1,2, figsize=(10,4), sharey=True)
for ax, t in zip(axs, (0.5,0.7)):
    for m in methods:
        rec, prec = curves[('C3',m,t)]; ax.plot(rec, prec, label=m, color=colors[m], lw=1.6)
    ax.set_title(f'C3, IoU {t}'); ax.set_xlabel('recall'); ax.set_xlim(0,1); ax.set_ylim(0.75,1.01)
    for s in ('top','right'): ax.spines[s].set_visible(False)
axs[0].set_ylabel('precision'); axs[0].legend(loc='lower left', fontsize=8)
plt.tight_layout(); plt.savefig('/mnt/c/Users/aliih/OneDrive/Desktop/opv2v_zips/phase3_outputs/figures/main/F8_precision_recall.png', dpi=170)
print('done')
