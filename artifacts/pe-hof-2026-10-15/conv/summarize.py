import csv
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
rows = {}
for path in sorted((root / 'results').glob('*.csv')):
    for row in csv.DictReader(path.open()):
        key = (row['case'], int(row['N']), int(row['threads']), row['variant'])
        rows.setdefault(key, []).append(float(row['median_ms']))
lines = ['| Shape (N,Cin,Cout,H,W) | Threads | Direct ms | Im2col ms | Winograd ms | Direct / Winograd |', '|---|---:|---:|---:|---:|---:|']
for n in (1,4):
    for t in (1,4):
        for name,ci,co,h,w in [('small_16',16,16,32,32),('resnet_32_64',32,64,28,28),('resnet_128',128,128,28,28),('vgg_512',512,512,7,7),('k27',3,64,32,32),('odd',13,17,17,19)]:
            vals = [median(rows[(name,n,t,v)]) for v in ('direct','trans_im2col','winograd')]
            assert all(len(rows[(name,n,t,v)]) == 3 for v in ('direct','trans_im2col','winograd'))
            lines.append(f'| {n},{ci},{co},{h},{w} | {t} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[0]/vals[2]:.2f}× |')
(root / 'table.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
