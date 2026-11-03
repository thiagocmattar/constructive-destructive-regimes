"""Descriptive three-arm view of existing pilot observations; no smoothing/training."""
import csv
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'aligned_diagnostics.csv'
METHODS=[('A0','Unchanged Adam','black'),('A1','Half learning rate','#0072B2'),('A4','Adam proposal clipping','#E69F00')]


def main():
    with SOURCE.open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    selected={arm:[r for r in rows if r['arm']==arm] for arm,_,_ in METHODS}
    for arm,data in selected.items():
        assert [int(r['state']) for r in data]==list(range(657,786))
        assert all(r['loss'] and r['raw_adjacent_cosine'] for r in data)
    all_loss=np.asarray([float(r['loss']) for data in selected.values() for r in data])
    low,high=float(all_loss.min()),float(all_loss.max());pad=.07*(high-low)
    fig,axes=plt.subplots(2,3,figsize=(11.4,5.6),sharex=True,sharey='row')
    fig.subplots_adjust(left=.08,right=.985,bottom=.13,top=.78,wspace=.17,hspace=.14)
    fig.suptitle('Existing local MLP pilot: a clearer three-method view',y=.975,fontsize=14)
    fig.text(.5,.915,'Descriptive selection after exploratory screening; not independent confirmation',ha='center',fontsize=11)
    plotted_values=0
    for col,(arm,name,color) in enumerate(METHODS):
        data=selected[arm];states=np.asarray([int(r['state']) for r in data])
        loss=np.asarray([float(r['loss']) for r in data]);cosine=np.asarray([float(r['raw_adjacent_cosine']) for r in data])
        axes[0,col].set_title(name,fontsize=12,pad=10)
        for row,values in enumerate([loss,cosine]):
            line,=axes[row,col].plot(states,values,color=color,lw=1.25)
            # Direct rendering-input check: all recorded points, with unchanged values.
            assert np.array_equal(line.get_xdata(),states) and np.array_equal(line.get_ydata(),values)
            plotted_values+=len(values)
            axes[row,col].grid(alpha=.17)
            axes[row,col].set_xlim(657,785)
            axes[row,col].set_xticks([657,689,721,753,785])
            axes[row,col].axvline(689,color='.6',ls=':',lw=.8)
        axes[0,col].set_ylim(low-pad,high+pad)
        axes[1,col].set_ylim(-1.05,1.05)
        axes[1,col].axhline(0,color='.6',ls=':',lw=.7)
        axes[1,col].set_xlabel('Parameter state',fontsize=10)
    axes[0,0].set_ylabel('Training loss',fontsize=11)
    axes[1,0].set_ylabel('Adjacent raw cosine',fontsize=11)
    fig.text(.5,.035,'Same scales within each row. All 129 states retained, unsmoothed. Vertical marker: original pre-rise state 689.',ha='center',fontsize=9)
    figure=HERE/'figures/focal_three_arm.pdf';fig.savefig(figure)
    scratch=HERE/'scratch';scratch.mkdir(exist_ok=True)
    fig.savefig(scratch/'focal_three_arm.png',dpi=180);plt.close(fig)
    assert plotted_values==774
    print('774 plotted scalar values match the existing CSV exactly;129 states per method, no smoothing.')
    print('CSV SHA256:',hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    print('Figure SHA256:',hashlib.sha256(figure.read_bytes()).hexdigest())


if __name__=='__main__':main()
