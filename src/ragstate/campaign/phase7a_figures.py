"""Dependency-free, deterministic SVG figures and PNG previews for Phase 7A."""
from __future__ import annotations

import html
import math
import struct
import zlib
from pathlib import Path

WIDTH, HEIGHT = 800, 500
COLORS = {'bm25':'#235789','dense':'#c05621','hybrid':'#2f855a',
          'gold':'#c05621','nongold':'#235789'}


class Canvas:
    def __init__(self, title, xlabel, ylabel):
        self.svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
                  '<rect width="800" height="500" fill="white"/>',
                  f'<text x="400" y="34" text-anchor="middle" font-size="20" font-family="sans-serif">{html.escape(title)}</text>',
                  f'<text x="400" y="480" text-anchor="middle" font-size="14" font-family="sans-serif">{html.escape(xlabel)}</text>',
                  f'<text x="20" y="250" transform="rotate(-90 20 250)" text-anchor="middle" font-size="14" font-family="sans-serif">{html.escape(ylabel)}</text>']
        self.pixels=bytearray([255])*(WIDTH*HEIGHT*3)
        self.rect(75,60,755,430,'#f8fafc')
        for value in (0,.25,.5,.75,1):
            y=self.y(value)
            self.line(75,y,755,y,'#d6dce4',1)
            self.label(65,y+4,f'{value:.2f}',anchor='end',size=11)
        self.line(75,60,75,430,'#222222',2)
        self.line(75,430,755,430,'#222222',2)

    @staticmethod
    def y(value):
        return int(430-370*max(0,min(1,value)))

    def _rgb(self,color):
        return bytes.fromhex(color[1:])

    def _pixel(self,x,y,color):
        if 0<=x<WIDTH and 0<=y<HEIGHT:
            index=(y*WIDTH+x)*3
            self.pixels[index:index+3]=self._rgb(color)

    def rect(self,x1,y1,x2,y2,color):
        self.svg.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="{color}"/>')
        rgb=self._rgb(color)
        for y in range(max(0,y1),min(HEIGHT,y2)):
            for x in range(max(0,x1),min(WIDTH,x2)):
                i=(y*WIDTH+x)*3
                self.pixels[i:i+3]=rgb

    def line(self,x1,y1,x2,y2,color,width=2):
        self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>')
        dx=abs(x2-x1);dy=abs(y2-y1);sx=1 if x1<x2 else -1;sy=1 if y1<y2 else -1;err=dx-dy
        while True:
            for ox in range(-width//2,width//2+1):
                for oy in range(-width//2,width//2+1):self._pixel(x1+ox,y1+oy,color)
            if x1==x2 and y1==y2:break
            e2=2*err
            if e2>-dy:err-=dy;x1+=sx
            if e2<dx:err+=dx;y1+=sy

    def label(self,x,y,value,anchor='middle',size=12):
        self.svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-family="sans-serif" fill="#25313f">{html.escape(str(value))}</text>')

    def legend(self,items):
        for i,(name,color) in enumerate(items):
            x=80+i*210
            self.rect(x,42,x+15,52,color)
            self.label(x+22,52,name,anchor='start',size=12)

    def save(self,path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        path.with_suffix('.svg').write_text('\n'.join(self.svg+['</svg>']))
        raw=b''.join(b'\x00'+self.pixels[y*WIDTH*3:(y+1)*WIDTH*3] for y in range(HEIGHT))
        def chunk(kind,data):
            return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
        png=(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',WIDTH,HEIGHT,8,2,0,0,0))+
             chunk(b'IDAT',zlib.compress(raw,9))+chunk(b'IEND',b''))
        path.with_suffix('.png').write_bytes(png)


def _series_plot(output,name,title,measure,depth_rows,scope):
    c=Canvas(title,'retrieval depth k',measure)
    c.legend([(method,COLORS[method]) for method in ('bm25','dense','hybrid')])
    ks=[5,10,20,50]
    for method in ('bm25','dense','hybrid'):
        values=[next(r[measure] for r in depth_rows if r['retriever']==method and r['depth']==k and r['scope']==scope) for k in ks]
        points=[(110+i*200,c.y(v)) for i,v in enumerate(values)]
        for i,(x,y) in enumerate(points):
            c.rect(x-4,y-4,x+5,y+5,COLORS[method])
            if i:c.line(*points[i-1],x,y,COLORS[method],3)
    for i,k in enumerate(ks):c.label(110+i*200,450,k)
    c.save(output/name)


def _hist_plot(output,name,title,values,other=None):
    c=Canvas(title,'value bins','fraction of observations')
    bins=12
    groups=[('gold',values)] if other is None else [('gold',values),('nongold',other)]
    c.legend([(label,COLORS[label]) for label,_ in groups])
    lo=min([x for _,group in groups for x in group],default=-1)
    hi=max([x for _,group in groups for x in group],default=1)
    if hi==lo:hi=lo+1
    for group_i,(label,group) in enumerate(groups):
        counts=[0]*bins
        for value in group:
            counts[min(bins-1,int((value-lo)/(hi-lo)*bins))]+=1
        for i,count in enumerate(counts):
            frac=count/len(group) if group else 0
            x=85+i*54+group_i*(22 if other is not None else 0)
            width=22 if other is not None else 46
            c.rect(x,c.y(frac),x+width,430,COLORS[label])
    for i in (0,3,6,9,12):c.label(85+i*54,450,round(lo+(hi-lo)*i/12,2),size=10)
    c.save(output/name)


def generate_figures(output,depth_rows,per_gold,model_rows,realism_rows):
    output=Path(output)/'figures'
    _series_plot(output,'cer_vs_depth','CER by retrieval depth','cer_at_k',depth_rows,'all_answerable')
    _series_plot(output,'scrs_vs_depth','SCRS by retrieval depth','scrs',depth_rows,'all_answerable')
    _series_plot(output,'multi_document_vs_depth','Multi-document all-required success',
                 'all_required_success',depth_rows,'multi_document')
    margins=[float(x['gold_margin']) for x in per_gold if int(x['dense_hit_at_10'])==0]
    _hist_plot(output,'dense_gold_margin','Dense gold margins for misses at 10',margins)
    c=Canvas('Domain clean SCRS by dense model','domain','SCRS')
    model_ids=list(dict.fromkeys(r['model_id'] for r in model_rows
                                if r['retriever']=='dense' and r['status']=='measured'))
    palette=['#c05621','#2f855a','#6b46c1']
    c.legend([(name.split('/')[-1],palette[i]) for i,name in enumerate(model_ids)])
    for i,domain in enumerate(('hr','finance','engineering','security','sales','legal')):
        for j,model_id in enumerate(model_ids):
            row=next(r for r in model_rows if r['retriever']=='dense' and
                     r['model_id']==model_id and r['scope']==domain and r['status']=='measured')
            x=95+i*108+j*24
            c.rect(x,c.y(float(row['scrs'])),x+20,430,palette[j])
        c.label(125+i*108,449,domain,size=10)
    c.save(output/'domain_scrs_by_model')
    _hist_plot(output,'lexical_overlap','Query token coverage: gold vs best BM25 nongold',
               [x['query_gold_overlap'] for x in realism_rows],
               [x['query_best_nongold_overlap'] for x in realism_rows])
