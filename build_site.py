"""Build the self-contained interactive findings site (talent_flow_findings.html).

Reads the pipeline outputs in data/processed/flow/ plus the executed notebook,
reconciles duplicate org records, and renders interactive Plotly charts into a
single offline HTML file. Run after `make collect`:

    python build_site.py
"""
import html, re, json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parent
F = ROOT/"data/processed/flow"

persons  = pd.read_parquet(F/"persons.parquet")
training = pd.read_parquet(F/"person_training.parquet")
employer = pd.read_parquet(F/"person_employer.parquet")
weights  = pd.read_csv(F/"person_weights.csv")
works    = pd.read_parquet(ROOT/"data/processed/works.parquet")
auth     = pd.read_parquet(ROOT/"data/processed/authorships.parquet")
orgs     = pd.read_csv(ROOT/"data/orgs.csv")
flow     = pd.read_parquet(F/"flow_matrix.parquet")

INDUSTRY_TYPES={"industry","industry_lab","industry_parent"}
name_of=orgs.set_index("org_id")["canonical_name"].to_dict()
type_of=orgs.set_index("org_id")["org_type"].to_dict()
country_of=orgs.set_index("org_id")["country"].to_dict()
GROUPS=[('Meta',('meta','facebook','fair')),('Alphabet / Google',('google','deepmind','alphabet','waymo')),
 ('Microsoft',('microsoft','msra')),('Amazon',('amazon',)),('Apple',('apple',)),('NVIDIA',('nvidia',)),
 ('Alibaba',('alibaba','damo')),('Tencent',('tencent',)),('ByteDance',('bytedance','tiktok')),('Baidu',('baidu',)),
 ('Huawei',('huawei','noah')),('Ant Group',('ant group','ant financial')),('Adobe',('adobe',)),
 ('IBM',('ibm','international business mach')),('Samsung',('samsung',)),('Intel',('intel',)),('Tesla',('tesla',)),
 ('OpenAI',('openai',)),('Anthropic',('anthropic',)),('DeepSeek',('deepseek',)),('Salesforce',('salesforce',)),
 ('Xiaomi',('xiaomi',)),('JD.com',('jingdong','jd.com')),('Snap',('snapchat','snap inc')),
 ('Moonshot AI',('moonshot','kimi')),('Mistral AI',('mistral',)),('xAI',('xai','x.ai')),('Cohere',('cohere',)),
 ('Zhipu AI',('zhipu',)),('Sony',('sony',)),('Bosch',('bosch',)),('Qualcomm',('qualcomm',))]
def group_label(oid):
    if oid is None or (isinstance(oid,float) and pd.isna(oid)): return None
    n=str(name_of.get(oid,oid)); t=type_of.get(oid,'')
    if t in INDUSTRY_TYPES:
        nl=n.lower()
        for lab,keys in GROUPS:
            if any(k in nl for k in keys): return lab
    return re.sub(r'\s*\(.*?\)\s*',' ',n).strip()
grp_ind={}
for oid,t in type_of.items():
    g=group_label(oid); grp_ind[g]=grp_ind.get(g,False) or (t in INDUSTRY_TYPES)
def is_ind(g): return bool(grp_ind.get(g,False))

# ---- computations ----
pt=training.dropna(subset=["training_org_id"]).copy(); pt["inst"]=pt["training_org_id"].map(group_label)
inst_rank=pt.groupby("inst")["author_dblp_pid"].nunique().sort_values(ascending=False).head(15)
lab=pt.dropna(subset=["advisor_name"]).copy(); lab["lab"]=lab["advisor_name"].astype(str)+" — "+lab["inst"].str.slice(0,32)
lab_rank=lab.groupby("lab")["author_dblp_pid"].nunique().sort_values(ascending=False).head(15)

def bucket(oid):
    t=type_of.get(oid,'')
    if t in INDUSTRY_TYPES: return 'Industry'
    return {'academic':'Academia','nonprofit':'Non-profit','government':'Government'}.get(t,'Other')
emp=employer.dropna(subset=["employer_org_id"]).copy()
emp["bucket"]=emp["employer_org_id"].map(bucket); emp["grp"]=emp["employer_org_id"].map(group_label)
bucket_counts=emp["bucket"].value_counts()
top_dest=emp[emp.grp.map(is_ind)].groupby("grp")["author_dblp_pid"].nunique().sort_values(ascending=False).head(15)

w=weights[["author_dblp_pid","weight"]]
pe=employer.merge(w,on="author_dblp_pid",how="left")
pe["weight"]=pe["weight"].fillna(0); pe["conf"]=pe["confidence"].fillna(0.4)
pe["contribution"]=pe["weight"]*pe["conf"]+pe["is_founder"].astype(bool)*pe["weight"]
pe["grp"]=pe["employer_org_id"].map(group_label)
gi=pe.groupby("grp").agg(intensity=("contribution","sum"),n=("author_dblp_pid","nunique")).reset_index()
gi=gi[gi["grp"].map(is_ind)].sort_values("intensity",ascending=False)
gi["idx"]=100*gi["intensity"]/gi["intensity"].max()
inten_rank=gi.set_index("grp")["idx"].head(15); top=gi.head(12)

fl=flow.copy()
fl["labn"]=fl["training_org_id"].map(group_label).str.slice(0,22)+" — "+fl["advisor_dblp_pid"].astype(str).str.slice(0,12)
fl["dest"]=fl["employer_org_id"].map(group_label); fl=fl[fl["dest"].map(is_ind)]
tl=fl.groupby("labn")["n"].sum().sort_values(ascending=False).head(12).index
td=fl.groupby("dest")["n"].sum().sort_values(ascending=False).head(10).index
mat=(fl[fl.labn.isin(tl)&fl.dest.isin(td)].pivot_table(index="labn",columns="dest",values="n",aggfunc="sum",fill_value=0)
     .reindex(index=tl,columns=td,fill_value=0)).loc[tl[::-1]]

cf=employer.merge(training[["author_dblp_pid","training_org_id"]],on="author_dblp_pid",how="inner")
cf["from"]=cf["training_org_id"].map(country_of); cf["to"]=cf["employer_org_id"].map(country_of)
cf=cf[(cf["from"].astype(str).str.len()==2)&(cf["to"].astype(str).str.len()==2)]
tc=pd.concat([cf["from"],cf["to"]]).value_counts().head(8).index
cm=(cf[cf["from"].isin(tc)&cf["to"].isin(tc)].groupby(["from","to"]).size().unstack(fill_value=0)
    .reindex(index=tc,columns=tc,fill_value=0)).loc[tc[::-1]]

top_firm=gi.iloc[0]["grp"]
feed=pe[pe.grp==top_firm].merge(training[["author_dblp_pid","training_org_id"]],on="author_dblp_pid",how="left")
feed["inst"]=feed["training_org_id"].map(group_label)
feed_rank=feed.dropna(subset=["inst"]).groupby("inst")["author_dblp_pid"].nunique().sort_values(ascending=False).head(10)

# ---- campus -> company flow (institution -> industry firm), for the interactive drill-down + treemap ----
_ei=employer.dropna(subset=["employer_org_id"]).copy(); _ei["company"]=_ei["employer_org_id"].map(group_label)
_ei=_ei[_ei["company"].map(is_ind)][["author_dblp_pid","company"]].drop_duplicates("author_dblp_pid")
_ti=training.dropna(subset=["training_org_id"]).copy(); _ti["inst"]=_ti["training_org_id"].map(group_label)
_ti=_ti[["author_dblp_pid","inst"]].dropna().drop_duplicates("author_dblp_pid")
flow_ic=(_ti.merge(_ei,on="author_dblp_pid",how="inner")
            .groupby(["inst","company"])["author_dblp_pid"].nunique().reset_index(name="n"))
inst_ind_total=flow_ic.groupby("inst")["n"].sum().sort_values(ascending=False).head(15)

def firms_for(inst,topn=10):
    sub=flow_ic[flow_ic["inst"]==inst].sort_values("n",ascending=False)
    total=int(sub["n"].sum()); head=sub.head(topn); tail=int(sub["n"].iloc[topn:].sum())
    labels=list(head["company"]); counts=[int(x) for x in head["n"]]
    if tail>0: labels.append("Other"); counts.append(tail)
    return labels,counts,total

FLOW_JSON=[]
for _inst in inst_ind_total.index:
    _lab,_cnt,_tot=firms_for(_inst)
    FLOW_JSON.append({"inst":_inst,"total":_tot,
        "companies":[{"company":c,"count":n,"prop":round(n/_tot,4)} for c,n in zip(_lab,_cnt)]})
FLOW_DATA=json.dumps(FLOW_JSON)

S=dict(talent=len(persons),works=len(works),auth=len(auth),orgs=len(orgs),
       employed=len(employer),trained=len(training),firms=int(gi.shape[0]),
       flow_people=int(flow_ic["n"].sum()))

# ---- plotly helpers ----
FONT="system-ui,-apple-system,'Segoe UI',Roboto,sans-serif"
BLUE="#2a78d6"; SEC="#52514e"; MUT="#898781"; INK="#0b0b0b"; GRID="#e6e5df"; SURF="#ffffff"
BLUESCALE=[[0,'#eef5fd'],[0.25,'#cde2fb'],[0.5,'#86b6ef'],[0.75,'#3987e5'],[1,'#0d366b']]
def _lay(fig,h):
    fig.update_layout(height=h,margin=dict(l=8,r=30,t=8,b=44),paper_bgcolor=SURF,plot_bgcolor=SURF,
        font=dict(family=FONT,size=13,color=INK),showlegend=False,bargap=0.30,
        hoverlabel=dict(bgcolor=INK,font=dict(color='#fff',family=FONT,size=13)))
    fig.update_xaxes(showgrid=True,gridcolor=GRID,zeroline=False,tickfont=dict(color=MUT,size=11),title_standoff=10,
        title_font=dict(color=MUT,size=12))
    fig.update_yaxes(showgrid=False,zeroline=False,automargin=True,tickfont=dict(color=SEC,size=12.5))
    return fig
def _div(fig): return fig.to_html(full_html=False,include_plotlyjs=False,
        config={'displayModeBar':False,'responsive':True},default_width='100%')
def hbar(series,xlabel,fmt="{:.0f}",accent=BLUE):
    d=series.iloc[::-1]
    fig=go.Figure(go.Bar(x=list(d.values),y=[str(i) for i in d.index],orientation='h',
        marker=dict(color=accent,line_width=0),text=[fmt.format(v) for v in d.values],
        textposition='outside',textfont=dict(color=SEC,size=11),cliponaxis=False,
        hovertemplate='<b>%{y}</b><br>'+xlabel+': %{x}<extra></extra>'))
    fig.update_xaxes(title_text=xlabel)
    return _div(_lay(fig,36*len(d)+150))
def heat(mat,xlabel,ylabel):
    fig=go.Figure(go.Heatmap(z=mat.values,x=list(mat.columns),y=list(mat.index),colorscale=BLUESCALE,
        text=mat.values,texttemplate="%{text:d}",textfont=dict(size=11),xgap=2,ygap=2,
        hovertemplate=ylabel+' %{y} → '+xlabel+' %{x}<br>people: %{z}<extra></extra>',
        colorbar=dict(title=dict(text='people',font=dict(size=11,color=MUT)),thickness=12,len=0.75,outlinewidth=0)))
    fig.update_xaxes(title_text=xlabel,tickangle=-30,tickfont=dict(color=SEC,size=12),title_font=dict(color=MUT,size=12))
    fig.update_yaxes(title_text=ylabel,tickfont=dict(color=SEC,size=12),automargin=True,title_font=dict(color=MUT,size=12))
    _lay(fig,44*len(mat.index)+190); fig.update_layout(margin=dict(l=8,r=8,t=8,b=96))
    return _div(fig)

# ---- drill-down treemap: university -> firm, with luminance-aware text + no grey frame ----
ACCENT="#0d366b"
def _hex2rgb(h): n=int(h.lstrip('#'),16); return ((n>>16)&255,(n>>8)&255,n&255)
_SC=[(p,_hex2rgb(h)) for p,h in BLUESCALE]
def _scale_rgb(t):
    t=min(max(t,0.0),1.0)
    for (p0,c0),(p1,c1) in zip(_SC,_SC[1:]):
        if t<=p1:
            f=0 if p1==p0 else (t-p0)/(p1-p0); return tuple(c0[k]+f*(c1[k]-c0[k]) for k in range(3))
    return _SC[-1][1]
def treemap():
    ids,labs,par,val=[],[],[],[]
    for rec in FLOW_JSON:
        ids.append(rec["inst"]); labs.append(rec["inst"]); par.append(""); val.append(rec["total"])
        for c in rec["companies"]:
            ids.append(f'{rec["inst"]} | {c["company"]}'); labs.append(c["company"])
            par.append(rec["inst"]); val.append(c["count"])
    cmax=max(val) if val else 1
    def tc(v):
        r,g,b=_scale_rgb(v/cmax); return '#ffffff' if (0.2126*r+0.7152*g+0.0722*b)/255<0.6 else INK
    fig=go.Figure(go.Treemap(ids=ids,labels=labs,parents=par,values=val,branchvalues="total",maxdepth=2,
        marker=dict(colors=val,colorscale=BLUESCALE,cmin=0,cmax=cmax,line=dict(color=SURF,width=1.5),cornerradius=4),
        root_color="rgba(0,0,0,0)",tiling=dict(packing="squarify",pad=1),
        texttemplate="%{label}<br>%{value} · %{percentParent:.0%}",textposition="middle center",
        textfont=dict(size=12,color=[tc(v) for v in val]),
        pathbar=dict(visible=True,thickness=20),
        hovertemplate="<b>%{label}</b><br>%{value} people · %{percentParent:.0%} of parent<extra></extra>"))
    fig.update_layout(height=540,margin=dict(l=4,r=4,t=4,b=4),paper_bgcolor=SURF,plot_bgcolor=SURF,
        font=dict(family=FONT,size=12,color=INK))
    return _div(fig)

# ---- client-side interactive: click a university bar -> its campus->company Sankey (plain string; braces are JS, not f-string) ----
JS_BLOCK="<script>\nconst FLOW = "+FLOW_DATA+";\n"+r"""
(function(){
  const INK='#0b0b0b',SEC='#52514e',MUT='#898781',SURF='#ffffff',GRID='#e6e5df',ACC='#0d366b';
  const FONT="system-ui,-apple-system,'Segoe UI',Roboto,sans-serif";
  const STOPS=['#cde2fb','#86b6ef','#3987e5','#256abf','#0d366b'];
  const CFG={displayModeBar:false,responsive:true};
  const order=FLOW.slice().reverse();                 // bottom-up so the leader sits on top
  function shade(v,mx){const t=mx?v/mx:0;return STOPS[Math.min(STOPS.length-1,Math.floor(t*STOPS.length))];}
  function rgba(h,a){const n=parseInt(h.slice(1),16);return 'rgba('+((n>>16)&255)+','+((n>>8)&255)+','+(n&255)+','+a+')';}
  function barColors(sel){return order.map(r=>r.inst===sel?ACC:'#a9c8f0');}
  function drawBar(sel){
    Plotly.react('flowbar',[{type:'bar',orientation:'h',
      x:order.map(r=>r.total),y:order.map(r=>r.inst),
      marker:{color:barColors(sel),line:{width:0}},
      text:order.map(r=>r.total),textposition:'outside',textfont:{color:SEC,size:11},cliponaxis:false,
      hovertemplate:'%{y}<br>%{x} to industry — click to drill<extra></extra>'}],
      {height:440,margin:{l:212,r:34,t:8,b:42},paper_bgcolor:SURF,plot_bgcolor:SURF,
       font:{family:FONT,size:12,color:INK},bargap:0.3,
       xaxis:{title:{text:'talent → industry (authors)',font:{size:11,color:MUT}},gridcolor:GRID,zeroline:false,tickfont:{color:MUT,size:10}},
       yaxis:{tickfont:{color:INK,size:11},automargin:true}},CFG);
  }
  function drawSankey(rec){
    const comps=rec.companies,mx=Math.max.apply(null,comps.map(c=>c.count));
    const labels=[rec.inst].concat(comps.map(c=>c.company+' — '+Math.round(c.prop*100)+'%'));
    const ncol=[ACC].concat(comps.map(c=>shade(c.count,mx)));
    Plotly.react('flowsankey',[{type:'sankey',arrangement:'snap',
      node:{label:labels,color:ncol,pad:15,thickness:16,line:{color:'white',width:1.2},hovertemplate:'%{label}<extra></extra>'},
      link:{source:comps.map(()=>0),target:comps.map((_,i)=>i+1),value:comps.map(c=>c.count),
            color:comps.map(c=>rgba(shade(c.count,mx),0.45)),hovertemplate:'%{value} people<extra></extra>'}}],
      {height:440,margin:{l:10,r:10,t:30,b:10},paper_bgcolor:SURF,font:{family:FONT,size:11,color:INK},
       title:{text:'<b>'+rec.inst+'</b> → firms · <span style="color:#898781">'+rec.total+' to industry</span>',font:{size:13},x:0,xanchor:'left'}},CFG);
  }
  function init(){
    if(!window.Plotly){return setTimeout(init,60);}
    drawBar(FLOW[0].inst); drawSankey(FLOW[0]);
    document.getElementById('flowbar').on('plotly_click',function(e){
      if(!e.points||!e.points.length)return;
      const inst=e.points[0].y, rec=FLOW.find(r=>r.inst===inst);
      if(rec){drawSankey(rec); drawBar(inst);}
    });
  }
  if(document.readyState!=='loading')init(); else document.addEventListener('DOMContentLoaded',init);
})();
</script>"""

FIG={
 'inst':hbar(inst_rank,"scholars trained"),
 'lab':hbar(lab_rank,"trainees"),
 'bucket':hbar(bucket_counts,"scholars"),
 'dest':hbar(top_dest,"scholars hired"),
 'labflow':heat(mat,"firm","lab"),
 'country':heat(cm,"employer country","training country"),
 'feed':hbar(feed_rank,"scholars"),
 'treemap':treemap(),
}
INTERP={
 'inst':"Where the world's 2024–25 AI authors did their doctoral-era training. Chinese universities (Tsinghua, Peking, SJTU, "
   "Zhejiang, USTC, CAS) lead the raw counts — consistent with China's large share of top-venue output — ahead of leading US "
   "and European schools. Ranked by author volume; an elite-tier or citation-weighted cut would tilt toward US institutions.",
 'lab':"The individual labs that trained the most 2024–25 authors. Advisor is inferred from a scholar's earliest-career "
   "co-authorship with a known faculty member — a proxy for the doctoral advisor.",
 'bucket':"Of scholars whose current employer we could resolve, most remain in academia; industry is the smaller, high-value "
   "stream. Employer resolution is partial, so treat these as lower bounds.",
 'dest':"Among industry-bound talent, Microsoft, Google/Alphabet and Meta lead alongside the Chinese giants (Alibaba, Tencent, "
   "ByteDance, Huawei, Ant) — the firms hiring the most top-venue authors.",
 'labflow':"Which training labs feed which firms. Hover any cell for the exact count — it reads as a supply chain from elite "
   "labs into the major industrial employers.",
 'country':"Training country → current-employer country. Domestic retention dominates (China→China, US→US), with a pronounced "
   "US↔China cross-border stream.",
 'feed':f"For the #1 firm ({html.escape(top_firm)}), the training institutions that supplied the most of its talent.",
 'flow':"Click any university on the left to redraw the Sankey on the right: exactly which firms its industry-bound "
   "talent joined, with each firm's share of that university's industry hires on the node label. Counts are lower "
   "bounds (employer resolved for a minority of talent), so read the mix and direction rather than the absolute size.",
 'treemap':"The same campus→company flow as a treemap. Click a university tile to zoom into its firms; click the "
   "breadcrumb bar to zoom back out. Tile size and shade encode people; the label shows each firm's share of its university.",
}
def card(key,title):
    return (f'<figure><h3>{html.escape(title)}</h3>{FIG[key]}'
            f'<p class="interp">{INTERP[key]}</p></figure>')

MEDAL={1:'👑',2:'🥈',3:'🥉'}
rows="".join(
    (f"<tr class='lead-row'>" if i==0 else "<tr>")
    +f"<td class='r'>{MEDAL.get(i+1,str(i+1))}</td>"
    +f"<td>{html.escape(x.grp)}</td>"
    +f"<td class='meter-cell'><span class='meter'><i style='width:{x.idx:.0f}%'></i></span><span class='mv'>{x.idx:.1f}</span></td>"
    +f"<td class='n'>{int(x.n)}</td></tr>"
    for i,x in enumerate(top.itertuples()))
def stat(n,l): return f'<div class="stat"><div class="n">{n}</div><div class="l">{l}</div></div>'

HTML=f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Talent Intensity — Person-Centric Bibliometric</title>
<script type="text/javascript">{get_plotlyjs()}</script>
<style>
:root{{--surface:#fcfcfb;--ink:#0b0b0b;--sec:#52514e;--muted:#898781;--blue:#2a78d6;--blue-l:#5598e7;
--grid:#e6e5df;--card:#ffffff;--amber-bg:#fff8ee;--amber-bd:#f0dcae;}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}}
body{{margin:0;background:var(--surface);color:var(--ink);font-family:{FONT};line-height:1.6;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:940px;margin:0 auto;padding:0 22px}}
.hero{{background:linear-gradient(180deg,#eef4fc 0%,var(--surface) 100%);color:var(--ink);
padding:58px 0 46px;border-top:3px solid var(--blue);border-bottom:1px solid var(--grid)}}
.hero .eyebrow{{color:var(--blue);font-size:.82rem;font-weight:600;letter-spacing:.14em;text-transform:uppercase;margin-bottom:16px}}
.hero h1{{font-size:clamp(2.1rem,5vw,3.2rem);font-weight:800;letter-spacing:-.025em;line-height:1.05;margin:0 0 14px;max-width:17ch}}
.hero h1 .hl{{color:var(--blue)}}
.hero .sub{{color:var(--sec);font-size:1.1rem;max-width:60ch;margin:0}}
.stats{{display:flex;flex-wrap:wrap;gap:14px;margin:36px 0 28px}}
.stat{{flex:1 1 170px;background:var(--card);border:1px solid var(--grid);border-radius:16px;padding:22px 20px;
box-shadow:0 1px 2px rgba(11,11,11,.04),0 10px 30px rgba(11,11,11,.05)}}
.stat .n{{font-size:clamp(2.1rem,4.5vw,2.9rem);font-weight:800;letter-spacing:-.02em;color:var(--blue);line-height:1;font-variant-numeric:tabular-nums}}
.stat .l{{color:var(--sec);font-size:.9rem;margin-top:8px}}
.method{{border-left:3px solid var(--blue);padding:2px 0 2px 18px;color:var(--sec);max-width:84ch;font-size:1.02rem}}
.method b{{color:var(--ink)}}
main{{padding:8px 0 90px}}
.sec{{padding-top:50px}}
.sec-head{{display:flex;align-items:center;gap:14px;margin-bottom:4px}}
.badge{{flex:none;width:34px;height:34px;border-radius:50%;background:var(--blue);color:#fff;font-weight:700;display:grid;place-items:center}}
.sec-head h2{{font-size:1.55rem;letter-spacing:-.01em;margin:0}}
.lead{{color:var(--sec);max-width:80ch;margin:2px 0 6px 48px}}
figure{{margin:22px 0 0;background:var(--card);border:1px solid var(--grid);border-radius:18px;padding:20px 20px 22px;
box-shadow:0 1px 2px rgba(11,11,11,.04),0 10px 30px rgba(11,11,11,.06)}}
figure h3{{margin:.1em 0 .3em;font-size:1.16rem;letter-spacing:-.01em}}
.interp{{color:var(--sec);font-size:.96rem;margin:1.1em 2px 0}}
.flowgrid{{display:grid;grid-template-columns:.82fr 1.18fr;gap:16px;align-items:start}}
.flowgrid>div{{min-height:440px}}
@media(max-width:760px){{.flowgrid{{grid-template-columns:1fr}}}}
.tbl{{margin:22px 0 0;background:var(--card);border:1px solid var(--grid);border-radius:18px;overflow:hidden;box-shadow:0 10px 30px rgba(11,11,11,.06)}}
table{{border-collapse:collapse;width:100%;font-size:.96rem}}
th,td{{padding:11px 16px;border-bottom:1px solid var(--grid);text-align:left}}
tr:last-child td{{border-bottom:none}} th{{color:var(--muted);font-weight:600;font-size:.74rem;text-transform:uppercase;letter-spacing:.06em;background:#faf9f6}}
td.r{{width:46px;text-align:center;font-size:1.1rem;line-height:1;color:var(--muted);font-variant-numeric:tabular-nums}} td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}} td:nth-child(2){{font-weight:600}}
tr.lead-row{{background:linear-gradient(90deg,#eef5fd 0%,rgba(238,245,253,0) 70%)}}
tr.lead-row td:nth-child(2){{color:var(--blue)}}
.meter-cell{{white-space:nowrap}}
.meter{{display:inline-block;width:74px;height:7px;background:var(--grid);border-radius:4px;overflow:hidden;vertical-align:middle;margin-right:9px}}
.meter i{{display:block;height:100%;background:var(--blue);border-radius:4px}}
.mv{{font-variant-numeric:tabular-nums;font-weight:600;color:var(--sec)}}
.prov{{margin-left:7px;font-size:.6rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;color:#9a7b2a;background:#fff5e0;border:1px solid #f0dcae;border-radius:10px;padding:1px 7px;vertical-align:middle}}
.caveat{{margin-top:22px;background:var(--amber-bg);border:1px solid var(--amber-bd);border-radius:16px;padding:20px 24px}}
.caveat h3{{margin:.1em 0 .5em;font-size:1.1rem}} .caveat li{{margin:.3em 0;color:#5a4a2a}}
footer{{color:var(--muted);font-size:.85rem;margin-top:54px;border-top:1px solid var(--grid);padding-top:22px}}
</style></head><body>
<section class="hero"><div class="wrap">
<div class="eyebrow">AI Intensity · person-centric bibliometric study</div>
<h1>Where the world's <span class="hl">AI talent</span> trains — and where it goes</h1>
<p class="sub">A lab- and company-level map of elite AI researchers, reconstructed from public scholarly records.</p>
<div class="stats">{stat(f"{S['talent']:,}","scholars identified")}{stat(f"{S['works']:,}","papers scanned")}{stat(f"{S['orgs']:,}","organizations mapped")}</div>
<p class="method">We pull from <b>five public sources</b> (DBLP, CSRankings, Semantic Scholar, ORCID, Wikidata) to identify every author
at <b>nine top AI venues in 2024–2025</b>, <b>backtrack</b> each scholar's full publication history to their doctoral-era lab and
advisor, and <b>map</b> affiliations to a curated organization registry — tracing where talent trained and where it went.</p>
</div></section>
<main class="wrap">
<section class="sec"><div class="sec-head"><div class="badge">1</div><h2>How firms rank on AI-talent intensity</h2></div>
<p class="lead">A provisional ranking of firms by the volume and citation-weight of the top-venue talent they employ.
The intensity score is still experimental and not yet formally defined — read the <em>ordering</em> as indicative, not the exact values.</p>
<div class="tbl"><table><thead><tr><th class="r">#</th><th>Firm</th><th>Talent-intensity index<span class="prov">provisional</span></th><th class="n">Scholars</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="interp" style="margin-left:2px">Bars are scaled to the leader (index&nbsp;=&nbsp;100); “Scholars” is the count of resolved employees in the talent set. See <em>How to read these numbers</em> below for how the score is currently computed.</p>
</section>
<section class="sec"><div class="sec-head"><div class="badge">2</div><h2>Where the world's AI talent trains</h2></div>
<p class="lead">Start with supply: the institutions and individual labs (institution + advisor) that educate the most 2024–25 top-venue authors.</p>
{card('inst','Top training institutions — where the most authors at top AI conferences (2024–25) trained')}{card('lab','Top advisors — who mentored the most authors at top AI conferences (2024–25)')}
</section>
<section class="sec"><div class="sec-head"><div class="badge">3</div><h2>Where that talent goes: academia vs industry</h2></div>
<p class="lead">Of scholars whose current employer we could resolve, most stay in academia; the flow into industry is the smaller, high-value stream — and these are the firms hiring the most of it.</p>
{card('bucket','Where talent sits now')}{card('dest','Top industry destinations')}
</section>
<section class="sec"><div class="sec-head"><div class="badge">4</div><h2>From campus to company</h2></div>
<p class="lead">Now zoom into the industry stream: for each top university, exactly which firms did its talent join? Click a university to trace its flow.</p>
<figure><h3>University → firm — click a university to drill in</h3>
<div class="flowgrid"><div id="flowbar"></div><div id="flowsankey"></div></div>
<p class="interp">{INTERP['flow']}</p></figure>
{card('treemap','Drill-down treemap: university → firm')}
</section>
<section class="sec"><div class="sec-head"><div class="badge">5</div><h2>Pipelines &amp; borders</h2></div>
<p class="lead">The finer-grained supply chain — from individual labs into firms, and across national borders. Hover any cell for exact counts.</p>
{card('labflow','Lab → firm flow')}{card('country','Cross-border talent flow')}{card('feed','Which labs feed the #1 firm')}
</section>
<section class="sec"><div class="sec-head"><div class="badge">!</div><h2>How to read these numbers</h2></div>
<div class="caveat"><h3>Honest caveats</h3><ul>
<li><b>Publication-visibility bias.</b> Talent is measured through papers, so firms that publish little relative to headcount
(e.g. OpenAI, Anthropic) are under-counted — they appear here mainly via <em>founders</em>, not employee mass.</li>
<li><b>Volume vs. eliteness.</b> Institutions are ranked by author count, which favors high-output universities.</li>
<li><b>Employer coverage is partial</b> ({S['employed']:,} of {S['talent']:,} resolved; higher when weighted by citations) —
destination <em>counts</em> are lower bounds, <em>rankings</em> are robust.</li>
<li><b>Advisor is a co-authorship proxy</b>, not a verified enrollment record.</li>
</ul></div></section>
<footer>Built from public scholarly records · seed window 2024–2025 · interactive figures regenerated from the analysis pipeline.
Unlike MacroPolo's manually-verified, country-level tracker on three ML venues, this automates a lab- and company-level map across nine venues.</footer>
</main>{JS_BLOCK}</body></html>"""

out=ROOT/"talent_flow_findings.html"
out.write_text(HTML,encoding="utf-8")
print("wrote",out,f"({len(HTML)/1024/1024:.1f} MB)")
print("top firm:",top_firm,"| charts:",len(FIG))
