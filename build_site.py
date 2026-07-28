"""Build the self-contained interactive findings site (talent_flow_findings.html).

Reads the pipeline outputs in data/processed/flow/ plus the executed notebook,
reconciles duplicate org records, and renders interactive Plotly charts into a
single offline HTML file. Run after `make collect`:

    python build_site.py
"""
import html, re
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
 ('Huawei',('huawei','noah')),('Ant Group',('ant group','ant financial')),('Adobe',('adobe',)),('IBM',('ibm',)),
 ('Samsung',('samsung',)),('Intel',('intel',)),('OpenAI',('openai',)),('Anthropic',('anthropic',)),
 ('DeepSeek',('deepseek',)),('Xiaomi',('xiaomi',)),('JD.com',('jingdong','jd.com')),('Snap',('snapchat','snap inc')),
 ('Moonshot AI',('moonshot','kimi')),('Mistral AI',('mistral',)),('xAI',('xai','x.ai'))]
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

S=dict(talent=len(persons),works=len(works),auth=len(auth),orgs=len(orgs),
       employed=len(employer),trained=len(training),firms=int(gi.shape[0]))

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

FIG={
 'inst':hbar(inst_rank,"scholars trained"),
 'lab':hbar(lab_rank,"trainees"),
 'bucket':hbar(bucket_counts,"scholars"),
 'dest':hbar(top_dest,"scholars hired"),
 'inten':hbar(inten_rank,"intensity (0–100)",fmt="{:.1f}"),
 'labflow':heat(mat,"firm","lab"),
 'country':heat(cm,"employer country","training country"),
 'feed':hbar(feed_rank,"scholars"),
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
 'inten':"The headline metric: each firm's talent mass, weighting every person by citations and publications (log-damped, "
   "normalized within venue family) with a founder bonus, reconciled across duplicate org records and indexed 0–100.",
 'labflow':"Which training labs feed which firms. Hover any cell for the exact count — it reads as a supply chain from elite "
   "labs into the major industrial employers.",
 'country':"Training country → current-employer country. Domestic retention dominates (China→China, US→US), with a pronounced "
   "US↔China cross-border stream.",
 'feed':f"For the #1 firm ({html.escape(top_firm)}), the training institutions that supplied the most of its talent.",
}
def card(key,title):
    return (f'<figure><h3>{html.escape(title)}</h3>{FIG[key]}'
            f'<p class="interp">{INTERP[key]}</p></figure>')

rows="".join(f"<tr><td class='r'>{i+1}</td><td>{html.escape(x.grp)}</td>"
             f"<td class='n'>{x.idx:.1f}</td><td class='n'>{int(x.n)}</td></tr>" for i,x in enumerate(top.itertuples()))
def stat(n,l): return f'<div class="stat"><div class="n">{n}</div><div class="l">{l}</div></div>'

HTML=f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Global AI Talent Flow</title>
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
.tbl{{margin:22px 0 0;background:var(--card);border:1px solid var(--grid);border-radius:18px;overflow:hidden;box-shadow:0 10px 30px rgba(11,11,11,.06)}}
table{{border-collapse:collapse;width:100%;font-size:.96rem}}
th,td{{padding:11px 16px;border-bottom:1px solid var(--grid);text-align:left}}
tr:last-child td{{border-bottom:none}} th{{color:var(--muted);font-weight:600;font-size:.74rem;text-transform:uppercase;letter-spacing:.06em;background:#faf9f6}}
td.r{{color:var(--muted);width:38px;font-variant-numeric:tabular-nums}} td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}} td:nth-child(2){{font-weight:600}}
.caveat{{margin-top:22px;background:var(--amber-bg);border:1px solid var(--amber-bd);border-radius:16px;padding:20px 24px}}
.caveat h3{{margin:.1em 0 .5em;font-size:1.1rem}} .caveat li{{margin:.3em 0;color:#5a4a2a}}
footer{{color:var(--muted);font-size:.85rem;margin-top:54px;border-top:1px solid var(--grid);padding-top:22px}}
</style></head><body>
<section class="hero"><div class="wrap">
<div class="eyebrow">Global AI Talent · person-centric bibliometric study</div>
<h1>Where the world's <span class="hl">AI talent</span> trains — and where it goes</h1>
<p class="sub">A lab- and company-level map of elite AI researchers, reconstructed from public scholarly records.</p>
<div class="stats">{stat(f"{S['talent']:,}","scholars identified")}{stat(f"{S['works']:,}","papers scanned")}{stat(f"{S['orgs']:,}","organizations mapped")}</div>
<p class="method">We pull from <b>five public sources</b> (DBLP, CSRankings, Semantic Scholar, ORCID, Wikidata) to identify every author
at <b>nine top AI venues in 2024–2025</b>, <b>backtrack</b> each scholar's full publication history to their doctoral-era lab and
advisor, and <b>map</b> affiliations to a curated organization registry — tracing where talent trained and where it went.</p>
</div></section>
<main class="wrap">
<section class="sec"><div class="sec-head"><div class="badge">1</div><h2>How firms rank on AI-talent intensity</h2></div>
<p class="lead">The headline result — citation- and publication-weighted talent mass per firm, indexed 0–100.</p>
{card('inten','AI Talent Intensity Index')}
<div class="tbl"><table><thead><tr><th class="r">#</th><th>Firm</th><th class="n">Intensity</th><th class="n">Scholars</th></tr></thead><tbody>{rows}</tbody></table></div>
</section>
<section class="sec"><div class="sec-head"><div class="badge">2</div><h2>The top AI labs in the world</h2></div>
<p class="lead">Where talent is produced — by institution, then by individual lab (institution + advisor).</p>
{card('inst','Top training institutions')}{card('lab','Top labs (institution + advisor)')}
</section>
<section class="sec"><div class="sec-head"><div class="badge">3</div><h2>Where AI talent went</h2></div>
<p class="lead">Most researchers stay in academia; the flow into industry is the smaller, high-value stream.</p>
{card('bucket','Where talent sits now')}{card('dest','Top industry destinations')}
</section>
<section class="sec"><div class="sec-head"><div class="badge">4</div><h2>Flows &amp; pipelines</h2></div>
<p class="lead">From training labs into firms, and across borders. Hover any cell for exact counts.</p>
{card('country','Cross-border talent flow')}{card('feed','Which labs feed the #1 firm')}
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
</main></body></html>"""

out=ROOT/"talent_flow_findings.html"
out.write_text(HTML,encoding="utf-8")
print("wrote",out,f"({len(HTML)/1024/1024:.1f} MB)")
print("top firm:",top_firm,"| charts:",len(FIG))
