import os
import math
import json
import tempfile
import urllib.parse
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

try:
    import paramiko
    from scp import SCPClient as _SCPClient
    _PARAMIKO = True
except ImportError:
    _PARAMIKO = False

st.set_page_config(
    page_title="Molecular Energy Landscape",
    layout="wide",
    initial_sidebar_state="expanded",
)

_BASE = os.path.dirname(os.path.abspath(__file__))
_DEFAULTS = {
    "min.data":  os.path.join(_BASE, "Results", "min.data"),
    "ts.data":   os.path.join(_BASE, "Results", "ts.data"),
    "path.info": os.path.join(_BASE, "Results", "path.info"),
}


# ── File reading ──────────────────────────────────────────────────────────────

def read_files(uploaded_list, key, folder="0"):
    """Return list of (filename, text) from uploads, SSH fetch, or the default file."""
    results = []
    for f in (uploaded_list or []):
        raw = f.read()
        try:
            results.append((f.name, raw.decode("utf-8")))
        except UnicodeDecodeError:
            st.error(
                f"**{f.name}** could not be read — it appears to be a binary file "
                f"(e.g. a Word document). `{key}` must be a plain text file."
            )
    ssh_files = st.session_state.get(f"ssh_files_{folder}", {})
    if key in ssh_files:
        results.append(ssh_files[key])
    for name, text in st.session_state.get(f"saved_session_{folder}", {}).get(key, []):
        results.append((name, text))
    if not results:
        path = _DEFAULTS.get(key)
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                results.append((os.path.basename(path), fh.read()))
    return results


# ── Parsers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def parse_min(text):
    result = []
    for line in text.splitlines():
        p = line.split()
        if p:
            try:
                result.append((float(p[0]), line))
            except ValueError:
                pass
    return result   # list of (energy, original_line)


@st.cache_data(show_spinner=False)
def parse_ts(text):
    entries = []
    for line in text.splitlines():
        p = line.split()
        if len(p) >= 5:
            try:
                entries.append((float(p[0]), int(p[3]), int(p[4]), p))
            except ValueError:
                pass
    return entries   # list of (energy, min_a_idx, min_b_idx, all_parts)


@st.cache_data(show_spinner=False)
def parse_path(text):
    lines = text.splitlines()
    structures = []
    i = 0
    while i < len(lines):
        raw = lines[i].strip()
        if not raw:
            i += 1
            continue
        try:
            energy = float(raw)
        except ValueError:
            i += 1
            continue
        i += 1
        if i < len(lines):
            i += 1          # skip symmetry line
        coords = []
        while i < len(lines) and len(coords) < 100:
            parts = lines[i].split()
            if len(parts) == 3:
                try:
                    coords.append([float(x) for x in parts])
                    i += 1
                except ValueError:
                    break
            else:
                break
        if coords:
            structures.append({"e": energy, "c": coords})
    return [
        (structures[k], structures[k + 1], structures[k + 2])
        for k in range(0, len(structures) - 2, 3)
    ]


# ── Merge and deduplication ───────────────────────────────────────────────────

def merge_min(file_texts):
    """Combine min.data from multiple files; dedup by full-precision energy."""
    seen = {}       # energy -> (energy, original_line)
    stats = []
    for fname, text in file_texts:
        entries = parse_min(text)   # list of (energy, line)
        added = dup = 0
        for e, line in entries:
            if e not in seen:
                seen[e] = (e, line)
                added += 1
            else:
                dup += 1
        stats.append({"file": fname, "entries": len(entries),
                      "new": added, "duplicates removed": dup})
    pairs = list(seen.values())
    energies = [e for e, _ in pairs]
    lines    = [ln for _, ln in pairs]
    return energies, lines, stats


def merge_ts(ts_file_texts, min_file_texts):
    """Combine ts.data from multiple files; dedup by TS energy.

    ts.data stores connections as integer node indices that are local to the
    min.data file they were computed alongside. This function converts those
    indices to actual energies by pairing each ts.data with its corresponding
    min.data (matched by upload order; the last min.data is reused for extras).
    build_network then re-maps those energies to the merged node set using
    nearest-energy matching, exactly as it does for path.info triplets.
    """
    min_parsed = [parse_min(text) for _, text in min_file_texts]
    if not min_parsed:
        return [], [], []

    seen = set()
    merged = []     # (ts_energy, min_a_energy, min_b_energy, all_parts)
    stats = []
    for i, (fname, text) in enumerate(ts_file_texts):
        local_min = min_parsed[min(i, len(min_parsed) - 1)]  # list of (energy, line)
        raw = parse_ts(text)
        added = dup = 0
        for ts_e_val, raw_a, raw_b, parts in raw:
            if not (1 <= raw_a <= len(local_min) and 1 <= raw_b <= len(local_min)):
                continue
            ea, eb = local_min[raw_a - 1][0], local_min[raw_b - 1][0]
            if ts_e_val not in seen:
                seen.add(ts_e_val)
                merged.append((ts_e_val, ea, eb, parts))
                added += 1
            else:
                dup += 1
        stats.append({"file": fname, "entries": len(raw),
                      "new": added, "duplicates removed": dup})
    ts_tuples = [(e, ea, eb) for e, ea, eb, _ in merged]
    ts_parts  = [p            for _, _,  _,  p in merged]
    return ts_tuples, ts_parts, stats


def merge_path(file_texts):
    """Combine path.info from multiple files; dedup triplets by TS energy."""
    seen = set()
    merged = []
    stats = []
    for fname, text in file_texts:
        triplets = parse_path(text)
        added = dup = 0
        for mA, ts, mB in triplets:
            if ts["e"] not in seen:
                seen.add(ts["e"])
                merged.append((mA, ts, mB))
                added += 1
            else:
                dup += 1
        stats.append({"file": fname, "entries": len(triplets),
                      "new": added, "duplicates removed": dup})
    return merged, stats


# ── Network construction ──────────────────────────────────────────────────────

_MATCH_TOL = 5e-14  # half a unit in the 13th decimal place
_RMSD_TOL  = 1e-4   # Ångströms — only catches truly identical rotated structures


def _kabsch_rmsd(coords_a, coords_b):
    """RMSD between two coordinate sets after optimal rotation alignment (Kabsch).

    Returns infinity if the atom counts differ.
    """
    if len(coords_a) != len(coords_b):
        return float("inf")
    A = np.array(coords_a, dtype=float)
    B = np.array(coords_b, dtype=float)
    A -= A.mean(axis=0)
    B -= B.mean(axis=0)
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    # Correct for improper rotation (reflection)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = (A @ R.T) - B
    return float(np.sqrt((diff ** 2).sum() / len(A)))


def build_network(min_e_base, ts_e, triplets):
    """Build the edge list, TS node list, and node table from all available data.

    Nodes are seeded from min.data (min_e_base). Each path.info structure is
    matched first by energy (_MATCH_TOL), then by RMSD after optimal rotation
    (_RMSD_TOL). Unmatched structures become new nodes.
    Returns (edges, node_coords, all_e, ts_nodes, n_rmsd_merged).
    """
    all_e = list(min_e_base)
    node_coords = {}
    rmsd_count = [0]

    def find_match(e):
        if not all_e:
            return None
        best = min(range(len(all_e)), key=lambda j: abs(all_e[j] - e))
        return best + 1 if abs(all_e[best] - e) <= _MATCH_TOL else None

    def find_rmsd_match(coords):
        for nid, nc in node_coords.items():
            if _kabsch_rmsd(nc, coords) <= _RMSD_TOL:
                return nid
        return None

    def get_or_create(struct):
        nid = find_match(struct["e"])
        if nid is None and struct.get("c"):
            nid = find_rmsd_match(struct["c"])
            if nid is not None:
                rmsd_count[0] += 1
        if nid is None:
            all_e.append(struct["e"])
            nid = len(all_e)
        if struct.get("c") and nid not in node_coords:
            node_coords[nid] = struct["c"]
        return nid

    # Collect all raw TS entries from both sources
    raw_ts = []
    for mA, ts, mB in triplets:
        idA = get_or_create(mA)
        idB = get_or_create(mB)
        raw_ts.append((ts["e"], idA, idB, ts.get("c"), "path.info"))

    for ts_energy, ea, eb in ts_e:
        idA = find_match(ea)
        idB = find_match(eb)
        if idA is None or idB is None:
            continue
        raw_ts.append((ts_energy, idA, idB, None, "ts.data only"))

    # Group TS entries by energy within _MATCH_TOL → one ts_node per unique TS
    ts_nodes = []

    def find_ts_group(e):
        for i, grp in enumerate(ts_nodes):
            if abs(grp["e"] - e) <= _MATCH_TOL:
                return i
        return None

    for ts_e_val, idA, idB, coords, source in raw_ts:
        pair = (min(idA, idB), max(idA, idB))
        gi = find_ts_group(ts_e_val)
        if gi is None:
            ts_nodes.append({"e": ts_e_val, "c": coords, "source": source,
                              "pairs": [pair]})
        else:
            if pair not in ts_nodes[gi]["pairs"]:
                ts_nodes[gi]["pairs"].append(pair)
            if coords and not ts_nodes[gi]["c"]:
                ts_nodes[gi]["c"] = coords
            if source == "path.info":
                ts_nodes[gi]["source"] = "path.info"

    # Flat edge list for energy profiles and summary counts
    edges = []
    for tsn in ts_nodes:
        for pair in tsn["pairs"]:
            edges.append({"idA": pair[0], "idB": pair[1],
                          "e": tsn["e"], "c": tsn["c"], "source": tsn["source"]})

    return edges, node_coords, all_e, ts_nodes, rmsd_count[0]


# ── Layout ────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def find_longest_chain(edge_tuples, n_nodes):
    """Return the longest simple path in the graph as a tuple of node IDs.

    Uses DFS with a call-count guard so it never hangs on dense graphs.
    """
    if not edge_tuples:
        return (1,) if n_nodes >= 1 else ()

    adj = {}
    for a, b in edge_tuples:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    best = []
    calls = [0]

    def dfs(node, vis, path):
        calls[0] += 1
        if calls[0] > 200_000:
            return
        if len(path) > len(best):
            best[:] = path[:]
        for nb in adj.get(node, []):
            if nb not in vis:
                vis.add(nb)
                path.append(nb)
                dfs(nb, vis, path)
                path.pop()
                vis.remove(nb)

    for start in sorted(adj.keys()):
        if calls[0] > 200_000:
            break
        dfs(start, {start}, [start])

    return tuple(best) if best else (1,)


@st.cache_data(show_spinner=False)
def positions(n_nodes, spine):
    """Spine nodes centred at x=470, evenly spaced top-to-bottom.
    Remaining nodes arranged in ring pairs around the centre.
    """
    cx, cy, r = 470, 470, 310
    pos = {}

    # Place spine along the vertical centre
    k = len(spine)
    y_top, y_bot = 70, 870
    for i, node_id in enumerate(spine):
        y = y_top if k == 1 else y_top + i * (y_bot - y_top) / (k - 1)
        pos[node_id] = (cx, round(y, 2))

    # Remaining nodes in ring pairs
    ring_ids = [i for i in range(1, n_nodes + 1) if i not in pos]
    n_pairs  = math.ceil(len(ring_ids) / 2)

    for p_idx in range(n_pairs):
        ang  = math.radians(p_idx * (360 / max(n_pairs, 1)))
        pcx  = cx + r * math.cos(ang)
        pcy  = cy + r * math.sin(ang)
        ox   = -math.sin(ang) * 22
        oy   =  math.cos(ang) * 22
        a    = ring_ids[2 * p_idx]
        pos[a] = (round(pcx - ox, 2), round(pcy - oy, 2))
        if 2 * p_idx + 1 < len(ring_ids):
            b      = ring_ids[2 * p_idx + 1]
            pos[b] = (round(pcx + ox, 2), round(pcy + oy, 2))

    return pos


# ── Colours ───────────────────────────────────────────────────────────────────

def node_col(e, lo, hi):
    t = (e - lo) / max(hi - lo, 1e-9)
    # Five-stop spectrum: blue → cyan → green → yellow → red
    stops = [
        (0.00, (0,   60,  220)),
        (0.25, (0,   210, 220)),
        (0.50, (0,   200,  50)),
        (0.75, (220, 210,   0)),
        (1.00, (220,   0,   0)),
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1 or i == len(stops) - 2:
            s = (t - t0) / max(t1 - t0, 1e-9)
            r = int(c0[0] + s * (c1[0] - c0[0]))
            g = int(c0[1] + s * (c1[1] - c0[1]))
            b = int(c0[2] + s * (c1[2] - c0[2]))
            return f"rgb({r},{g},{b})"
    return "rgb(220,0,0)"


def edge_col(e, lo, hi, source):
    t = (e - lo) / max(hi - lo, 1e-9)
    if source == "ts.data only":
        return f"rgb(100,{int(180*(1-t))},100)"
    return f"rgb(180,{int(200*(1-t))},180)"


# ── 3D molecular viewer ───────────────────────────────────────────────────────

def build_3d_viewer_html(coords, element, height=300):
    """Return a self-contained HTML page with a 3Dmol.js viewer.

    coords  — list of [x, y, z] floats
    element — chemical symbol to label every atom (e.g. "Au", "C")
    """
    if not coords:
        return "<p style='color:#888;font-family:sans-serif'>No coordinates.</p>"
    xyz_lines = [str(len(coords)), "structure"]
    for x, y, z in coords:
        xyz_lines.append(f"{element} {x:.6f} {y:.6f} {z:.6f}")
    xyz_block = "\n".join(xyz_lines).replace("\\", "\\\\").replace("`", "\\`")
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.3/3Dmol-min.js"></script>
</head>
<body style="margin:0;background:#0f0f1a">
<div id="v" style="width:100%;height:{height}px;position:relative"></div>
<script>
(function(){{
  if(typeof $3Dmol==="undefined"){{
    document.getElementById("v").innerHTML=
      "<p style='color:#aaa;padding:12px;font-family:sans-serif'>" +
      "3Dmol.js could not load — check your internet connection.</p>";
    return;
  }}
  let v=$3Dmol.createViewer("v",{{backgroundColor:"#0f0f1a"}});
  v.addModel(`{xyz_block}`,"xyz");
  v.setStyle({{}},{{sphere:{{radius:0.35,colorscheme:"Jmol"}}}});
  v.zoomTo();v.render();
}})();
</script>
</body></html>"""


# ── Draggable SVG diagram ─────────────────────────────────────────────────────

def build_draggable_html(min_e, ts_nodes, node_coords, pos):
    """Return a self-contained HTML page: SVG diagram with drag-and-drop nodes.

    Transition states are rendered as diamonds at the centroid of their connected
    minima. Lines radiate from each diamond to every minimum it connects. Identical
    TS from different sources share one diamond, so lines from all their minima
    converge on the same point.
    """
    e_lo, e_hi = min(min_e), max(min_e)
    ts_e_all = [tsn["e"] for tsn in ts_nodes]
    te_lo = min(ts_e_all, default=0.0)
    te_hi = max(ts_e_all, default=0.0)

    nodes_data = []
    for idx, e in enumerate(min_e, start=1):
        if idx not in pos:
            continue
        x, y = pos[idx]
        nodes_data.append({
            "id": idx,
            "x": round(x, 2),
            "y": round(y, 2),
            "fill": node_col(e, e_lo, e_hi),
            "stroke": "#00ff00" if idx in node_coords else "white",
            "energy": f"{e:.14f}",
            "hasCoords": idx in node_coords,
        })

    ts_data = []
    for tsn in ts_nodes:
        valid_pairs = [[a, b] for a, b in tsn["pairs"] if a in pos and b in pos]
        if not valid_pairs:
            continue
        ts_data.append({
            "e": f"{tsn['e']:.14f}",
            "source": tsn["source"],
            "color": edge_col(tsn["e"], te_lo, te_hi, tsn["source"]),
            "pairs": valid_pairs,
        })

    template = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f1a;overflow:hidden;font-family:sans-serif}
#bar{position:absolute;top:6px;right:8px;display:flex;gap:8px;align-items:center;z-index:10}
#bar button{background:#1a1a2e;color:#ccc;border:1px solid #555;padding:3px 12px;
  border-radius:4px;cursor:pointer;font-size:12px;letter-spacing:.3px}
#bar button:hover{background:#2a2a4e;color:#fff}
#bar label{color:#999;font-size:12px;user-select:none;cursor:pointer}
#tip{position:fixed;display:none;background:rgba(0,0,0,.9);color:#e0e0e0;
  border:1px solid #555;border-radius:4px;padding:6px 10px;font-size:11px;
  pointer-events:none;line-height:1.7;z-index:50;max-width:300px}
svg{display:block;width:100%;height:700px}
</style></head>
<body>
<div id="bar">
  <button id="rst">RESET</button>
  <label><input type="checkbox" id="lbl">&nbsp;TS labels</label>
</div>
<div id="tip"></div>
<svg id="g" viewBox="0 0 940 940" preserveAspectRatio="xMidYMid meet">
  <g id="eG"></g><g id="lG"></g><g id="tG"></g><g id="nG"></g>
</svg>
<script>
const NS='http://www.w3.org/2000/svg';
const NODES=__NODES__;
const TS=__TS__;
const orig={};NODES.forEach(n=>orig[n.id]={x:n.x,y:n.y});
const P={};NODES.forEach(n=>P[n.id]={x:n.x,y:n.y});
const svg=document.getElementById('g');
const eG=document.getElementById('eG'),lG=document.getElementById('lG');
const tG=document.getElementById('tG'),nG=document.getElementById('nG');
const tip=document.getElementById('tip');
let showL=false,drag=null,ds=null;

// Build TS elements: one line per connected minimum + a diamond at the centroid
const tsEls=[];
TS.forEach(ts=>{
  const uMins=new Set();
  ts.pairs.forEach(p=>{uMins.add(p[0]);uMins.add(p[1]);});
  const pStr=ts.pairs.map(p=>p[0]+' ↔ '+p[1]).join(', ');
  const tipTxt='Transition state\nEnergy: '+ts.e+'\nConnects: '+pStr+'\nSource: '+ts.source;
  const lines=[];
  uMins.forEach(mid=>{
    const ln=document.createElementNS(NS,'line');
    ln.setAttribute('stroke',ts.color);ln.setAttribute('stroke-width','1.8');
    eG.appendChild(ln);
    const hit=document.createElementNS(NS,'line');
    hit.setAttribute('stroke','transparent');hit.setAttribute('stroke-width','14');
    hit.addEventListener('mouseenter',ev=>showTip(ev,tipTxt));
    hit.addEventListener('mousemove',moveTip);hit.addEventListener('mouseleave',hideTip);
    eG.appendChild(hit);
    lines.push({ln,hit,mid});
  });
  const tx=document.createElementNS(NS,'text');
  tx.setAttribute('font-size','8');tx.setAttribute('fill','#666');
  tx.setAttribute('text-anchor','middle');tx.setAttribute('dominant-baseline','middle');
  tx.setAttribute('pointer-events','none');
  tx.textContent=parseFloat(ts.e).toExponential(3);
  lG.appendChild(tx);
  const poly=document.createElementNS(NS,'polygon');
  poly.setAttribute('fill',ts.color);poly.setAttribute('stroke','white');
  poly.setAttribute('stroke-width','1.5');
  tG.appendChild(poly);
  const hitP=document.createElementNS(NS,'polygon');
  hitP.setAttribute('fill','transparent');hitP.setAttribute('stroke','transparent');
  hitP.addEventListener('mouseenter',ev=>showTip(ev,tipTxt));
  hitP.addEventListener('mousemove',moveTip);hitP.addEventListener('mouseleave',hideTip);
  tG.appendChild(hitP);
  tsEls.push({lines,poly,hitP,tx});
});

// Build min node elements
const nEls={};
NODES.forEach(n=>{
  const g=document.createElementNS(NS,'g');g.style.cursor='grab';
  const c=document.createElementNS(NS,'circle');
  c.setAttribute('r','15');c.setAttribute('fill',n.fill);
  c.setAttribute('stroke',n.stroke);c.setAttribute('stroke-width','2');
  const t=document.createElementNS(NS,'text');
  t.setAttribute('font-size','9');t.setAttribute('fill','white');
  t.setAttribute('text-anchor','middle');t.setAttribute('dominant-baseline','middle');
  t.setAttribute('pointer-events','none');t.textContent=n.id;
  g.appendChild(c);g.appendChild(t);
  g.addEventListener('mouseenter',ev=>showTip(ev,'Node '+n.id+'\nEnergy: '+n.energy+(n.hasCoords?'\nCoordinates available':'')));
  g.addEventListener('mousemove',moveTip);g.addEventListener('mouseleave',hideTip);
  g.addEventListener('mousedown',ev=>startDrag(ev,n.id));
  g.addEventListener('touchstart',ev=>{ev.preventDefault();startDrag(ev.touches[0],n.id);},{passive:false});
  nG.appendChild(g);nEls[n.id]=g;
});

function tsCenter(ts){
  const uMins=new Set();ts.pairs.forEach(p=>{uMins.add(p[0]);uMins.add(p[1]);});
  let sx=0,sy=0,n=0;
  uMins.forEach(mid=>{if(P[mid]){sx+=P[mid].x;sy+=P[mid].y;n++;}});
  return n>0?{x:sx/n,y:sy/n}:null;
}
function dPts(cx,cy,s){
  return cx+','+(cy-s)+' '+(cx+s)+','+cy+' '+cx+','+(cy+s)+' '+(cx-s)+','+cy;
}

function redraw(){
  TS.forEach((ts,i)=>{
    const tp=tsCenter(ts);const el=tsEls[i];if(!tp)return;
    el.lines.forEach(({ln,hit,mid})=>{
      const mp=P[mid];if(!mp)return;
      ln.setAttribute('x1',mp.x);ln.setAttribute('y1',mp.y);
      ln.setAttribute('x2',tp.x);ln.setAttribute('y2',tp.y);
      hit.setAttribute('x1',mp.x);hit.setAttribute('y1',mp.y);
      hit.setAttribute('x2',tp.x);hit.setAttribute('y2',tp.y);
    });
    el.tx.setAttribute('x',tp.x);el.tx.setAttribute('y',tp.y-14);
    el.tx.style.display=showL?'':'none';
    el.poly.setAttribute('points',dPts(tp.x,tp.y,8));
    el.hitP.setAttribute('points',dPts(tp.x,tp.y,14));
  });
  NODES.forEach(n=>{
    const p=P[n.id];if(!p)return;
    nEls[n.id].setAttribute('transform','translate('+p.x+','+p.y+')');
  });
}
redraw();

function svgPt(e){
  const pt=svg.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}
function startDrag(e,id){
  if(e.preventDefault)e.preventDefault();
  drag=id;const p=svgPt(e);
  ds={sx:p.x,sy:p.y,ox:P[id].x,oy:P[id].y};
  nEls[id].style.cursor='grabbing';hideTip();
}
function onMove(e){
  if(!drag)return;
  const cl=e.touches?e.touches[0]:e;
  const p=svgPt(cl);
  P[drag].x=ds.ox+(p.x-ds.sx);
  P[drag].y=ds.oy+(p.y-ds.sy);
  redraw();
}
function onUp(){if(drag){nEls[drag].style.cursor='grab';drag=null;}}
svg.addEventListener('mousemove',onMove);
svg.addEventListener('mouseup',onUp);
svg.addEventListener('mouseleave',onUp);
svg.addEventListener('touchmove',e=>{e.preventDefault();onMove(e);},{passive:false});
svg.addEventListener('touchend',onUp);

document.getElementById('rst').addEventListener('click',()=>{
  NODES.forEach(n=>{P[n.id]={x:orig[n.id].x,y:orig[n.id].y};});redraw();
});
document.getElementById('lbl').addEventListener('change',e=>{showL=e.target.checked;redraw();});

function showTip(e,txt){
  tip.innerHTML=txt.replace(/\n/g,'<br>');tip.style.display='block';moveTip(e);
}
function moveTip(e){tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-10)+'px';}
function hideTip(){tip.style.display='none';}
</script></body></html>"""

    return (template
            .replace("__NODES__", json.dumps(nodes_data))
            .replace("__TS__", json.dumps(ts_data)))


# ── Energy profile line charts ───────────────────────────────────────────────

def find_component_paths(edges, min_e):
    """Return (components, isolated) where each component is
    (comp_nodes, all_paths, truncated_bool)."""
    adj = {}
    for ed in edges:
        a, b, e = ed["idA"], ed["idB"], ed["e"]
        adj.setdefault(a, []).append((b, e))
        adj.setdefault(b, []).append((a, e))

    all_nodes = set(range(1, len(min_e) + 1))
    visited_g = set()
    raw_components = []
    for start in sorted(all_nodes):
        if start in visited_g:
            continue
        comp = set()
        queue = [start]
        while queue:
            n = queue.pop(0)
            if n in comp:
                continue
            comp.add(n)
            for nb, _ in adj.get(n, []):
                queue.append(nb)
        visited_g.update(comp)
        raw_components.append(sorted(comp))

    def dfs_paths(root):
        result = []
        def dfs(node, vis, path):
            if len(result) >= 100:
                return
            unvis = sorted(
                [(n, e) for n, e in adj.get(node, []) if n not in vis],
                key=lambda x: x[0],
            )
            if not unvis:
                result.append(list(path))
                return
            for nb, ts_e in unvis:
                vis.add(nb)
                path.append(("T", ts_e, node, nb))
                path.append(("M", min_e[nb - 1], nb))
                dfs(nb, vis, path)
                path.pop(); path.pop()
                vis.remove(nb)
        dfs(root, {root}, [("M", min_e[root - 1], root)])
        return result or [[("M", min_e[root - 1], root)]]

    components = []
    isolated = []
    for comp_nodes in raw_components:
        if len(comp_nodes) == 1:
            isolated.append(comp_nodes[0])
            continue
        paths = dfs_paths(comp_nodes[0])
        components.append((comp_nodes, paths, len(paths) >= 100))
    return components, isolated


def build_single_path_fig(path, comp_nodes, all_min_e):
    """Build a Plotly energy profile for one chain.

    Minima are coloured with the same blue-to-red gradient as the spider web.
    TS are shown as grey diamonds. A spline connects all points.
    """
    e_lo, e_hi = min(all_min_e), max(all_min_e)
    y_vals = [item[1] for item in path]
    x_vals = list(range(len(path)))
    colors, symbols, sizes, hover = [], [], [], []
    for item in path:
        if item[0] == "M":
            colors.append(node_col(item[1], e_lo, e_hi))
            symbols.append("circle")
            sizes.append(14)
            hover.append(f"Node {item[2]}<br>E = {item[1]:.14f}")
        else:
            colors.append("#999999")
            symbols.append("diamond")
            sizes.append(9)
            hover.append(f"TS {item[2]} ↔ {item[3]}<br>E = {item[1]:.14f}")
    tick_text = [f"Node {p[2]}" if p[0] == "M" else "TS" for p in path]
    node_list = " · ".join(str(n) for n in comp_nodes[:12])
    if len(comp_nodes) > 12:
        node_list += " …"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="lines",
        showlegend=False,
        line=dict(color="#2a3a5e", width=2, shape="spline", smoothing=0.6),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers",
        showlegend=False,
        marker=dict(
            size=sizes,
            symbol=symbols,
            color=colors,
            line=dict(color="white", width=1),
        ),
        hovertext=hover,
        hoverinfo="text",
    ))
    fig.update_layout(
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0a0a18",
        font=dict(color="#e0e0e0"),
        title=dict(text=f"Nodes: {node_list}", font=dict(size=12, color="#667799")),
        xaxis=dict(
            title="",
            tickmode="array",
            tickvals=list(range(len(tick_text))),
            ticktext=tick_text,
            tickangle=45,
            gridcolor="#1a1a30",
            showgrid=True,
            tickfont=dict(size=10, color="#667799"),
            zeroline=False,
        ),
        yaxis=dict(
            title="Energy",
            gridcolor="#1a1a30",
            showgrid=True,
            zeroline=False,
        ),
        margin=dict(l=70, r=20, t=50, b=90),
        height=420,
        hovermode="closest",
    )
    return fig


# ── Merge stats helper ────────────────────────────────────────────────────────

def show_merge_stats(label, stats, n_unique):
    if not stats:
        return
    total_dups = sum(s["duplicates removed"] for s in stats)
    st.markdown(f"**{label}**: {len(stats)} file(s) · {n_unique} unique")
    if total_dups:
        st.caption(f"{total_dups} duplicate(s) removed")
    if len(stats) > 1:
        for s in stats:
            st.caption(f"  {s['file']}: {s['new']} new, {s['duplicates removed']} dup")


# ── App ───────────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar: file uploads ──
    if "show_guide" not in st.session_state:
        st.session_state["show_guide"] = False

    with st.sidebar:
        guide_open = st.session_state["show_guide"]
        if st.button(
            "Close Guide" if guide_open else "User Guide",
            use_container_width=True,
        ):
            st.session_state["show_guide"] = not guide_open
            st.rerun()

        st.divider()
        st.title("Compress")
        folder = st.selectbox(
            "Subfolder",
            options=["0", "-1", "-2", "-3", "-4", "-5"],
            key="active_folder",
        )
        st.divider()

        # Session restore
        with st.expander("Load saved session"):
            up_session = st.file_uploader(
                "Session file (.json)", type=["json"], key=f"up_session_{folder}"
            )
            if up_session is not None:
                try:
                    data = json.loads(up_session.read().decode("utf-8"))
                    if data.get("version") == 1:
                        st.session_state[f"saved_session_{folder}"] = {
                            k: [(n, t) for n, t in v]
                            for k, v in data.get("files", {}).items()
                        }
                        st.success("Session restored.")
                    else:
                        st.error("Unrecognised session file format.")
                except Exception as exc:
                    st.error(f"Could not load session: {exc}")
            if f"saved_session_{folder}" in st.session_state:
                counts = {
                    k: len(v)
                    for k, v in st.session_state[f"saved_session_{folder}"].items()
                }
                st.caption(
                    "Loaded: "
                    + ", ".join(f"{k} ({n})" for k, n in counts.items())
                )
                if st.button("Clear saved session", key=f"btn_clear_session_{folder}"):
                    st.session_state.pop(f"saved_session_{folder}", None)
                    st.rerun()

        st.divider()
        st.caption("Each uploader accepts multiple files. Duplicates are removed automatically.")
        up_min  = st.file_uploader("min.data",  accept_multiple_files=True, key=f"up_min_{folder}")
        up_ts   = st.file_uploader("ts.data",   accept_multiple_files=True, key=f"up_ts_{folder}")
        up_path = st.file_uploader("path.info", accept_multiple_files=True, key=f"up_path_{folder}")

        st.divider()
        st.markdown("**Copy from Linux server (SCP)**")
        if not _PARAMIKO:
            st.warning("Run `pip install paramiko scp` to enable SCP fetching.")
        else:
            with st.expander("SCP connection"):
                ssh_host = st.text_input("Host",     key="ssh_host")
                ssh_port = st.number_input("Port",   key="ssh_port", value=22,
                                           min_value=1, max_value=65535, step=1)
                ssh_user = st.text_input("Username", key="ssh_user")
                ssh_pass = st.text_input("Password", key="ssh_pass", type="password")
                ssh_mfa  = st.text_input("MFA code (Microsoft Authenticator)", key="ssh_mfa",
                                         placeholder="6-digit code — enter just before clicking Fetch")
                ssh_key  = st.text_input("Key file (optional)", key="ssh_key",
                                         placeholder="~/.ssh/id_rsa")
                ssh_dir  = st.text_input("Results directory", key="ssh_dir",
                                         placeholder="~/7_1_op/optimization  or full path")

                col_fetch, col_clear = st.columns(2)
                fetch_clicked = col_fetch.button("Fetch", use_container_width=True)
                clear_clicked = col_clear.button("Clear", use_container_width=True)

                if clear_clicked:
                    st.session_state.pop(f"ssh_files_{folder}", None)
                    st.rerun()

                if fetch_clicked and ssh_host and ssh_user and ssh_dir:
                    with st.spinner("Connecting…"):
                        try:
                            t = paramiko.Transport((ssh_host, int(ssh_port)))
                            t.start_client(timeout=10)
                            if ssh_key:
                                pkey = paramiko.RSAKey.from_private_key_file(
                                    os.path.expanduser(ssh_key)
                                )
                                t.auth_publickey(ssh_user, pkey)
                            else:
                                def _kbd(_title, _instructions, prompt_list):
                                    responses = []
                                    for prompt_text, _ in prompt_list:
                                        low = prompt_text.lower()
                                        if any(kw in low for kw in
                                               ["code", "otp", "token", "verification",
                                                "authenticator", "mfa", "2fa"]):
                                            responses.append(ssh_mfa)
                                        else:
                                            responses.append(ssh_pass)
                                    return responses
                                t.auth_interactive(ssh_user, _kbd)
                            # Resolve ~ to the actual home directory on the server
                            resolved_dir = ssh_dir.strip()
                            if resolved_dir.startswith("~"):
                                chan = t.open_session()
                                chan.exec_command("echo $HOME")
                                home = chan.makefile().read().decode().strip()
                                chan.recv_exit_status()
                                chan.close()
                                resolved_dir = home + resolved_dir[1:]
                            fetched = {}
                            file_errors = {}
                            with _SCPClient(t) as scp:
                                for fname in ["min.data", "ts.data", "path.info"]:
                                    remote = f"{resolved_dir.rstrip('/')}/{fname}"
                                    tmp = None
                                    try:
                                        fd, tmp = tempfile.mkstemp()
                                        os.close(fd)
                                        scp.get(remote, tmp)
                                        with open(tmp, encoding="utf-8") as fh:
                                            text = fh.read()
                                        fetched[fname] = (f"{ssh_host}:{fname}", text)
                                    except Exception as fe:
                                        file_errors[fname] = str(fe)
                                    finally:
                                        if tmp and os.path.exists(tmp):
                                            os.unlink(tmp)
                            t.close()
                            st.session_state[f"ssh_files_{folder}"] = fetched
                            if fetched:
                                for fname in fetched:
                                    st.success(f"{fname} has been uploaded successfully.")
                            else:
                                st.error("Nothing fetched — check the directory path below.")
                            for fname, err in file_errors.items():
                                st.warning(f"{fname}: {err}")
                        except Exception as exc:
                            st.error(f"Connection failed: {exc}")

                if f"ssh_files_{folder}" in st.session_state:
                    found = list(st.session_state[f"ssh_files_{folder}"])
                    st.caption(f"Loaded from server: {', '.join(found)}")

        st.divider()
        element = st.text_input(
            "Atom element for 3D viewer",
            value="Au",
            max_chars=3,
            help="Chemical symbol applied to every atom when building the 3D view. "
                 "Change this to match your molecule (e.g. C, N, Pt).",
            key="element_input",
        ).strip() or "C"

        st.divider()
        st.markdown("**Colour key**")
        st.markdown(
            """
<div style='margin-bottom:4px'>
  <div style='
    background:linear-gradient(to right,
      rgb(0,60,220),
      rgb(0,210,220),
      rgb(0,200,50),
      rgb(220,210,0),
      rgb(220,0,0));
    height:12px;border-radius:4px;
  '></div>
  <div style='display:flex;justify-content:space-between;
              font-size:11px;color:#aaa;margin-top:3px'>
    <span>Stable</span><span>Metastable</span>
  </div>
</div>
<div style='font-size:12px;color:#ccc;margin-top:6px'>
  🟢 Green border = path.info coordinates
</div>
""",
            unsafe_allow_html=True,
        )

    # ── User Guide overlay ──
    if st.session_state.get("show_guide", False):
        st.subheader("User Guide")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Loading your files")
            st.markdown(
                """
**min.data** and **ts.data** can be loaded in two ways:
- Copy them from the Linux server using the **SCP connection** panel in the sidebar.
- Copy the file contents into a text file (e.g. Notepad) and upload it using the file uploaders.

**path.info** should be copied directly from the Linux server using SCP.
The file is too large to copy and paste by hand without risk of truncation.

You can upload more than one file into each slot. Duplicates are removed automatically.
"""
            )
            st.markdown("### The spider web diagram")
            st.markdown(
                """
- Each **circle (node)** is a minimum energy structure.
- Each **diamond** in the diagram is a transition state.
- Lines radiate from each diamond to the minima it connects.
- The longest chain in the network is always placed along the vertical centre.
- **Nodes are draggable** — click and drag any node to reposition it.
- Click **RESET** inside the diagram to restore original positions.
- Tick **TS labels** to show transition state energies.
- Hover over any node, diamond, or line to see its energy.
"""
            )
            st.markdown("### Node colours")
            st.markdown(
                """
<div style='margin:6px 0 4px'>
  <div style='
    background:linear-gradient(to right,
      rgb(0,60,220),rgb(0,210,220),rgb(0,200,50),rgb(220,210,0),rgb(220,0,0));
    height:14px;border-radius:4px;
  '></div>
  <div style='display:flex;justify-content:space-between;font-size:12px;color:#aaa;margin-top:4px'>
    <span>Stable (lowest energy)</span><span>Metastable (highest energy)</span>
  </div>
</div>

- **Green border** — 3D coordinates available from path.info.
- No green border — PATHSAMPLE-only node; energy shown but no 3D structure.
""",
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown("### Selecting a node")
            st.markdown(
                """
Use the **Select node** dropdown on the right of the spider web.

- The node's energy is always shown.
- If the node has a **green border**, a 3D molecular viewer appears below.
- Rotate and zoom the 3D structure using your mouse.
- Change the atom element symbol in the sidebar (default: Au).
"""
            )
            st.markdown("### Other tabs")
            st.markdown(
                """
| Tab | What it shows |
|---|---|
| **min.data energies** | All minimum energies and a bar chart. |
| **ts.data energies** | Transition state energies and connected nodes. |
| **path.info** | Triplets: min A, transition state, min B. |
| **Energy profiles** | Energy plotted along each connected pathway. Node colours match the spider web. |
"""
            )
            st.markdown("### Saving and sharing")
            st.markdown(
                """
- **Quick save** in the sidebar downloads the session as a JSON file.
- **Save as** lets you name the file first.
- **Share via email** opens a pre-filled email — attach the saved JSON file.
- Reload a previous session using **Load saved session** in the sidebar.
"""
            )
            st.markdown("### Multiple runs")
            st.markdown(
                """
The sidebar has six independent slots: **0, −1, −2, −3, −4, −5**.

Each slot holds its own files and session state.
Switch between slots to load different runs side by side without losing any data.
"""
            )
        return

    # ── Load and merge ──
    min_files  = read_files(up_min,  "min.data",  folder)
    ts_files   = read_files(up_ts,   "ts.data",   folder)
    path_files = read_files(up_path, "path.info", folder)

    if not min_files and not path_files:
        st.error("Load at least one file — min.data, path.info, or both — to begin.")
        return

    min_e_base, min_lines, min_stats = merge_min(min_files) if min_files else ([], [], [])
    ts_e,       ts_parts,  ts_stats  = merge_ts(ts_files, min_files) if ts_files else ([], [], [])
    triplets,   path_stats            = merge_path(path_files) if path_files else ([], [])

    edges, node_coords, min_e, ts_nodes, n_rmsd_merged = build_network(min_e_base, ts_e, triplets)
    n_path_only = len(min_e) - len(min_e_base)

    edge_tuples = tuple((ed["idA"], ed["idB"]) for ed in edges)
    spine = find_longest_chain(edge_tuples, len(min_e))
    pos = positions(len(min_e), spine)

    # Merge summary + session save in sidebar
    with st.sidebar:
        st.divider()
        st.caption("Merge summary")
        show_merge_stats("min.data",  min_stats,  len(min_e_base))
        show_merge_stats("ts.data",   ts_stats,   len(ts_e))
        show_merge_stats("path.info", path_stats, len(triplets))
        if n_path_only:
            st.caption(f"{n_path_only} node(s) added from path.info with no min.data match")
        if n_rmsd_merged:
            st.caption(f"{n_rmsd_merged} node(s) merged by rotation alignment (RMSD ≤ {_RMSD_TOL:.0e} Å)")

        st.divider()
        st.caption("Save & share")

        session_payload = json.dumps({
            "version": 1,
            "files": {
                "min.data":  [[n, t] for n, t in min_files],
                "ts.data":   [[n, t] for n, t in ts_files],
                "path.info": [[n, t] for n, t in path_files],
            },
        })

        # Quick save
        st.download_button(
            "Quick save",
            data=session_payload,
            file_name="session.json",
            mime="application/json",
            use_container_width=True,
            key="btn_quick_save",
        )

        # Save as
        with st.expander("Save as…"):
            raw_name = st.text_input(
                "Filename (no extension needed)",
                value="session",
                key="save_as_name",
            )
            safe_name = raw_name.strip() or "session"
            st.download_button(
                "Download",
                data=session_payload,
                file_name=f"{safe_name}.json",
                mime="application/json",
                use_container_width=True,
                key="btn_save_as",
            )

        # Share via email
        with st.expander("Share via email"):
            st.caption(
                "Save the session file first, then attach it to your email."
            )
            subj = urllib.parse.quote("Molecular Energy Landscape Session")
            body = urllib.parse.quote(
                f"Hi,\n\nI am sharing a molecular energy landscape session "
                f"(subfolder {folder}, {len(min_e)} minima, "
                f"{len(edges)} transition states).\n\n"
                f"Please find the session file (session.json) attached.\n\nRegards"
            )
            gmail   = f"https://mail.google.com/mail/?view=cm&fs=1&su={subj}&body={body}"
            outlook = f"https://outlook.live.com/mail/0/deeplink/compose?subject={subj}&body={body}"
            mailto  = f"mailto:?subject={subj}&body={body}"
            st.markdown(
                f"""
<div style='display:flex;flex-direction:column;gap:6px;margin-top:4px'>
  <a href="{gmail}" target="_blank"
     style="display:block;padding:5px 10px;border-radius:4px;
            background:#c5221f;color:white;text-decoration:none;
            font-size:12px;text-align:center">
    Open in Gmail
  </a>
  <a href="{outlook}" target="_blank"
     style="display:block;padding:5px 10px;border-radius:4px;
            background:#0078d4;color:white;text-decoration:none;
            font-size:12px;text-align:center">
    Open in Outlook
  </a>
  <a href="{mailto}"
     style="display:block;padding:5px 10px;border-radius:4px;
            background:#444;color:#eee;text-decoration:none;
            font-size:12px;text-align:center">
    Default mail client
  </a>
</div>
""",
                unsafe_allow_html=True,
            )

    # Warn if optional files absent
    missing = [k for k, fl in [("ts.data", ts_files), ("path.info", path_files)] if not fl]
    if missing:
        st.info(f"Running without: {', '.join(missing)}. Some edges or coordinate data may be absent.")

    # ── Session state (folder-specific) ──
    sel_key = f"sel_{folder}"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = None

    # ── Tabs ──
    tab_web, tab_min, tab_ts, tab_path, tab_ep = st.tabs(
        ["Spider web", "min.data energies", "ts.data energies", "path.info", "Energy profiles"]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1 — Spider web
    # ─────────────────────────────────────────────────────────────────────────
    with tab_web:
        st.markdown(
            f"**{len(min_e)} minima · {len(edges)} transition states** "
            f"({sum(1 for ed in edges if ed['source']=='path.info')} from path.info, "
            f"{sum(1 for ed in edges if ed['source']=='ts.data only')} from ts.data only)"
        )
        spine_str = " → ".join(str(n) for n in spine)
        st.caption(f"Longest chain ({len(spine)} nodes, centred): {spine_str}")
        graph_col, info_col = st.columns([3, 2])

        with graph_col:
            st.components.v1.html(
                build_draggable_html(min_e, ts_nodes, node_coords, pos),
                height=720,
                scrolling=False,
            )

        with info_col:
            sel_options = [None] + list(range(1, len(min_e) + 1))
            dropdown = st.selectbox(
                "Select node",
                options=sel_options,
                format_func=lambda v: "— select —" if v is None else f"Node {v}",
                index=0 if st.session_state[sel_key] is None
                      else sel_options.index(st.session_state[sel_key]),
                key=f"node_select_{folder}",
            )
            if dropdown != st.session_state[sel_key]:
                st.session_state[sel_key] = dropdown
                st.rerun()

            st.divider()

            sel = st.session_state[sel_key]
            if sel is None:
                st.info("Use the selector above to view node details.")
            else:
                e_val = min_e[sel - 1]
                has_c = sel in node_coords
                st.markdown(f"### Node {sel}")
                st.code(f"Energy (14 d.p.):  {e_val:.14f}", language=None)
                if has_c:
                    st.success("Coordinates available (path.info)")
                    rows = node_coords[sel]
                    st.components.v1.html(
                        build_3d_viewer_html(rows, element, height=280),
                        height=280,
                        scrolling=False,
                    )
                    with st.expander(f"Raw coordinates — {len(rows)} atoms"):
                        st.dataframe(
                            pd.DataFrame(rows, columns=["x", "y", "z"],
                                         index=range(1, len(rows) + 1)),
                            use_container_width=True,
                        )
                else:
                    st.caption("No coordinates — PATHSAMPLE-only node.")

                if st.button("Clear selection", key=f"btn_clear_sel_{folder}"):
                    st.session_state[sel_key] = None
                    st.rerun()

            st.divider()
            if not triplets:
                st.caption("No path.info loaded — triplet data unavailable.")
            else:
                st.markdown(f"#### OPTIM triplets ({len(triplets)})")
                st.caption("Min A → transition state → Min B")

                def nearest_min_id(e):
                    return min(range(len(min_e)), key=lambda j: abs(min_e[j] - e)) + 1

                for i, (mA, ts_s, mB) in enumerate(triplets):
                    idA, idB = nearest_min_id(mA["e"]), nearest_min_id(mB["e"])
                    with st.expander(
                        f"Triplet {i+1}  ·  {idA} ↔ {idB}  |  TS = {ts_s['e']:.14f}",
                        expanded=False,
                    ):
                        st.markdown(f"**Min A** (node {idA}) — E = `{mA['e']:.14f}`")
                        st.components.v1.html(
                            build_3d_viewer_html(mA["c"], element, height=220),
                            height=220, scrolling=False,
                        )
                        with st.expander("Raw coordinates A", expanded=False):
                            st.dataframe(
                                pd.DataFrame(mA["c"], columns=["x", "y", "z"],
                                             index=range(1, len(mA["c"]) + 1)),
                                use_container_width=True,
                            )
                        st.markdown(f"**Transition state** — E = `{ts_s['e']:.14f}`")
                        st.components.v1.html(
                            build_3d_viewer_html(ts_s["c"], element, height=220),
                            height=220, scrolling=False,
                        )
                        with st.expander("Raw coordinates TS", expanded=False):
                            st.dataframe(
                                pd.DataFrame(ts_s["c"], columns=["x", "y", "z"],
                                             index=range(1, len(ts_s["c"]) + 1)),
                                use_container_width=True,
                            )
                        st.markdown(f"**Min B** (node {idB}) — E = `{mB['e']:.14f}`")
                        st.components.v1.html(
                            build_3d_viewer_html(mB["c"], element, height=220),
                            height=220, scrolling=False,
                        )
                        with st.expander("Raw coordinates B", expanded=False):
                            st.dataframe(
                                pd.DataFrame(mB["c"], columns=["x", "y", "z"],
                                             index=range(1, len(mB["c"]) + 1)),
                                use_container_width=True,
                            )

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3 — min.data energies
    # ─────────────────────────────────────────────────────────────────────────
    with tab_min:
        st.subheader("Minimum energies (min.data)")
        st.caption("Merged result — line number = node number.")
        st.code("\n".join(min_lines), language=None, line_numbers=True)

        if len(min_stats) > 1:
            st.divider()
            st.markdown("**Per-file merge detail**")
            st.dataframe(pd.DataFrame(min_stats), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**Energy distribution**")
        st.bar_chart(
            pd.DataFrame({"Energy": min_e},
                         index=[f"Node {i}" for i in range(1, len(min_e) + 1)])
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 4 — ts.data energies
    # ─────────────────────────────────────────────────────────────────────────
    with tab_ts:
        st.subheader("Transition state energies (ts.data)")
        if not ts_e:
            st.info("No ts.data loaded. Upload it in the sidebar to see transition state energies.")
        else:
            def _nearest(e):
                return min(range(len(min_e)), key=lambda j: abs(min_e[j] - e)) + 1

            st.caption(
                "Merged result — columns 4 and 5 (min_A, min_B) are renumbered "
                "to match the merged min.data above. All other columns are unchanged."
            )
            reindexed = []
            for (_, ea, eb), parts in zip(ts_e, ts_parts):
                new_parts = parts[:]
                new_parts[3] = str(_nearest(ea))
                new_parts[4] = str(_nearest(eb))
                reindexed.append("    ".join(new_parts))
            st.code("\n".join(reindexed), language=None, line_numbers=True)

            if len(ts_stats) > 1:
                st.divider()
                st.markdown("**Per-file merge detail**")
                st.dataframe(pd.DataFrame(ts_stats), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("**TS energy distribution**")
            st.bar_chart(
                pd.DataFrame({"Energy": [t[0] for t in ts_e]},
                             index=[f"TS {i}" for i in range(1, len(ts_e) + 1)])
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 5 — path.info
    # ─────────────────────────────────────────────────────────────────────────
    with tab_path:
        st.subheader("Path.info triplets")
        if not triplets:
            st.info("No path.info loaded. Upload it in the sidebar to see triplet data.")
        else:
            def _pid(e):
                return min(range(len(min_e)), key=lambda j: abs(min_e[j] - e)) + 1

            st.caption(
                "Merged result in file order — three energies per triplet "
                "(min A, transition state, min B). "
                "Coordinates are available in the Spider web tab."
            )
            block_lines = []
            for i, (mA, ts_s, mB) in enumerate(triplets, start=1):
                idA, idB = _pid(mA["e"]), _pid(mB["e"])
                block_lines.append(f"# Triplet {i}  (node {idA} ↔ node {idB})")
                block_lines.append(f"{mA['e']:.14f}")
                block_lines.append(f"{ts_s['e']:.14f}")
                block_lines.append(f"{mB['e']:.14f}")
                block_lines.append("")
            st.code("\n".join(block_lines), language=None, line_numbers=True)

            if len(path_stats) > 1:
                st.divider()
                st.markdown("**Per-file merge detail**")
                st.dataframe(
                    pd.DataFrame(path_stats), use_container_width=True, hide_index=True
                )

            st.divider()
            st.markdown("**Transition state energy distribution**")
            st.bar_chart(
                pd.DataFrame(
                    {"TS Energy": [ts_s["e"] for _, ts_s, _ in triplets]},
                    index=[f"Triplet {i}" for i in range(1, len(triplets) + 1)],
                )
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 6 — Energy profiles
    # ─────────────────────────────────────────────────────────────────────────
    with tab_ep:
        st.subheader("Energy profiles along pathways")
        st.caption(
            "One chain at a time. Use the selector to step through all paths in each "
            "connected component. Circles = minima; diamonds = transition states. "
            "Hover over any point for its energy."
        )
        if not edges:
            st.info("No edges loaded. Upload ts.data or path.info to see energy profiles.")
        else:
            ep_components, ep_isolated = find_component_paths(edges, min_e)
            if ep_isolated:
                st.caption(
                    "Isolated nodes (no connections): "
                    + ", ".join(str(n) for n in ep_isolated)
                )
            if not ep_components:
                st.info("No connected components with edges found.")
            for ci, (comp_nodes, paths, truncated) in enumerate(ep_components):
                if truncated:
                    st.caption(f"Component {ci + 1}: showing first 100 of more paths.")
                path_idx = st.number_input(
                    f"Chain (1 – {len(paths)})",
                    min_value=1, max_value=len(paths), value=1, step=1,
                    key=f"ep_chain_{folder}_{ci}",
                )
                st.plotly_chart(
                    build_single_path_fig(paths[path_idx - 1], comp_nodes, min_e),
                    use_container_width=True,
                )



main()
