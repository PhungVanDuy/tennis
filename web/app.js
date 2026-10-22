// Rybakina 3D review site: one page over data/site.json (written by `r3d site`).
// Sections: overview · character · moveset (synced source | GVHMR | 3D) · artwork · footage index · notes.

import {
  THREE, Stage, JointSkeleton, loadAsset, instantiate, measure, fitUnits, placeOnGround, findRig, bodyFrame,
  fromBody,
} from './viewer.js';
import { Timeline } from './player.js';

const VIEWS = ['overview', 'character', 'moveset', 'artwork', 'footage', 'notes'];
const STROKE_ORDER = ['serve', 'forehand', 'backhand', 'slice', 'volley', 'overhead', 'celebration'];
const VIEW_META = {
  main_broadcast: { label: 'Main broadcast', color: 'var(--v-main)' },
  behind_low: { label: 'Behind, low', color: 'var(--v-behind)' },
  side: { label: 'Side', color: 'var(--v-side)' },
  closeup: { label: 'Close-up', color: 'var(--v-closeup)' },
  other: { label: 'Other', color: 'var(--v-other)' },
};
const svg = (body, fill = false) => `<svg viewBox="0 0 24 24" aria-hidden="true" ${fill
  ? 'fill="currentColor"'
  : 'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'}>${body}</svg>`;
const ICON = {
  play: svg('<path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5z"/>', true),
  pause: svg('<rect x="6.5" y="5" width="4" height="14" rx="1"/><rect x="13.5" y="5" width="4" height="14" rx="1"/>', true),
  back: svg('<path d="M6 5v14"/><path d="M18 6l-8 6 8 6z" fill="currentColor"/>'),
  fwd: svg('<path d="M18 5v14"/><path d="M6 6l8 6-8 6z" fill="currentColor"/>'),
  loop: svg('<path d="M17 2l4 4-4 4"/><path d="M3 11v-1a4 4 0 0 1 4-4h14"/><path d="M7 22l-4-4 4-4"/><path d="M21 13v1a4 4 0 0 1-4 4H3"/>'),
  sound: svg('<path d="M11 5L6 9H2v6h4l5 4z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M19 5a10 10 0 0 1 0 14"/>'),
  mute: svg('<path d="M11 5L6 9H2v6h4l5 4z"/><path d="M23 9l-6 6M17 9l6 6"/>'),
  prev: svg('<path d="M15 18l-6-6 6-6"/>'),
  next: svg('<path d="M9 18l6-6-6-6"/>'),
  contact: svg('<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="2.5" fill="currentColor"/>'),
  external: svg('<path d="M14 3h7v7"/><path d="M10 14L21 3"/><path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"/>'),
};

let D = null;                       // site.json
const S = { view: null, clip: null };
let CV = null;                      // character viewer
let MV = null;                      // moveset review (stage + timeline)

// ------------------------------------------------------------------------------------------------ utils
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
}[c]));
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1).replace(/_/g, ' ') : '');
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
function num(v, d = 0) {
  if (!isNum(v)) return '–';
  return v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
}
const pct = (v, d = 0) => (isNum(v) ? `${num(v * 100, d)}%` : '–');
function mmss(t, d = 1) {
  if (!isNum(t)) return '–';
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${m}:${s.toFixed(d).padStart(d ? 3 + d : 2, '0')}`;
}
function fmtDate(iso) {
  if (!iso) return '–';
  const d = new Date(iso);
  return d.toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' });
}
const viewLabel = (v) => VIEW_META[v]?.label || cap(v);
const viewColor = (v) => VIEW_META[v]?.color || 'var(--line-2)';
const pendingBox = (text, cls = '') => `<div class="pending-box ${cls}">${esc(text)}</div>`;
const badgeFor = (status) => {
  const s = (status || '').toUpperCase();
  const cls = { PASS: 'pass', WARN: 'warn', FAIL: 'fail' }[s] || 'pending';
  return `<span class="badge ${cls}">${esc(s || 'QA pending')}</span>`;
};
const imgCard = (url, caption, cls = '', { dark = false, alt, full, eager = false } = {}) => (url
  ? `<figure class="img-card ${cls}${dark ? ' dark' : ''}" data-full="${esc(url)}" data-caption="${esc(full || caption)}">
       <img src="${esc(url)}" alt="${esc(alt || caption)}"${eager ? '' : ' loading="lazy"'}>
       ${caption ? `<figcaption class="cap">${esc(caption)}</figcaption>` : ''}</figure>`
  : pendingBox(`${caption}: pending`, cls));
const kv = (rows) => `<dl class="kv">${rows.filter(Boolean).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('')}</dl>`;

const jsonCache = new Map();
function fetchJSON(url) {
  if (!jsonCache.has(url)) {
    const p = fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
      return r.json();
    });
    p.catch(() => jsonCache.delete(url));
    jsonCache.set(url, p);
  }
  return jsonCache.get(url);
}

// ------------------------------------------------------------------------------------------------ curation
// `curate` checks each clip's motion against its label: with keep_all every clip is animated and failures are
// flagged; otherwise it keeps the best matching clips per stroke and the rest stay reviewable.
const isKept = (c) => !D.curation || !!c?.curation?.kept;
const isRejected = (c) => !!D.curation && !isKept(c);
const REJECT_ORDER = { quota: 0, label: 1, qa: 2 };

/** [[stroke, {kept, rejected}]]: kept clips in curation order (best first), then the rejected ones. */
function clipsByStroke() {
  const groups = new Map();
  for (const c of D.clips) {
    const s = c.meta?.stroke || 'other';
    if (!groups.has(s)) groups.set(s, { kept: [], rejected: [] });
    groups.get(s)[isKept(c) ? 'kept' : 'rejected'].push(c);
  }
  const order = D.curation?.kept_ids || [];
  const pos = (c) => { const i = order.indexOf(c.id); return i < 0 ? 999 : i; };
  for (const g of groups.values()) {
    g.kept.sort((a, b) => pos(a) - pos(b));
    g.rejected.sort((a, b) => (REJECT_ORDER[a.curation?.category] ?? 3) - (REJECT_ORDER[b.curation?.category] ?? 3)
      || (b.curation?.score ?? 0) - (a.curation?.score ?? 0));
  }
  return [...groups.entries()].sort(([a], [b]) => rank(a) - rank(b));
  function rank(s) { const i = STROKE_ORDER.indexOf(s); return i < 0 ? 99 : i; }
}

/** The clip the moveset opens on: curate's best clip of the first stroke (the best serve), never a rejected one. */
function defaultClipId() {
  const best = (D.curation?.kept_ids || []).find((id) => D.clipById[id]);
  if (best) return best;
  const first = clipsByStroke().find(([, g]) => g.kept.length);
  return first ? first[1].kept[0].id : D.clips[0]?.id;
}

/** Up / Down and the prev / next buttons walk the kept clips; from a rejected clip, every clip in list order. */
function navOrder(c) {
  const groups = clipsByStroke();
  return isKept(c)
    ? groups.flatMap(([, g]) => g.kept.map((x) => x.id))
    : groups.flatMap(([, g]) => [...g.kept, ...g.rejected].map((x) => x.id));
}

function curationBadge(c, long = false) {
  const cu = c.curation;
  if (!cu) return '';
  if (cu.category === 'flagged') return `<span class="badge flagged" title="${esc(cu.why)}">${long ? 'kept · label flagged' : 'flagged'}</span>`;
  if (cu.kept) return `<span class="badge kept" title="${esc(cu.why)}">kept${cu.rank ? ` #${cu.rank}${long ? ` of ${cu.quota}` : ''}` : ''}</span>`;
  return `<span class="badge rejected" title="${esc(cu.why)}">${long ? 'rejected by curate' : 'rejected'}</span>`;
}

// ------------------------------------------------------------------------------------------------ signature
// signature.json: per kept clip and per stroke (mean ± sd), normalised so it compares across players.
const SIG_METRICS = [
  { key: 'contact_height', label: 'Contact height', unit: '% of her height', suffix: '%', scale: 100, d: 0,
    hint: 'how high her hitting hand is above the court at contact' },
  { key: 'knee_flex_deg', label: 'Knee flex', unit: 'degrees', suffix: '°', scale: 1, d: 0,
    hint: 'deepest knee bend in the 1.2 s before contact' },
  { key: 'hip_shoulder_sep_deg', label: 'Hip–shoulder separation', unit: 'degrees', suffix: '°', scale: 1, d: 0,
    hint: 'the coil: how far her shoulders turn past her hips' },
  { key: 'stance_width', label: 'Stance width', unit: '× leg length', suffix: '×', scale: 1, d: 2,
    hint: 'distance between her feet at contact' },
  { key: 'toss_above_head_m', label: 'Toss above head', unit: 'metres', suffix: ' m', scale: 1, d: 2,
    hint: 'toss hand at its highest point, above her head' },
  { key: 'toss_to_contact_s', label: 'Toss → contact', unit: 'seconds', suffix: ' s', scale: 1, d: 2,
    hint: 'time from the top of the toss to the ball strike' },
  { key: 'hand_speed_ms', label: 'Hand speed', unit: 'm/s', suffix: ' m/s', scale: 1, d: 1,
    hint: 'hitting wrist at contact (the wrist, not the racket head)' },
];
const sigVal = (m, v) => (isNum(v) ? num(v * m.scale, m.d) : '–');
const sigMetrics = (row) => SIG_METRICS.filter((m) => row && row[m.key] && isNum(row[m.key].mean));

/** Shared axis per metric over every measured clip, so a stroke's spread reads against the others. */
function sigDomain(key) {
  const vals = Object.values(D.signature?.per_clip || {}).map((r) => r[key]).filter(isNum);
  if (!vals.length) return [0, 1];
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 1;
  return [lo - pad, hi + pad];
}

function sigStrip(m, stroke, me = null) {
  const [lo, hi] = sigDomain(m.key);
  const xn = (v) => clamp(((v - lo) / (hi - lo)) * 100, 0, 100);
  const x = (v) => `${xn(v).toFixed(2)}%`;
  const st = D.signature?.by_stroke?.[stroke]?.[m.key];
  const dots = Object.entries(D.signature?.per_clip || {})
    .filter(([, r]) => r.stroke === stroke && isNum(r[m.key]))
    .map(([id, r]) => `<a class="dot${id === me ? ' me' : ''}" href="#/moveset/${esc(id)}" style="left:${x(r[m.key])}"
      data-tip="<b>${esc(id)}</b><br>${esc(m.label)}: ${sigVal(m, r[m.key])}${esc(m.suffix)}" aria-label="${esc(id)}"></a>`).join('');
  const band = st && isNum(st.sd) && st.sd > 0
    ? `<span class="band" style="left:${x(st.mean - st.sd)};width:${(xn(st.mean + st.sd) - xn(st.mean - st.sd)).toFixed(2)}%"></span>` : '';
  return `<div class="sig-strip" aria-hidden="true">${band}${st ? `<span class="mean" style="left:${x(st.mean)}"></span>` : ''}${dots}</div>`;
}

const sigMeanSd = (m, st) => `${sigVal(m, st.mean)}${st.sd > 0 ? ` <small>± ${sigVal(m, st.sd)}</small>` : ''}<span class="u">${esc(m.suffix.trim())}</span>`;

// ------------------------------------------------------------------------------------------------ boot
boot();

async function boot() {
  try {
    const res = await fetch('data/site.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`data/site.json: HTTP ${res.status}`);
    D = await res.json();
  } catch (e) {
    const b = $('#boot');
    b.classList.add('error');
    b.innerHTML = `<p>Could not load the run data (${esc(e.message)}).<br>Build it with <code>r3d site</code>
      and open it through <code>r3d serve</code>.</p>`;
    return;
  }
  $('#boot').remove();
  D.clipById = Object.fromEntries(D.clips.map((c) => [c.id, c]));
  D.shotById = Object.fromEntries(D.footage.shots.map((s) => [s.id, s]));
  $('#build-info').textContent = `run built ${fmtDate(D.generated)}`;
  renderOverview();
  renderCharacter();
  renderMoveset();
  renderArtwork();
  renderFootage();
  renderNotes();
  setupLightbox();
  setupTooltip();
  window.addEventListener('hashchange', route);
  route();
  // console handle for reviewers (and the headless check): r3dReview.moveset().tl.frame, …
  window.r3dReview = { data: () => D, state: S, moveset: () => MV, character: () => CV, variants: () => VR };
}

function route() {
  const parts = (location.hash || '#/overview').replace(/^#\/?/, '').split('/');
  const view = VIEWS.includes(parts[0]) ? parts[0] : 'overview';
  const changed = S.view !== view;
  S.view = view;
  for (const sec of $$('.view')) sec.hidden = sec.dataset.view !== view;
  for (const a of $$('#tabs a')) a.classList.toggle('active', a.dataset.view === view);
  if (changed) window.scrollTo({ top: 0 });

  CV?.stage.setActive(view === 'character');
  motionActive(view === 'artwork');
  variantActive(view === 'artwork');
  if (view !== 'moveset' && MV) {
    MV.tl.pause();
    MV.stage.setActive(false);
  }
  if (view === 'overview') {
    requestAnimationFrame(() => {
      drawEdges();
      // #/overview/<section>: jump to a section of the overview (e.g. the stroke signature)
      if (parts[1]) document.getElementById(parts[1])?.scrollIntoView({ block: 'start' });
    });
  }
  if (view === 'character') ensureCharViewer();
  if (view === 'moveset' && D.clips.length) {
    ensureMoveset();
    const id = decodeURIComponent(parts[1] || '') || S.clip || defaultClipId();
    if (id) selectClip(id);
  }
  if (view === 'footage' && parts[1]) openShot(Number(parts[1]), true);
}

// ------------------------------------------------------------------------------------------------ overview
const EDGES = [
  ['footage', 'index'], ['index', 'select'], ['select', 'clips'], ['clips', 'track'], ['track', 'camera'],
  ['camera', 'motion'], ['motion', 'qa'], ['qa', 'curate'], ['curate', 'retarget'], ['retarget', 'signature'],
  ['refs', 'stylize'], ['stylize', 'mesh'], ['mesh', 'retarget', 'cross'], ['mesh', 'artwork'],
  ['retarget', 'artwork', 'cross'],
];

function renderOverview() {
  const f = D.footage;
  const ch = D.character;
  const rig = ch.rig.stats || {};
  const nMotion = D.clips.filter((c) => c.motion).length;
  const nAnim = D.clips.filter((c) => c.anim).length;
  const cu = D.curation;
  const primary = ch.mesh.generators[ch.mesh.primary] || Object.values(ch.mesh.generators)[0];
  const thumb = primary?.files?.thumbnail?.url;
  const heroModel = ch.sheet.prompts?.hero?.model;
  const probe = f.probe || {};
  const kpis = [
    ['Footage', f.minutes != null ? `${num(f.minutes, 1)}<small>min</small>` : '–',
      probe.width ? `${probe.width}×${probe.height} @ ${num(probe.fps)} fps` : 'probe pending'],
    ['Shots', num(f.n_shots), `${num(f.usable)} usable for capture`],
    ['Her strokes labelled', num(f.n_strokes), `${num(f.opponent_strokes)} opponent strokes`],
    ['Stroke clips', num(D.clips.length), cu ? `${num(cu.kept)} kept by curate · ${num(D.selection.n_candidates)} candidates`
      : `of ${num(D.selection.n_candidates)} candidates`],
    ['Motion recovered', `${nMotion}<small>/ ${D.clips.length}</small>`, 'GVHMR, world-grounded', nMotion === D.clips.length],
    ['Triangles', rig.triangles ? num(rig.triangles) : '–', `${cap(ch.mesh.primary || 'primary')} rigged mesh`],
    ['Bones', rig.joints ? num(rig.joints.length) : '–', 'humanoid auto-rig'],
    ['Animations', D.moveset.url ? num(nAnim) : '–', D.moveset.url ? (cu && nAnim >= cu.kept ? 'all kept clips, in the GLB' : 'in the moveset GLB') : 'retarget pending'],
  ];
  $('#view-overview').innerHTML = `
    <div class="hero">
      <div>
        <div class="eyebrow">Character and moveset review</div>
        <h1>Elena Rybakina in the client's Naomi Osaka art style, <em>moving like her</em>.</h1>
        <p class="lede">A ${isNum(f.minutes) ? `${num(f.minutes, 0)}-minute ` : ''}WTA compilation and ${ch.refs.length || 'a set of'} reference images go in.
          Out comes a stylized, rigged game character whose moveset is recovered from her real strokes: Gemini finds
          them in the broadcast, the audio pins the ball contact, GVHMR lifts each stroke to 3D and Blender retargets it
          onto the rig.</p>
        <div class="hero-cta">
          <a class="btn primary" href="#/moveset">Review the moveset</a>
          <a class="btn" href="#/character">Inspect the character</a>
          <a class="btn" href="#/overview/signature">Stroke signature</a>
          <a class="btn" href="#/footage">Footage index</a>
        </div>
      </div>
      <figure class="hero-art">
        ${ch.sheet.hero ? `<img src="${esc(ch.sheet.hero)}" alt="Stylized hero render" data-full="${esc(ch.sheet.hero)}" data-caption="Hero render">` : pendingBox('hero pending')}
        ${thumb ? `<img class="dark" src="${esc(thumb)}" alt="3D mesh render" data-full="${esc(thumb)}" data-caption="${esc(cap(ch.mesh.primary))} mesh">` : pendingBox('mesh pending')}
        <figcaption><span>2D hero${heroModel ? ` · ${esc(heroModel)}` : ''}</span><span>3D · ${esc(cap(ch.mesh.primary || 'mesh'))}</span></figcaption>
      </figure>
    </div>
    <div class="kpis">${kpis.map(([label, value, sub, accent]) => `
      <div class="kpi${accent ? ' accent' : ''}"><div class="kpi-label">${esc(label)}</div>
        <div class="kpi-value">${value}</div><div class="kpi-sub">${esc(sub)}</div></div>`).join('')}
    </div>
    ${signatureBlock()}
    <div class="block">
      <h2>Pipeline <small>live status from the run folder</small></h2>
      <div class="card pipeline" id="pipeline">
        <svg class="edges" id="edges"></svg>
        <div class="lane-label">Motion: footage to moveset</div>
        <div class="lane motion">${D.pipeline.filter((s) => s.lane === 'motion').map(pnode).join('')}</div>
        <div class="lane-label">Character: references to rig to artwork</div>
        <div class="lane character">${D.pipeline.filter((s) => s.lane === 'character').map(pnode).join('')}</div>
      </div>
    </div>
    <div class="block">
      <h2>Models and tools</h2>
      <div class="card table-wrap"><table class="data">
        <thead><tr><th>Stage</th><th>Model / tool</th><th>Role</th><th>Status</th><th>Updated</th></tr></thead>
        <tbody>${D.pipeline.map((s) => `<tr>
          <td><b>${esc(s.name)}</b></td>
          <td>${s.models.map((m) => `<code>${esc(m)}</code>`).join('<br>')}</td>
          <td class="muted">${esc(s.what)}</td>
          <td><span class="view-tag status-${s.status}"><span class="status-dot"></span>${esc(s.detail)}</span>${breakdown(s)}</td>
          <td class="faint">${s.updated ? fmtDate(s.updated) : 'pending'}${isNum(s.seconds) ? ` · ${num(s.seconds)} s` : ''}</td>
        </tr>`).join('')}</tbody>
      </table></div>
    </div>`;
  new ResizeObserver(() => drawEdges()).observe($('#pipeline'));
}

const breakdown = (s) => (s.breakdown?.length
  ? `<ul class="breakdown">${s.breakdown.map((t) => `<li${/^0 /.test(t) ? ' class="zero"' : ''}>${esc(t)}</li>`).join('')}</ul>` : '');

function pnode(s) {
  return `<div class="pnode status-${s.status}" data-node="${esc(s.id)}">
    <div class="pnode-head"><span class="status-dot"></span>${esc(s.name)}</div>
    <div class="pnode-detail">${esc(s.detail)}</div>${breakdown(s)}
    <div class="pnode-what">${esc(s.what)}</div>
    <div class="pnode-models">${s.models.map((m) => `<code title="${esc(m)}">${esc(m.replace(/^fal-ai\//, ''))}</code>`).join('')}</div>
  </div>`;
}

function signatureBlock() {
  const by = D.signature?.by_stroke || {};
  const strokes = Object.keys(by).sort((a, b) => STROKE_ORDER.indexOf(a) - STROKE_ORDER.indexOf(b));
  const intro = `<p class="block-lede">These numbers are what makes the motion hers: measured on her recovered 3D motion
    (only clips whose motion matches their stroke label), normalised by her height and leg length, so they can be compared directly against another player
    run through the same pipeline. Each dot is one clip (click to open it); the bar marks the mean, the band ± 1 sd.</p>`;
  if (!strokes.length) {
    return `<div class="block" id="signature"><h2>Stroke signature</h2>${intro}${pendingBox('signature.json pending: r3d signature')}</div>`;
  }
  return `<div class="block" id="signature">
    <h2>Stroke signature <small>mean ± sd over the clips whose motion matches their label</small></h2>
    ${intro}
    <div class="sig-grid">${strokes.map((st) => {
      const row = by[st];
      return `<div class="card card-pad sig-card">
        <div class="sig-head"><h3>${esc(cap(st))}</h3><span>${row.n === 1 ? 'one clip, no spread yet' : `n = ${num(row.n)} clips`}</span></div>
        <div class="sig-list">${sigMetrics(row).map((m) => `
          <div class="sig-row" title="${esc(m.hint)}">
            <div class="sig-label"><b>${esc(m.label)}</b><span>${esc(m.unit)}</span></div>
            <div class="sig-value">${sigMeanSd(m, row[m.key])}</div>
            ${sigStrip(m, st)}
          </div>`).join('')}
        </div>
      </div>`;
    }).join('')}</div>
  </div>`;
}

function drawEdges() {
  const wrap = $('#pipeline');
  const edges = $('#edges');
  if (!wrap || !edges || wrap.offsetParent === null) return;
  const box = wrap.getBoundingClientRect();
  const rect = (id) => {
    const el = wrap.querySelector(`[data-node="${id}"]`);
    if (!el) return null;
    const r = el.getBoundingClientRect();
    const l = r.left - box.left;
    const t = r.top - box.top;
    return { l, t, r: l + r.width, b: t + r.height, cx: l + r.width / 2, cy: t + r.height / 2 };
  };
  const narrow = window.innerWidth <= 720;
  const paths = EDGES.map(([a, b, cls = '']) => {
    const A = rect(a);
    const B = rect(b);
    if (!A || !B || (narrow && cls)) return '';
    let d;
    if (Math.abs(A.t - B.t) < 6 && B.l > A.r) d = `M${A.r},${A.t + 22} L${B.l - 3},${B.t + 22}`;
    else if (B.t > A.b) {
      const my = (A.b + B.t) / 2;
      d = `M${A.cx},${A.b} L${A.cx},${my} L${B.cx},${my} L${B.cx},${B.t - 3}`;
    } else if (B.b < A.t) {
      const my = (B.b + A.t) / 2;
      d = `M${A.cx},${A.t} L${A.cx},${my} L${B.cx},${my} L${B.cx},${B.b + 3}`;
    } else d = `M${A.r},${A.cy} L${B.l - 3},${B.cy}`;
    return `<path class="${cls}" d="${d}" marker-end="url(#${cls ? 'arrow-accent' : 'arrow'})"/>`;
  });
  const marker = (id) => `<marker id="${id}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7"
    orient="auto"><path d="M0,0 L8,4 L0,8 z"/></marker>`;
  edges.innerHTML = `<defs>${marker('arrow')}${marker('arrow-accent')}</defs>${paths.join('')}`;
}

// ------------------------------------------------------------------------------------------------ character
function viewerOptions() {
  const out = [];
  const ch = D.character;
  for (const [gen, g] of Object.entries(ch.mesh.generators)) {
    const files = g.files || {};
    const stats = g.stats || {};
    if (gen === ch.mesh.primary && D.moveset.url) {
      out.push({ gen, id: 'moveset', label: 'Moveset', url: D.moveset.url, kind: 'glb', stats: D.moveset.stats });
    }
    if (files.rigged_character_glb) {
      out.push({ gen, id: 'rigged', label: 'Rigged', url: files.rigged_character_glb.url, kind: 'glb',
        stats: stats.rigged_character_glb, size: files.rigged_character_glb.size_mb });
    }
    const raw = files.model_glb || (files.model && ['glb', 'fbx'].includes(files.model.kind) ? files.model : null);
    if (raw) {
      out.push({ gen, id: 'raw', label: 'Raw mesh', url: raw.url, kind: raw.kind, stats: stats.model_glb || stats.model,
        size: raw.size_mb });
    }
  }
  if (!out.length && ch.rig.url) {
    out.push({ gen: 'primary', id: 'rigged', label: 'Rigged', url: ch.rig.url, kind: 'glb', stats: ch.rig.stats,
      size: ch.rig.size_mb });
  }
  return out;
}

function renderCharacter() {
  const ch = D.character;
  const gens = [...new Set(viewerOptions().map((o) => o.gen))];
  const groups = [
    ['identity', 'Identity', 'photos of her: face, build, hair'],
    ['style', 'Style', 'the client\'s avatar and the Osaka target look'],
    ['sheet', 'Sheet layout', 'how the client lays out a character sheet'],
  ];
  const genStats = (g) => {
    const s = g?.stats || {};
    return s.rigged_character_glb || s.model_glb || null;
  };
  const gm = ch.mesh.generators;
  const compareRows = [
    ['Triangles', (s) => num(s?.triangles)],
    ['Vertices', (s) => num(s?.vertices)],
    ['Bones', (s) => (s?.joints?.length ? num(s.joints.length) : '–')],
    ['Textures', (s) => num(s?.textures)],
    ['File (rigged)', (s, g) => (g?.files?.rigged_character_glb ? `${num(g.files.rigged_character_glb.size_mb, 1)} MB` : '–')],
  ];
  const genNames = Object.keys(gm);
  $('#view-character').innerHTML = `
    <div class="section-head"><div>
      <h1>Character</h1>
      <p>Identity comes from photos of her, the look from the client's own avatar and the Naomi Osaka target render.
        One hero image, an A-pose turnaround cut into four views, then multi-view image-to-3D and a humanoid auto-rig
        (Meshy as primary, Tripo as the A/B).</p>
    </div></div>
    <div class="char-layout">
      <div class="card viewer">
        <div class="viewer-toolbar">
          <div class="seg accent" id="cv-gen" role="group" aria-label="Generator">${gens.map((g) => `
            <button type="button" data-gen="${esc(g)}">${esc(cap(g))}</button>`).join('')}</div>
          <div class="seg" id="cv-variant" role="group" aria-label="Variant"></div>
          <select id="cv-anim" aria-label="Animation"></select>
          <span class="spacer"></span>
          <button class="btn small" id="cv-bones" type="button">Bones</button>
          <button class="btn small on" id="cv-spin" type="button">Turntable</button>
          <button class="btn small" id="cv-reset" type="button">Reset view</button>
        </div>
        <div class="stage char-stage" id="cv-stage">
          <div class="stage-overlay" id="cv-overlay"><div class="inner"><div class="spinner"></div><span>Loading…</span></div></div>
          <div class="stage-note" id="cv-note"></div>
        </div>
      </div>
      <aside class="char-side">
        <div class="card card-pad side-card"><h3>Loaded model</h3><dl class="stats-list" id="cv-stats"></dl></div>
        <div class="card card-pad side-card">
          <h3>Meshy vs Tripo</h3>
          ${genNames.length ? `<table class="data compare">
            <thead><tr><th></th>${genNames.map((g) => `<th>${esc(cap(g))}${g === ch.mesh.primary ? ' <span class="badge accent">primary</span>' : ''}</th>`).join('')}</tr></thead>
            <tbody>${compareRows.map(([label, fn]) => `<tr><td class="faint">${label}</td>${genNames.map((g) => `
              <td class="num">${gm[g].error ? '<span class="badge fail">failed</span>' : fn(genStats(gm[g]), gm[g])}</td>`).join('')}</tr>`).join('')}
            </tbody></table>
            <p class="faint" style="margin-top:10px;font-size:12px">${genNames.map((g) => `${esc(cap(g))}: <code>${esc(gm[g].endpoint || '')}</code>`).join('<br>')}</p>`
            : pendingBox('mesh pending')}
        </div>
      </aside>
    </div>

    <div class="block">
      <h2>Style sheet <small>${esc(ch.sheet.prompts?.hero?.model || '')}</small></h2>
      <div class="sheet">
        ${imgCard(ch.sheet.hero, 'Hero, front A-pose', 'hero-img')}
        <div class="right">
          ${imgCard(ch.sheet.turnaround, 'Turnaround sheet', 'turn')}
          <div class="views">${['front', 'left', 'back', 'right'].map((v) => {
            const file = ch.sheet.view_files?.[v];
            return imgCard(ch.sheet.views?.[v], VIEW_NAMES[v], '', { alt: `Her ${VIEW_NAMES[v].toLowerCase()}`,
              full: `${VIEW_NAMES[v]}${file ? ` (views/${file})` : ''}` });
          }).join('')}</div>
          ${swappedNote(ch.sheet.view_files)}
        </div>
      </div>
      ${Object.keys(ch.sheet.prompts || {}).length ? `<details class="prompts"><summary>Image prompts</summary>
        ${Object.entries(ch.sheet.prompts).map(([k, p]) => `<pre><b>${esc(cap(k))}</b> · ${esc(p.model)}\n\n${esc(p.prompt)}</pre>`).join('')}
      </details>` : ''}
    </div>

    <div class="block">
      <h2>References <small>${ch.refs.length} images</small></h2>
      <div class="refs-groups">${groups.map(([role, title, sub]) => {
        const refs = ch.refs.filter((r) => r.role === role);
        if (!refs.length) return '';
        const wide = role === 'sheet';  // a very wide strip: shown whole, full width
        return `<div class="refs-group"><h3>${title} <span>${esc(sub)}</span></h3><div class="refs${wide ? ' refs-wide' : ''}">${refs.map((r) => `
          <div class="ref"><figure>${imgCard(r.url, '', '', { alt: r.name, full: r.name.replace(/_/g, ' '), eager: wide })}
            <figcaption><b>${esc(r.name.replace(/_/g, ' '))}</b>${esc(r.note || r.source || '')}</figcaption></figure></div>`).join('')}
          </div></div>`;
      }).join('')}</div>
    </div>`;
}

// views/*.png are keyed by the side of her they show (mesh.json); the files are where the image model drew them
const VIEW_NAMES = { front: 'Front', left: 'Left side', back: 'Back', right: 'Right side' };

function swappedNote(files) {
  const swapped = Object.entries(files || {}).filter(([v, f]) => f && f.replace(/\.\w+$/, '') !== v);
  if (!swapped.length) return '';
  return `<p class="views-note">Labelled by what each view shows (<code>mesh.json</code>): the image model drew the side
    views in swapped positions, so ${swapped.map(([v, f]) => `her ${esc(VIEW_NAMES[v]?.toLowerCase() || v)} is
    <code>${esc(f)}</code>`).join(' and ')}. Image-to-3D received them with these labels.</p>`;
}

function ensureCharViewer() {
  if (CV) {
    CV.stage.setActive(true);
    return;
  }
  const opts = viewerOptions();
  const el = $('#cv-stage');
  if (!opts.length) {
    $('#cv-overlay').innerHTML = '<div class="inner">No mesh yet: run <code>r3d character mesh</code>.</div>';
    return;
  }
  let stage;
  try {
    stage = new Stage(el);
  } catch (e) {
    $('#cv-overlay').innerHTML = `<div class="inner">WebGL is not available (${esc(e.message)}).</div>`;
    return;
  }
  const primary = D.character.mesh.primary || opts[0].gen;
  CV = { stage, opts, gen: opts.some((o) => o.gen === primary) ? primary : opts[0].gen, variant: null, obj: null,
    asset: null, mixer: null, helper: null, bones: false, token: 0, rig: null };
  CV.variant = (opts.find((o) => o.gen === CV.gen && o.id === 'moveset') || opts.find((o) => o.gen === CV.gen)).id;
  stage.controls.autoRotate = true;
  stage.onFrame((dt) => CV.mixer?.update(dt));
  stage.setActive(true);

  $('#cv-gen').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-gen]');
    if (!b || b.dataset.gen === CV.gen) return;
    CV.gen = b.dataset.gen;
    const same = opts.find((o) => o.gen === CV.gen && o.id === CV.variant);
    CV.variant = (same || opts.find((o) => o.gen === CV.gen)).id;
    loadCharModel();
  });
  $('#cv-variant').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-variant]');
    if (!b || b.dataset.variant === CV.variant) return;
    CV.variant = b.dataset.variant;
    loadCharModel();
  });
  $('#cv-anim').addEventListener('change', (e) => playCharAnim(e.target.value));
  $('#cv-bones').addEventListener('click', (e) => {
    CV.bones = !CV.bones;
    e.currentTarget.classList.toggle('on', CV.bones);
    if (CV.helper) CV.helper.visible = CV.bones;
  });
  $('#cv-spin').addEventListener('click', (e) => {
    stage.controls.autoRotate = !stage.controls.autoRotate;
    e.currentTarget.classList.toggle('on', stage.controls.autoRotate);
  });
  $('#cv-reset').addEventListener('click', () => frameCharacter(true));
  loadCharModel();
}

function currentCharOption() {
  return CV.opts.find((o) => o.gen === CV.gen && o.id === CV.variant) || CV.opts[0];
}

async function loadCharModel() {
  const opt = currentCharOption();
  const token = ++CV.token;
  $$('#cv-gen button').forEach((b) => b.classList.toggle('active', b.dataset.gen === CV.gen));
  $('#cv-variant').innerHTML = CV.opts.filter((o) => o.gen === CV.gen).map((o) => `
    <button type="button" data-variant="${o.id}" class="${o.id === CV.variant ? 'active' : ''}">${esc(o.label)}</button>`).join('');
  const overlay = $('#cv-overlay');
  overlay.hidden = false;
  overlay.innerHTML = `<div class="inner"><div class="spinner"></div><span id="cv-progress">Loading ${esc(cap(opt.gen))} · ${esc(opt.label)}…</span></div>`;
  try {
    const asset = await loadAsset(opt.url, opt.kind, (loaded, total) => {
      const p = $('#cv-progress');
      if (p && total) p.textContent = `Loading ${cap(opt.gen)} · ${opt.label}… ${Math.round((loaded / total) * 100)}%`;
    });
    if (token !== CV.token) return;
    if (CV.obj) CV.stage.remove(CV.obj);
    if (CV.helper) CV.stage.remove(CV.helper);
    CV.mixer?.stopAllAction();
    const obj = instantiate(asset);
    fitUnits(obj, D.character.height_m);
    const height = placeOnGround(obj);
    CV.stage.add(obj);
    CV.obj = obj;
    CV.asset = asset;
    CV.height = height;
    CV.rig = findRig(obj);
    CV.rest = CV.rig.bones.map((b) => [b, b.position.clone(), b.quaternion.clone(), b.scale.clone()]);
    CV.mixer = asset.animations.length ? new THREE.AnimationMixer(obj) : null;
    CV.helper = CV.rig.bones.length ? new THREE.SkeletonHelper(obj) : null;
    if (CV.helper) {
      CV.helper.visible = CV.bones;
      CV.stage.add(CV.helper);
    }
    $('#cv-bones').disabled = !CV.helper;
    const select = $('#cv-anim');
    const anims = asset.animations.map((a) => a.name);
    select.innerHTML = `<option value="">Rest pose</option>${anims.map((a) => `<option value="${esc(a)}">${esc(a)}</option>`).join('')}`;
    select.disabled = !anims.length;
    const first = opt.id === 'moveset' ? anims.find((a) => D.clipById[a]) || anims[0] : '';
    select.value = first || '';
    playCharAnim(select.value);
    frameCharacter(true);
    overlay.hidden = true;
    const measured = measure(obj);
    const s = opt.stats || {};
    const joints = s.joints?.length ?? measured.joints;
    $('#cv-stats').innerHTML = [
      ['Generator', esc(cap(opt.gen))],
      ['Variant', esc(opt.label)],
      ['Triangles', num(s.triangles ?? measured.triangles)],
      ['Vertices', num(s.vertices ?? measured.vertices)],
      ['Bones', joints ? num(joints) : 'unrigged'],
      ['Textures', isNum(s.textures) ? num(s.textures) : '–'],
      ['Animations', num(anims.length)],
      ['Height', `${num(height, 2)} m`],
      ['File', `${opt.kind.toUpperCase()} · ${isNum(s.size_mb ?? opt.size) ? `${num(s.size_mb ?? opt.size, 1)} MB` : '–'}`],
    ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
    $('#cv-note').innerHTML = `<span>${esc(cap(opt.gen))} · ${esc(opt.label)}</span><span>drag to orbit · scroll to zoom</span>`;
  } catch (e) {
    if (token !== CV.token) return;
    overlay.hidden = false;
    overlay.innerHTML = `<div class="inner">Could not load ${esc(opt.label)}: ${esc(e.message || e)}</div>`;
  }
}

function playCharAnim(name) {
  CV.stage.setFollow(null);
  if (!CV.mixer) return;
  CV.mixer.stopAllAction();
  if (!name) {
    // back to the authored rest pose (Skeleton.pose() would re-apply the armature scale to the root bone)
    for (const [b, p, q, sc] of CV.rest) {
      b.position.copy(p);
      b.quaternion.copy(q);
      b.scale.copy(sc);
    }
    frameCharacter(false);
    return;
  }
  const clip = CV.asset.animations.find((a) => a.name === name);
  if (!clip) return;
  CV.mixer.clipAction(clip).reset().setLoop(THREE.LoopRepeat, Infinity).play();
  const hips = CV.rig?.hips;
  const p = new THREE.Vector3();
  CV.stage.setFollow(hips ? () => hips.getWorldPosition(p) : null);
}

function frameCharacter(instant) {
  const h = CV.height || D.character.height_m || 1.8;
  const target = new THREE.Vector3(0, h * 0.55, 0);
  if (CV.stage.follow) target.copy(CV.stage.follow());
  CV.stage.flyTo(target, new THREE.Vector3(1.6, 0.35, 3.3).multiplyScalar(h / 1.8), 35, instant);
}

// ------------------------------------------------------------------------------------------------ moveset
function renderMoveset() {
  const groups = clipsByStroke();
  if (!groups.length) {
    $('#view-moveset').innerHTML = `<div class="section-head"><div><h1>Moveset</h1>
      <p>No stroke clips yet: run <code>r3d select</code>, then <code>r3d remote clips</code>.</p></div></div>`;
    return;
  }
  $('#view-moveset').innerHTML = `
    <div class="moveset">
      <aside class="clip-list" aria-label="Clips">
        ${D.curation ? `<p class="clip-list-note"><span class="badge kept">kept</span> clips make the moveset;
          the ones curate rejected stay reviewable under each stroke.</p>` : ''}
        <div class="clip-groups">${groups.map(([stroke, g]) => `
          <div class="clip-group" data-stroke="${esc(stroke)}">
            <div class="clip-group-head"><h3>${esc(cap(stroke))}</h3>
              <span>${D.curation ? `<b>${g.kept.length}</b> kept${g.rejected.length ? ` · ${g.rejected.length} rejected` : ''}` : g.kept.length}</span></div>
            ${g.kept.length ? `<div class="clip-cards">${g.kept.map(clipCard).join('')}</div>`
              : '<p class="clip-none">No clip kept for this stroke.</p>'}
            ${g.rejected.length ? `<details class="rejected-list" data-stroke="${esc(stroke)}">
              <summary>Rejected by curate <span>(${g.rejected.length})</span></summary>
              <div class="clip-cards">${g.rejected.map(clipCard).join('')}</div>
            </details>` : ''}
          </div>`).join('')}
        </div>
      </aside>
      <div class="review">
        <div class="review-head" id="mv-head"></div>
        <div class="triple">
          <div class="panes">
            <figure class="pane pane-src">
              <video id="mv-src" muted playsinline preload="auto" disablepictureinpicture></video>
              <div class="pane-label"><span class="sw" style="background:var(--text)"></span>Broadcast source</div>
              <div class="pane-pending" id="mv-src-pending" hidden></div>
              <div class="frame-badge" id="mv-badge"></div>
            </figure>
            <figure class="pane pane-ovl">
              <video id="mv-ovl" muted playsinline preload="auto" disablepictureinpicture></video>
              <div class="pane-label"><span class="sw" style="background:var(--left)"></span>GVHMR skeleton overlay</div>
              <div class="pane-pending" id="mv-ovl-pending" hidden></div>
            </figure>
            <figure class="pane pane-3d">
              <div class="stage" id="mv-stage"></div>
              <div class="pane-label"><span class="sw"></span><span id="mv-3d-label">Character</span></div>
              <div class="pane-tools">
                <div class="seg" id="mv-cam" role="group" aria-label="Camera">
                  <button type="button" data-cam="follow" title="Orbit around her, target on the hips">Follow</button>
                  <button type="button" data-cam="broadcast" title="Behind and above, from where GVHMR placed the TV camera">Broadcast</button>
                  <button type="button" data-cam="side" title="From her right side">Side</button>
                </div>
                <div class="seg" id="mv-layers" role="group" aria-label="Layers">
                  <button type="button" data-layer="char">Character</button>
                  <button type="button" data-layer="skel">GVHMR</button>
                </div>
              </div>
              <div class="stage-overlay" id="mv-overlay" hidden></div>
              <div class="stage-note" id="mv-note"></div>
            </figure>
          </div>
        </div>
        <div class="transport">
          <div class="group">
            <button class="btn icon" id="tp-back" type="button" title="Previous frame (Left)" aria-label="Previous frame">${ICON.back}</button>
            <button class="btn icon play" id="tp-play" type="button" title="Play / pause (Space)" aria-label="Play">${ICON.play}</button>
            <button class="btn icon" id="tp-fwd" type="button" title="Next frame (Right)" aria-label="Next frame">${ICON.fwd}</button>
          </div>
          <div class="scrub" id="tp-scrub" tabindex="0" role="slider" aria-label="Frame" aria-valuemin="0">
            <div class="scrub-track"><div class="scrub-fill" id="tp-fill"></div></div>
            <div class="scrub-ticks" id="tp-ticks"></div>
            <div class="scrub-contact" id="tp-contact-mark"></div>
            <div class="scrub-thumb" id="tp-thumb"></div>
          </div>
          <div class="readout" id="tp-readout"></div>
        </div>
        <div class="transport-row2">
          <div class="seg" id="tp-rate" role="group" aria-label="Playback speed">
            <button type="button" data-rate="0.25">0.25×</button>
            <button type="button" data-rate="0.5">0.5×</button>
            <button type="button" data-rate="1" class="active">1×</button>
          </div>
          <button class="btn small on" id="tp-loop" type="button">${ICON.loop}Loop</button>
          <button class="btn small" id="tp-contact" type="button">${ICON.contact}Contact frame</button>
          <button class="btn small" id="tp-sound" type="button">${ICON.mute}<span>Sound off</span></button>
          <div class="hints">
            <span><span class="kbd">Space</span> play</span>
            <span><span class="kbd">←</span> <span class="kbd">→</span> frame (Shift ×5)</span>
            <span><span class="kbd">C</span> contact</span>
            <span><span class="kbd">↑</span> <span class="kbd">↓</span> clip</span>
          </div>
        </div>
        <div class="meta-grid" id="mv-meta"></div>
      </div>
    </div>`;
}

function clipCard(c) {
  const m = c.meta || {};
  const cu = c.curation;
  const rejected = isRejected(c);
  const dur = isNum(m.n_frames) ? `${(m.n_frames / (m.play_fps || 25)).toFixed(1)} s` : '';
  const cam = c.motion ? (c.motion.static_camera ? 'static cam' : 'moving cam') : '';
  const qa = (c.qa?.status || '').toUpperCase();
  const qaCls = { PASS: 'ok', WARN: 'warn', FAIL: 'bad' }[qa] || '';
  return `<a class="clip-card${rejected ? ' rejected' : ' kept'}" href="#/moveset/${esc(c.id)}" data-clip="${esc(c.id)}"
    title="${esc(cu?.why || c.id)}">
    <div class="thumb">${c.media.track ? `<img src="${esc(c.media.track)}" alt="" loading="lazy">` : ''}<span class="dur">${dur}</span></div>
    <div>
      <div class="title"><span>${esc(c.id)}</span>${cu ? '' : badgeFor(c.qa?.status).replace('QA pending', 'QA –')}</div>
      <div class="sub">${esc(viewLabel(m.view))} · ${mmss(m.t_start, 0)}${cam ? ` · ${cam}` : ''}</div>
      <div class="flags">
        ${curationBadge(c)}
        ${cu ? `<span class="mini ${qaCls}">QA ${esc(qa.toLowerCase() || '–')}</span>` : ''}
        ${rejected ? '' : (c.anim ? '<span class="mini ok" title="retargeted: in the moveset GLB">anim</span>'
    : `<span class="mini ${c.motion ? 'ok' : ''}">GVHMR</span><span class="mini">anim</span>`)}
        ${c.track && c.track.coverage < 0.7 ? '<span class="mini warn">low cover</span>' : ''}
      </div>
      ${(rejected || cu?.category === 'flagged') && cu?.short ? `<div class="why ${esc(cu.category)}">${esc(cu.short)}</div>` : ''}
    </div>
  </a>`;
}

function ensureMoveset() {
  if (MV) {
    MV.stage?.setActive(true);
    return;
  }
  const tl = new Timeline();
  MV = { tl, stage: null, char: null, asset: null, rig: null, mixer: null, action: null, skel: null, clip: null,
    cam: 'follow', want: { char: true, skel: false }, ref: null, animStart: 0, animScale: 1, token: 0,
    muted: true, ready: Promise.resolve(), error: null };
  try {
    MV.stage = new Stage($('#mv-stage'));
  } catch (e) {
    MV.error = `WebGL is not available (${e.message})`;
  }
  tl.on(updateTransport);
  if (MV.stage) {
    const p = new THREE.Vector3();
    MV.stage.onFrame(apply3D);
    MV.stage.setFollow(() => followPoint(p));
    MV.stage.setActive(true);
    const url = D.moveset.url;
    if (url) {
      MV.ready = loadAsset(url, 'glb', (l, t) => setMvOverlay(t ? `Loading the moveset… ${Math.round((l / t) * 100)}%` : 'Loading the moveset…'))
        .then((asset) => {
          const obj = instantiate(asset);
          fitUnits(obj, D.character.height_m);
          MV.stage.add(obj);
          Object.assign(MV, { asset, char: obj, rig: findRig(obj), mixer: new THREE.AnimationMixer(obj) });
        })
        .catch((e) => { MV.error = `Could not load the moveset: ${e.message || e}`; });
    }
  }
  wireTransport();
}

function setMvOverlay(text) {
  const o = $('#mv-overlay');
  if (!text) { o.hidden = true; return; }
  o.hidden = false;
  o.innerHTML = `<div class="inner">${text.startsWith('Loading') ? '<div class="spinner"></div>' : ''}<span>${esc(text)}</span></div>`;
}

function wireTransport() {
  const tl = MV.tl;
  $('#tp-play').addEventListener('click', () => tl.toggle());
  $('#tp-back').addEventListener('click', () => tl.step(-1));
  $('#tp-fwd').addEventListener('click', () => tl.step(1));
  $('#tp-contact').addEventListener('click', () => { tl.pause(false); tl.seekFrame(tl.contact); });
  $('#tp-loop').addEventListener('click', (e) => {
    tl.setLoop(!tl.loop);
    e.currentTarget.classList.toggle('on', tl.loop);
  });
  $('#tp-sound').addEventListener('click', (e) => {
    MV.muted = !MV.muted;
    tl.setMuted(MV.muted);
    // only the broadcast preview carries sound (the ball strike); the overlay stays muted
    $('#mv-ovl').muted = true;
    e.currentTarget.classList.toggle('on', !MV.muted);
    e.currentTarget.innerHTML = `${MV.muted ? ICON.mute : ICON.sound}<span>${MV.muted ? 'Sound off' : 'Sound on'}</span>`;
  });
  $('#tp-rate').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-rate]');
    if (b) setRate(Number(b.dataset.rate));
  });
  $('#mv-cam').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-cam]');
    if (b) applyCam(b.dataset.cam, false);
  });
  $('#mv-meta').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-seek]');
    if (!b) return;
    tl.pause(false);
    tl.seekFrame(Number(b.dataset.seek));
    $('#view-moveset .triple').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  });
  $('#mv-layers').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-layer]');
    if (!b || b.disabled) return;
    MV.want[b.dataset.layer] = !MV.want[b.dataset.layer];
    if (!MV.want.char && !MV.want.skel) MV.want[b.dataset.layer === 'char' ? 'skel' : 'char'] = true;
    applyLayers();
  });

  const scrub = $('#tp-scrub');
  let dragging = false;
  const seekX = (e) => {
    const r = scrub.getBoundingClientRect();
    const x = clamp((e.clientX - r.left) / r.width, 0, 1);
    tl.seekFrame(Math.round(x * (tl.n - 1)));
  };
  scrub.addEventListener('pointerdown', (e) => {
    dragging = true;
    scrub.setPointerCapture(e.pointerId);
    tl.pause(false);
    seekX(e);
  });
  scrub.addEventListener('pointermove', (e) => { if (dragging) seekX(e); });
  const stop = (e) => {
    dragging = false;
    if (scrub.hasPointerCapture(e.pointerId)) scrub.releasePointerCapture(e.pointerId);
  };
  scrub.addEventListener('pointerup', stop);
  scrub.addEventListener('pointercancel', stop);

  const keys = (e) => {
    if (S.view !== 'moveset' || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target.closest?.('input, select, textarea')) return;
    if (e.key === ' ') {
      e.preventDefault(); // keeps a focused button from also activating on Space
      if (e.type === 'keydown' && !e.repeat) tl.toggle();
      return;
    }
    if (e.type !== 'keydown') return;
    const k = e.key;
    if (k === 'ArrowLeft' || k === 'ArrowRight') {
      e.preventDefault();
      tl.step((k === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 5 : 1));
    } else if (k === 'ArrowUp' || k === 'ArrowDown') {
      e.preventDefault();
      const ids = navOrder(D.clipById[S.clip]);
      const i = ids.indexOf(S.clip);
      const next = ids[clamp(i + (k === 'ArrowUp' ? -1 : 1), 0, ids.length - 1)];
      if (next && next !== S.clip) location.hash = `#/moveset/${next}`;
    } else if (k === 'c' || k === 'C') {
      tl.pause(false);
      tl.seekFrame(tl.contact);
    } else if (k === 'Home') tl.step(-tl.n);
    else if (k === 'End') tl.step(tl.n);
    else if (k === '1') setRate(0.25);
    else if (k === '2') setRate(0.5);
    else if (k === '3') setRate(1);
    else if (k === 'l' || k === 'L') $('#tp-loop').click();
  };
  document.addEventListener('keydown', keys);
  document.addEventListener('keyup', keys);
}

function setRate(rate) {
  MV.tl.setRate(rate);
  $$('#tp-rate button').forEach((b) => b.classList.toggle('active', Number(b.dataset.rate) === rate));
}

async function selectClip(id) {
  const c = D.clipById[id] || D.clipById[defaultClipId()];
  if (!c || !MV) return;
  if (MV.clip === c.id) return;
  MV.clip = c.id;
  S.clip = c.id;
  const token = ++MV.token;
  // a rejected clip opened by link: unfold its stroke's "Rejected by curate" list so the card shows
  const list = $(`.clip-card[data-clip="${CSS.escape(c.id)}"]`)?.closest('details');
  if (list) list.open = true;
  for (const card of $$('.clip-card')) {
    const on = card.dataset.clip === c.id;
    card.classList.toggle('active', on);
    if (on) card.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }
  const m = c.meta || {};
  const fps = m.play_fps || c.motion?.fps || 25;
  const n = m.n_frames || c.motion?.frames || 1;
  renderReviewHead(c);
  renderClipMeta(c);

  const videos = [];
  for (const [key, elId, pendId, label] of [
    ['preview', '#mv-src', '#mv-src-pending', 'Preview pending'],
    ['overlay', '#mv-ovl', '#mv-ovl-pending', 'GVHMR overlay pending'],
  ]) {
    const el = $(elId);
    const media = c.media[key];
    const pend = $(pendId);
    el.pause();
    if (media?.url) {
      el.hidden = false;
      pend.hidden = true;
      el.muted = key === 'overlay' ? true : MV.muted;
      el.src = media.url;
      el.load();
      videos.push({ el, start: media.start || 0 });
    } else {
      el.removeAttribute('src');
      el.load();
      el.hidden = true;
      pend.hidden = false;
      pend.style.backgroundImage = c.media.track ? `url("${c.media.track}")` : '';
      pend.innerHTML = `<span>${esc(label)}</span>`;
    }
  }
  MV.tl.load({ fps, n, contact: m.contact_index ?? 0, videos });
  setRate(MV.tl.rate);
  const ticks = [];
  for (let f = 0; f < n; f += 5) ticks.push(`<i style="left:${(f / Math.max(1, n - 1)) * 100}%"></i>`);
  $('#tp-ticks').innerHTML = ticks.join('');
  const cx = MV.tl.contact / Math.max(1, n - 1);
  $('#tp-contact-mark').style.left = `${cx * 100}%`;
  // a contact late in the clip puts its label left of the mark, so it never runs past the scrub (phone width)
  $('#tp-contact-mark').classList.toggle('flip', cx > 0.7);
  $('#tp-scrub').setAttribute('aria-valuemax', String(n - 1));

  if (!MV.stage) {
    setMvOverlay(MV.error || 'WebGL is not available');
    return;
  }
  await MV.ready;
  if (token !== MV.token) return;
  await setup3D(c, fps, n, token);
}

async function setup3D(c, fps, n, token) {
  const stage = MV.stage;
  MV.action?.stop();
  MV.action = null;
  const clip = c.anim && MV.asset ? MV.asset.animations.find((a) => a.name === c.anim) : null;
  if (clip) {
    MV.action = MV.mixer.clipAction(clip);
    MV.action.setLoop(THREE.LoopOnce, 1);
    MV.action.clampWhenFinished = true;
    MV.action.play();
    // clip frames are 1:1 with the video; tolerate an exporter that offsets or resamples the timeline
    const starts = clip.tracks.map((t) => t.times[0]);
    const ends = clip.tracks.map((t) => t.times[t.times.length - 1]);
    const start = starts.length ? Math.min(...starts) : 0;
    const span = (ends.length ? Math.max(...ends) : clip.duration) - start;
    const expected = (n - 1) / fps;
    MV.animStart = start;
    MV.animScale = expected > 0 && Math.abs(span - expected) > 1.5 / fps ? span / expected : 1;
  }

  if (MV.skel) {
    stage.remove(MV.skel.group);
    MV.skel.dispose();
    MV.skel = null;
  }
  if (c.skeleton) {
    try {
      setMvOverlay('Loading GVHMR joints…');
      const data = await fetchJSON(c.skeleton);
      if (token !== MV.token) return;
      MV.skel = new JointSkeleton(data, stage);
      stage.add(MV.skel.group);
    } catch (e) {
      console.warn(e);
    }
  }
  MV.hasAnim = !!clip;
  const k = clamp(c.meta?.contact_index ?? 0, 0, n - 1);

  // the body frame at contact anchors the camera presets and lines the GVHMR joints up with the character
  let ref = null;
  if (clip && MV.rig?.hips && MV.rig.left && MV.rig.right) {
    poseAt(k / fps);
    MV.char.updateMatrixWorld(true);
    const w = (b) => b.getWorldPosition(new THREE.Vector3());
    ref = bodyFrame(w(MV.rig.hips), w(MV.rig.left), w(MV.rig.right));
  }
  if (MV.skel) {
    const fs = MV.skel.frameAt(k);
    const g = MV.skel.group;
    g.position.set(0, 0, 0);
    g.rotation.set(0, 0, 0);
    g.scale.setScalar(1);
    if (ref) {
      // overlay GVHMR on the character: the retarget's body scale, pelvis on the hips at contact. GVHMR's own
      // ground sits up to ~20 cm off and the retarget re-grounds every frame, so away from contact the
      // vertical gap is exactly what the retarget corrected.
      const s = isNum(c.retarget?.scale) ? c.retarget.scale : 1;
      const yaw = Math.atan2(ref.forward.x, ref.forward.z) - Math.atan2(fs.forward.x, fs.forward.z);
      g.scale.setScalar(s);
      g.rotation.y = yaw;
      const p = fs.origin.clone().multiplyScalar(s).applyAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
      g.position.subVectors(ref.origin, p);
    }
    g.updateMatrixWorld(true);
    if (!ref) ref = { ...fs, origin: fs.origin.clone().applyMatrix4(g.matrixWorld) };
  }
  MV.ref = ref;
  MV.cameraRel = c.motion?.camera_rel || null;
  const cam = c.retarget?.camera_at_contact?.position_zup;
  MV.cameraWorld = Array.isArray(cam) ? new THREE.Vector3(cam[0], cam[2], -cam[1]) : null; // Blender Z-up → glTF
  applyLayers();
  apply3D();
  applyCam(MV.cam, true);

  const rejected = isRejected(c);
  const label = MV.hasAnim ? `Character · ${c.anim}`
    : (MV.skel ? (rejected ? 'GVHMR joints · not retargeted' : 'GVHMR joints (retarget pending)') : '3D pending');
  $('#mv-3d-label').textContent = label;
  if (!MV.hasAnim && !MV.skel) setMvOverlay(MV.error || 'No motion for this clip yet');
  else setMvOverlay(MV.error && !MV.skel ? MV.error : null);
  const notes = [];
  if (!MV.hasAnim) {
    notes.push(rejected ? 'rejected by curate: not in the moveset'
      : D.moveset.url ? 'no animation for this clip in the moveset' : 'moveset not exported yet');
  }
  if (MV.animScale !== 1) notes.push(`animation retimed ×${MV.animScale.toFixed(2)}`);
  $('#mv-note').innerHTML = notes.map((t) => `<span>${esc(t)}</span>`).join('');
}

function applyLayers() {
  const charOk = !!MV.hasAnim;
  const showChar = charOk && MV.want.char;
  const showSkel = !!MV.skel && (MV.want.skel || !showChar);
  if (MV.char) MV.char.visible = showChar;
  if (MV.skel) MV.skel.group.visible = showSkel;
  $$('#mv-layers button').forEach((b) => {
    const layer = b.dataset.layer;
    b.disabled = layer === 'char' ? !charOk : !MV.skel;
    b.classList.toggle('active', layer === 'char' ? showChar : showSkel);
    b.title = layer === 'char'
      ? (charOk ? '' : 'No retargeted animation for this clip yet')
      : (charOk ? 'GVHMR joints, pinned to her hips at the contact frame' : 'GVHMR joints in their own world frame');
  });
}

function followPoint(out) {
  if (MV.char?.visible && MV.rig?.hips) return MV.rig.hips.getWorldPosition(out);
  if (MV.skel?.group.visible) return MV.skel.pelvisWorld(out);
  return null;
}

function apply3D() {
  if (!MV?.stage) return;
  const tl = MV.tl;
  const t = tl.playing ? tl.t : tl.frame / tl.fps;
  poseAt(Math.min(t, (tl.n - 1) / tl.fps));
  MV.skel?.setFrame(t * tl.fps);
}

/** Pose the character at clip time t. Writing action.time and updating by 0 is exact; mixer.setTime() would
 *  run the LoopOnce end-of-clip logic, which pauses the action and then pins it to frame 0 for good. */
function poseAt(t) {
  if (!MV.action) return;
  MV.action.time = clamp(MV.animStart + t * MV.animScale, 0, MV.action.getClip().duration);
  MV.mixer.update(0);
}

function applyCam(mode, instant) {
  MV.cam = mode;
  $$('#mv-cam button').forEach((b) => b.classList.toggle('active', b.dataset.cam === mode));
  if (!MV.stage || !MV.ref) return;
  const ref = MV.ref;
  const target = followPoint(new THREE.Vector3()) || ref.origin.clone();
  let dir;
  let dist = 4.6;
  let fov = 35;
  if (mode === 'broadcast') {
    // azimuth of the TV camera as GVHMR recovered it, kept clearly elevated like the real gantry camera
    const v = MV.cameraWorld && MV.hasAnim
      ? MV.cameraWorld.clone().sub(ref.origin)
      : fromBody(ref, MV.cameraRel?.pos || [0, 0.4, -1]);
    const h = Math.hypot(v.x, v.z) || 1;
    const elev = clamp(Math.atan2(v.y, h), (12 * Math.PI) / 180, (35 * Math.PI) / 180);
    dir = new THREE.Vector3((v.x / h) * Math.cos(elev), Math.sin(elev), (v.z / h) * Math.cos(elev));
    // the TV camera's own distance and vertical field of view: she fills the pane as she fills the broadcast
    dist = clamp(v.length(), 8, 30);
    fov = MV.cameraRel?.vfov_deg || 22;
  } else if (mode === 'side') {
    dir = fromBody(ref, [1, 0.12, 0]).normalize();
    dist = 6.2;
  } else {
    dir = fromBody(ref, [0.55, 0.24, 1]).normalize();
  }
  MV.stage.flyTo(target, dir.multiplyScalar(dist), fov, instant);
}

function updateTransport(tl) {
  const f = tl.frame;
  const n = tl.n;
  const pos = n > 1 ? clamp((tl.playing ? tl.t * tl.fps : f) / (n - 1), 0, 1) : 0;
  $('#tp-fill').style.width = `${pos * 100}%`;
  $('#tp-thumb').style.left = `${pos * 100}%`;
  $('#tp-scrub').setAttribute('aria-valuenow', String(f));
  const d = f - tl.contact;
  const rel = d === 0 ? '<span class="rel at">at contact</span>'
    : `<span class="rel">${d > 0 ? '+' : ''}${d} f from contact</span>`;
  $('#tp-readout').innerHTML = `<b>f ${String(f).padStart(3, '0')}</b> / ${String(n - 1).padStart(3, '0')} · ${(f / tl.fps).toFixed(2)} s${rel}`;
  const play = $('#tp-play');
  if (play.dataset.state !== String(tl.playing)) {
    play.dataset.state = String(tl.playing);
    play.innerHTML = tl.playing ? ICON.pause : ICON.play;
    play.setAttribute('aria-label', tl.playing ? 'Pause' : 'Play');
  }
  const badge = $('#mv-badge');
  badge.textContent = d === 0 ? `contact · f ${f}` : `f ${f}`;
  badge.classList.toggle('contact', d === 0);
}

function renderReviewHead(c) {
  const m = c.meta || {};
  const cu = c.curation;
  const ids = navOrder(c);
  const i = ids.indexOf(c.id);
  const prev = ids[i - 1];
  const next = ids[i + 1];
  $('#mv-head').innerHTML = `
    <h1>${esc(c.id)}</h1>
    <div class="chips">
      <span class="chip">${esc(cap(m.stroke))}</span>
      <span class="chip"><span class="sw" style="background:${viewColor(m.view)}"></span>${esc(viewLabel(m.view))}</span>
      <span class="chip">${esc(mmss(m.t_start))} in the source</span>
      ${c.motion ? `<span class="chip">camera ${c.motion.static_camera ? 'static' : 'moving'}</span>` : ''}
      ${curationBadge(c, true)}
      ${badgeFor(c.qa?.status)}
    </div>
    <div class="nav">
      <a class="btn icon" ${prev ? `href="#/moveset/${esc(prev)}"` : 'aria-disabled="true"'} title="Previous ${isKept(c) ? 'kept ' : ''}clip (Up)" aria-label="Previous clip">${ICON.prev}</a>
      <a class="btn icon" ${next ? `href="#/moveset/${esc(next)}"` : 'aria-disabled="true"'} title="Next ${isKept(c) ? 'kept ' : ''}clip (Down)" aria-label="Next clip">${ICON.next}</a>
    </div>
    ${cu && !cu.kept ? `<div class="cur-banner ${esc(cu.category)}"><b>Rejected by curate, not in the moveset.</b>
      ${esc(cu.why)} Kept for review: the source, GVHMR overlay and joints still play.</div>` : ''}
    ${cu?.category === 'flagged' ? `<div class="cur-banner flagged"><b>Animated, but the stroke label is flagged.</b>
      ${esc(cu.why.replace(/^In the moveset \(every clip is animated\), but flagged: /, ''))}</div>` : ''}`;
}

function renderClipMeta(c) {
  const m = c.meta || {};
  const fps = m.play_fps || 25;
  const mo = c.motion;
  const yt = D.footage.youtube && isNum(m.t_start)
    ? ` <a href="${esc(D.footage.youtube)}&t=${Math.floor(m.t_start)}s" target="_blank" rel="noopener" title="Open the source on YouTube">${ICON.external.replace('<svg', '<svg width="12" height="12"')}</a>` : '';
  // each `curate` run appends "+motion" to the source: show every source once ("audio + motion")
  const contactSrc = [...new Set(String(m.contact_source || 'gemini').split('+'))].join(' + ');
  const clipCardHtml = `<div class="card card-pad meta-card"><h3>Clip</h3>${kv([
    ['Stroke', esc(cap(m.stroke))],
    ['View', `${esc(viewLabel(m.view))} · ${esc(m.side || '–')} side`],
    ['Source', `${mmss(m.t_start)} – ${mmss(m.t_start + (m.n_frames || 0) / fps)}${yt}`],
    ['Shot', isNum(m.shot_id) ? `<a href="#/footage/${m.shot_id}">#${m.shot_id}</a>` : '–'],
    ['Frames', `${num(m.n_frames)} @ ${num(fps)} fps · ${num((m.n_frames || 0) / fps, 2)} s`],
    ['Contact', `frame ${num(m.contact_index)} · ${esc(contactSrc)}`],
    isNum(m.contact_index_selected) && m.contact_index_selected !== m.contact_index
      ? ['Contact (select)', `frame ${num(m.contact_index_selected)}`] : null,
    isNum(m.onset_strength) && /audio/.test(contactSrc) ? ['Audio onset', `${num(m.onset_strength, 1)} σ`] : null,
    ['Gemini', `confidence ${num(m.confidence, 2)} · quality ${num(m.quality)}/5`],
    ['Hitter height', isNum(m.hitter_height_px) ? `${num(m.hitter_height_px)} px` : '–'],
    m.outfit ? ['Outfit', esc(m.outfit)] : null,
  ])}</div>`;

  const cal = c.camera_calib;
  const captureHtml = `<div class="card card-pad meta-card"><h3>Capture</h3>${kv([
    ['Track coverage', c.track ? `${pct(c.track.coverage)} · id ${c.track.track_id}` : 'pending'],
    ['Box height', c.track ? `${num(c.track.box_h_median_px)} px median` : '–'],
    ['Camera', mo ? `${mo.static_camera ? 'static' : 'moving'} · ${num(mo.camera_motion_px, 2)} px/frame` : 'pending'],
    ['Court calibration', cal ? `${pct(cal.ok_frac)} frames · ${num(cal.reproj_px, 1)} px` : (mo?.calibrated ? 'yes' : 'none')],
    // a calibration written after the motion was recovered has not changed the motion yet
    cal && mo ? ['Used by GVHMR', mo.calibrated ? 'yes' : '<span class="badge pending">not yet</span>'] : null,
    ['Lens', mo ? `${num(mo.hfov_deg, 1)}° · f ${num(mo.f_px)} px${mo.calibrated ? '' : ' (assumed)'}` : '–'],
  ])}</div>`;

  const motionHtml = `<div class="card card-pad meta-card"><h3>Motion · GVHMR</h3>${mo ? kv([
    ['Frames', `${num(mo.frames)} · ${num(mo.duration_s, 2)} s`],
    ['Hand speed at contact', `${num(mo.hand_speed_contact_ms, 1)} m/s`],
    ['Peak hand speed', `${num(mo.hand_speed_peak_ms, 1)} m/s`],
    ['Pelvis path', `${num(mo.pelvis_travel_m, 2)} m (net ${num(mo.pelvis_displacement_m, 2)} m)`],
    ['Planted frames', pct(mo.planted_frac)],
    ['Camera distance', mo.camera_rel ? `${num(mo.camera_rel.distance_m, 1)} m (estimated)` : '–'],
  ]) : pendingBox('motion.npz pending: r3d remote motion')}</div>`;

  const q = c.qa;
  let qaHtml;
  if (q) {
    const x = q.metrics || {};
    const g = (a, b) => x?.[a]?.[b];
    const rows = [
      ['Foot skate', isNum(g('skate', 'mean_cm')) ? `${num(g('skate', 'mean_cm'), 2)} cm/f (p95 ${num(g('skate', 'p95_cm'), 1)})` : null],
      ['Ground offset', isNum(g('ground', 'offset_cm')) ? `${num(g('ground', 'offset_cm'), 1)} cm` : null],
      ['Jitter', isNum(g('jitter', 'all_ms3')) ? `${num(g('jitter', 'all_ms3'))} m/s³` : null],
      ['Reprojection', isNum(g('reprojection', 'median_px')) ? `${num(g('reprojection', 'median_px'), 1)} px · PCK ${num(g('reprojection', 'pck'), 2)}` : null],
      ['Wrist peak vs contact', isNum(g('timing', 'offset_frames')) ? `${g('timing', 'offset_frames') > 0 ? '+' : ''}${g('timing', 'offset_frames')} f` : null],
      ['Retarget skate', isNum(g('retarget', 'mean_cm')) ? `${num(g('retarget', 'mean_cm'), 2)} cm/f` : null],
    ].filter(([, v]) => v);
    const generic = rows.length ? [] : flatScalars(x).slice(0, 10).map(([k, v]) => [k, esc(String(v))]);
    const reasons = (q.reasons || []).map((r) => {
      const lvl = /^FAIL/.test(r) ? 'fail' : /^WARN/.test(r) ? 'warn' : '';
      return `<li class="${lvl}">${esc(r.replace(/^(FAIL|WARN):\s*/, ''))}</li>`;
    }).concat((q.notes || []).map((r) => `<li>${esc(
      // QA writes "not retargeted yet" for every clip without an animation; a rejected clip never will be
      isRejected(c) && /^not retargeted/.test(r) ? 'not retargeted: rejected by curate' : r)}</li>`));
    qaHtml = `<div class="card card-pad meta-card"><h3>QA ${badgeFor(q.status)}</h3>${kv([...rows, ...generic])}
      ${reasons.length ? `<ul class="reasons">${reasons.join('')}</ul>` : ''}</div>`;
  } else {
    qaHtml = `<div class="card card-pad meta-card"><h3>QA ${badgeFor(null)}</h3>${pendingBox('qa.json pending: r3d qa')}</div>`;
  }

  const rt = c.retarget;
  const cu = c.curation;
  const known = rt && typeof rt === 'object' ? [
    isNum(rt.scale) ? ['Scale to rig', `×${num(rt.scale, 3)}`] : null,
    isNum(rt.ground_offset_m) ? ['Ground offset', `${num(rt.ground_offset_m * 100, 1)} cm · ${esc(rt.ground_method || '')}`] : null,
    isNum(rt.contact_frames) ? ['Planted frames', num(rt.contact_frames)] : null,
    rt.skate_cm_per_frame ? ['Foot skate', `${num(rt.skate_cm_per_frame.before, 2)} → ${num(rt.skate_cm_per_frame.after, 2)} cm/f`] : null,
    rt.lowest_foot_cm ? ['Lowest foot (median)', `${num(rt.lowest_foot_cm.p50, 1)} cm`] : null,
    isNum(rt.fk_error_mm) ? ['FK error', `${num(rt.fk_error_mm, 1)} mm`] : null,
  ].filter(Boolean) : [];
  const rows = known.length || !rt ? known
    : flatScalars(rt).filter(([k]) => k !== 'id').slice(0, 8).map(([k, v]) => [k, esc(String(v))]);
  const rtWarn = (rt?.warnings || []).map((w) => `<li class="warn">${esc(w)}</li>`).join('');
  const retargetHtml = `<div class="card card-pad meta-card"><h3>Retarget${c.anim ? ' <span class="badge pass">animated</span>' : ''}</h3>${kv([
    ['Animation', c.anim ? `<code>${esc(c.anim)}</code>` : (isRejected(c) ? 'not retargeted (rejected by curate)' : 'pending')],
    ...rows,
  ])}${rtWarn ? `<ul class="reasons">${rtWarn}</ul>` : ''}</div>`;

  // curate: does the motion match the label, and where the clip ranks in its stroke
  let curateHtml = '';
  if (cu) {
    const th = D.curation?.thresholds || {};
    const overhead = ['serve', 'overhead'].includes(cu.stroke);
    const check = isNum(cu.measure) ? (overhead
      ? `wrist ${cu.measure >= 0 ? '+' : ''}${num(cu.measure, 2)} m above head <span class="faint">(≥ ${num(th.overhead_min_rise_m ?? 0.25, 2)})</span>`
      : `hand ${num(cu.measure, 1)} m/s at contact <span class="faint">(≥ ${num(th.swing_min_speed_ms ?? 4, 1)})</span>`) : '–';
    const ok = cu.label_ok ?? (cu.category !== 'label' && cu.category !== 'qa');
    curateHtml = `<div class="card card-pad meta-card cur-card ${cu.kept ? 'is-kept' : 'is-rejected'}"><h3>Curate ${curationBadge(c, true)}</h3>${kv([
      ['Motion vs label', `${ok ? '<span class="badge pass">matches</span>' : '<span class="badge fail">mismatch</span>'} ${check}`],
      ['Score', `${num(cu.score, 2)}${cu.rank ? ` · #${cu.rank} in ${esc(cu.stroke)}` : ''}`],
      ['QA gate', `${badgeFor(cu.qa)}${cu.qa_at_curate && cu.qa_at_curate !== cu.qa ? ` <span class="faint">(${esc(cu.qa_at_curate)} at curate)</span>` : ''}`],
      ['Contact', `frame ${num(cu.contact_index)}`],
    ])}<p class="cur-why">${esc(cu.why)}</p></div>`;
  }

  // signature: this clip next to her mean for the stroke
  const stroke = m.stroke;
  const sigRow = D.signature?.by_stroke?.[stroke];
  let sigHtml = '';
  if (D.signature) {
    const mine = c.signature;
    const body = sigRow ? `<table class="data sig-table">
        <thead><tr><th></th><th class="num">This clip</th><th class="num">Her ${esc(stroke)}</th></tr></thead>
        <tbody>${sigMetrics(sigRow).map((mt) => {
          const v = mine?.[mt.key];
          const st = sigRow[mt.key];
          const z = isNum(v) && st.sd > 0 ? (v - st.mean) / st.sd : null;
          return `<tr title="${esc(mt.hint)}"><td><b>${esc(mt.label)}</b><span class="u-sub">${esc(mt.unit)}</span></td>
            <td class="num"><b>${sigVal(mt, v)}</b>${isNum(v) ? esc(mt.suffix) : ''}${isNum(z) ? `<span class="z${Math.abs(z) > 1.5 ? ' far' : ''}">${z >= 0 ? '+' : ''}${num(z, 1)} sd</span>` : ''}</td>
            <td class="num">${sigMeanSd(mt, st)}</td></tr>
            <tr class="strip-row"><td colspan="3">${sigStrip(mt, stroke, c.id)}</td></tr>`;
        }).join('')}</tbody></table>`
      : pendingBox(`No ${stroke} signature: no ${stroke} clip was kept`);
    sigHtml = `<div class="card card-pad meta-card sig-meta"><h3>Stroke signature <a class="more" href="#/overview/signature">all strokes →</a></h3>
      ${!mine && sigRow ? `<p class="faint sig-note">Not measured for this clip: the signature uses only clips whose motion matches their label. Her ${esc(stroke)} mean (n = ${num(sigRow.n)}):</p>` : ''}
      ${body}</div>`;
  }

  // review renders are wide strips (source | retarget | overlay | side, or the court fit): full width, own aspect
  const strips = [
    ...(c.checks || []).map((ck) => ({ ...ck, caption: isNum(ck.frame) ? `retarget check · frame ${ck.frame}` : ck.name })),
    ...(c.media.camera ? [{ url: c.media.camera, kind: 'image', caption: 'court-line calibration' }] : []),
  ];
  const stripsHtml = strips.length ? `<div class="card card-pad meta-card wide"><h3>Review renders</h3><div class="strips">${strips.map((ck) => `
    <div class="strip">${ck.kind === 'video'
      ? `<video src="${esc(ck.url)}" muted loop playsinline controls preload="metadata"></video>`
      : `<figure class="img-card" data-full="${esc(ck.url)}" data-caption="${esc(ck.caption)}"><img src="${esc(ck.url)}" alt="${esc(ck.caption)}" loading="lazy"></figure>`}
      <div class="strip-foot"><span>${esc(ck.caption)}</span>${isNum(ck.frame)
        ? `<button class="btn small" type="button" data-seek="${ck.frame}">${ICON.contact}Show frame ${ck.frame}</button>` : ''}</div></div>`).join('')}
    </div></div>` : '';

  $('#mv-meta').innerHTML = clipCardHtml + curateHtml + sigHtml + motionHtml + captureHtml + qaHtml + retargetHtml + stripsHtml;
}

function flatScalars(obj, prefix = '', out = []) {
  for (const [k, v] of Object.entries(obj || {})) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v == null) continue;
    if (typeof v === 'object' && !Array.isArray(v)) flatScalars(v, key, out);
    else if (typeof v !== 'object') out.push([key.replace(/_/g, ' '), isNum(v) ? +v.toFixed(3) : v]);
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ footage
const F = { views: new Set(Object.keys(VIEW_META)), stroke: '', her: false, noReplay: false, selected: false, usable: false };

function renderFootage() {
  const f = D.footage;
  const dur = f.probe?.duration || (f.shots.length ? f.shots[f.shots.length - 1].end : 1);
  const strokes = Object.keys(f.strokes || {}).filter((s) => f.strokes[s] > 0);
  const ticks = [];
  for (let t = 0; t <= dur; t += 300) ticks.push(`<span style="left:${(t / dur) * 100}%">${Math.round(t / 60)}:00</span>`);
  const marks = D.clips.filter((c) => isNum(c.meta?.contact_frame)).map((c) => {
    const t = c.meta.contact_frame / (c.meta.src_fps || 25);
    return `<a href="#/moveset/${esc(c.id)}"${isRejected(c) ? ' class="rejected"' : ''} style="left:${(t / dur) * 100}%" data-tip="<b>${esc(c.id)}</b> · ${isRejected(c) ? 'rejected by curate' : 'kept'}<br>contact ${mmss(t)}" aria-label="${esc(c.id)}"></a>`;
  }).join('');
  $('#view-footage').innerHTML = `
    <div class="section-head"><div>
      <h1>Footage index</h1>
      <p>Every shot of the ${isNum(f.minutes) ? `${num(f.minutes, 1)}-minute ` : ''}compilation as ${esc(f.index_model || 'Gemini')} labelled it: camera view,
        replay, her court side, capture quality and each stroke with its contact time. Shots that feed a moveset clip are
        highlighted; the triangles under the timeline mark each clip's contact.</p>
    </div></div>
    <div class="card timeline-card">
      <div class="legend" style="margin-bottom:12px">${Object.entries(VIEW_META).map(([v, meta]) => `
        <span class="chip"><span class="sw" style="background:${meta.color}"></span>${meta.label}
        <span class="faint">${num(f.by_view?.[v]?.shots || 0)} · ${mmss(f.by_view?.[v]?.seconds || 0, 0)}</span></span>`).join('')}
        <span class="chip"><span class="sw" style="background:var(--accent)"></span>moveset clip${D.curation ? ' (kept)' : ''}</span>
        ${D.curation ? '<span class="chip"><span class="sw" style="background:var(--text-3)"></span>rejected by curate</span>' : ''}
      </div>
      <div class="timeline" id="ft-timeline">${f.shots.map((s) => {
        const v = s.label?.view;
        return `<div class="seg-shot" data-shot="${s.id}" style="left:${(s.start / dur) * 100}%;width:${((s.end - s.start) / dur) * 100}%;background:${viewColor(v)}"></div>`;
      }).join('')}</div>
      <div class="timeline-marks">${marks}</div>
      <div class="timeline-axis">${ticks.join('')}</div>
    </div>
    <div class="filters" id="ft-filters">
      <div class="legend" id="ft-views">${Object.entries(VIEW_META).map(([v, meta]) => `
        <button type="button" class="chip active" data-view="${v}"><span class="sw" style="background:${meta.color}"></span>${meta.label}</button>`).join('')}</div>
      <select id="ft-stroke" aria-label="Stroke">
        <option value="">Any stroke</option>
        ${strokes.map((s) => `<option value="${s}">${cap(s)} (${f.strokes[s]})</option>`).join('')}
      </select>
      <label class="toggle"><input type="checkbox" id="ft-her"> her strokes only</label>
      <label class="toggle"><input type="checkbox" id="ft-noreplay"> hide replays</label>
      <label class="toggle"><input type="checkbox" id="ft-usable"> usable for capture</label>
      <label class="toggle"><input type="checkbox" id="ft-selected"> moveset clips only</label>
      <span class="count" id="ft-count"></span>
    </div>
    <div class="card table-wrap">
      <table class="data shots" id="ft-table">
        <thead><tr><th>#</th><th>Time</th><th class="num">Length</th><th>View</th><th>Her side</th><th>Quality</th>
          <th>Flags</th><th>Her strokes</th><th class="num">Opp.</th><th>Clips</th></tr></thead>
        <tbody>${f.shots.map(shotRow).join('')}</tbody>
      </table>
    </div>`;

  $('#ft-views').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-view]');
    if (!b) return;
    const v = b.dataset.view;
    if (F.views.has(v)) F.views.delete(v); else F.views.add(v);
    b.classList.toggle('active', F.views.has(v));
    applyFilters();
  });
  $('#ft-stroke').addEventListener('change', (e) => { F.stroke = e.target.value; applyFilters(); });
  for (const [id, key] of [['#ft-her', 'her'], ['#ft-noreplay', 'noReplay'], ['#ft-usable', 'usable'], ['#ft-selected', 'selected']]) {
    $(id).addEventListener('change', (e) => { F[key] = e.target.checked; applyFilters(); });
  }
  $('#ft-table tbody').addEventListener('click', (e) => {
    if (e.target.closest('a')) return;
    const row = e.target.closest('tr.row');
    if (row) openShot(Number(row.dataset.shot));
  });
  const tl = $('#ft-timeline');
  tl.addEventListener('click', (e) => {
    const seg = e.target.closest('.seg-shot');
    if (seg) openShot(Number(seg.dataset.shot), true);
  });
  tl.addEventListener('mousemove', (e) => {
    const seg = e.target.closest('.seg-shot');
    if (!seg) return hideTip();
    const s = D.shotById[Number(seg.dataset.shot)];
    if (!s) return hideTip();
    const l = s.label;
    const her = l ? l.events.filter((ev) => ev.hitter === 'rybakina').length : 0;
    showTip(e, `<b>Shot #${s.id}</b> · ${mmss(s.start, 0)}–${mmss(s.end, 0)}<br>${l ? `${esc(viewLabel(l.view))} · Q${l.quality}${l.replay ? ' · replay' : ''} · ${her} of her strokes` : 'not labelled'}${s.clips.length ? `<br>clips: ${s.clips.join(', ')}` : ''}`);
  });
  tl.addEventListener('mouseleave', hideTip);
  applyFilters();
}

function shotRow(s) {
  const l = s.label;
  const her = l ? l.events.filter((e) => e.hitter === 'rybakina' && e.type !== 'other') : [];
  const opp = l ? l.events.filter((e) => e.hitter === 'opponent').length : 0;
  const flags = l ? [l.replay && 'replay', l.slow_motion && 'slow-mo', l.full_body && 'full body'].filter(Boolean) : [];
  return `<tr class="row${s.clips.length ? ' selected' : ''}" data-shot="${s.id}">
    <td class="mono">${s.id}</td>
    <td class="mono">${mmss(s.start)}</td>
    <td class="num mono">${(s.end - s.start).toFixed(1)} s</td>
    <td>${l ? `<span class="view-tag"><i style="background:${viewColor(l.view)}"></i>${esc(viewLabel(l.view))}</span>` : '<span class="faint">unlabelled</span>'}</td>
    <td>${l ? esc(l.rybakina_side) : ''}</td>
    <td>${l ? `<span class="q" title="${l.quality}/5">${[1, 2, 3, 4, 5].map((i) => `<i class="${i <= l.quality ? 'on' : ''}"></i>`).join('')}</span>` : ''}</td>
    <td><span class="flagset">${flags.map((x) => `<span class="mini">${x}</span>`).join('')}</span></td>
    <td class="events">${her.map((e) => `<span class="ev her" title="${mmss(e.t)}">${esc(e.type)} +${(e.t - s.start).toFixed(1)}s</span>`).join('')}</td>
    <td class="num">${opp || ''}</td>
    <td>${s.clips.map((id) => {
      const rej = isRejected(D.clipById[id]);
      return `<a class="mono${rej ? ' rejected' : ''}" href="#/moveset/${esc(id)}" title="${rej ? 'rejected by curate' : 'kept in the moveset'}">${esc(id)}</a>`;
    }).join('<br>')}</td>
  </tr>`;
}

function shotMatches(s) {
  const l = s.label;
  if (F.selected && !s.clips.length) return false;
  if (!l) return F.views.size === Object.keys(VIEW_META).length && !F.stroke && !F.her && !F.noReplay && !F.usable;
  if (!F.views.has(l.view)) return false;
  if (F.noReplay && l.replay) return false;
  if (F.usable && l.skip) return false;
  const her = l.events.filter((e) => e.hitter === 'rybakina');
  if (F.her && !her.length) return false;
  if (F.stroke && !(F.her ? her : l.events).some((e) => e.type === F.stroke)) return false;
  return true;
}

function applyFilters() {
  let shown = 0;
  const visible = new Set();
  for (const s of D.footage.shots) {
    const ok = shotMatches(s);
    if (ok) { shown += 1; visible.add(s.id); }
  }
  for (const row of $$('#ft-table tr.row')) {
    const id = Number(row.dataset.shot);
    const ok = visible.has(id);
    row.hidden = !ok;
    const det = row.nextElementSibling;
    if (det?.classList.contains('detail')) det.hidden = !ok;
  }
  for (const seg of $$('#ft-timeline .seg-shot')) seg.classList.toggle('dim', !visible.has(Number(seg.dataset.shot)));
  $('#ft-count').textContent = `${shown} of ${D.footage.shots.length} shots`;
}

function openShot(id, scroll = false) {
  const row = $(`#ft-table tr.row[data-shot="${id}"]`);
  if (!row) return;
  const s = D.shotById[id];
  const open = row.nextElementSibling?.classList.contains('detail');
  for (const seg of $$('#ft-timeline .seg-shot')) seg.classList.toggle('focus', Number(seg.dataset.shot) === id);
  if (open && !scroll) {
    row.nextElementSibling.remove();
    row.classList.remove('open');
    return;
  }
  if (!open) {
    row.classList.add('open');
    row.insertAdjacentHTML('afterend', `<tr class="detail"><td colspan="10">${shotDetail(s)}</td></tr>`);
  }
  if (row.hidden) {
    row.hidden = false;
    row.nextElementSibling.hidden = false;
  }
  if (scroll) row.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

function shotDetail(s) {
  const l = s.label;
  if (!l) return '<p class="faint" style="padding-top:10px">This shot has no Gemini label.</p>';
  const clips = s.clips.map((id) => D.clipById[id]).filter(Boolean);
  return `<div class="shot-detail">
    <div>${kv([
      ['Shot', `#${s.id} · frames ${num(s.start_frame)}–${num(s.end_frame)}`],
      ['Time', `${mmss(s.start)} – ${mmss(s.end)} (${(s.end - s.start).toFixed(2)} s)`],
      ['View', esc(viewLabel(l.view))],
      ['Her side', esc(l.rybakina_side)],
      ['Full body', l.full_body ? 'yes' : 'no'],
      ['Replay / slow-mo', `${l.replay ? 'replay' : 'live'} · ${l.slow_motion ? 'slow motion' : 'real time'}`],
      ['Quality', `${l.quality} / 5`],
      ['Outfit', esc(l.outfit || '–')],
      ['For capture', l.skip ? `<span class="badge pending">skipped</span> ${esc(l.skip)}` : '<span class="badge pass">usable</span>'],
    ])}
    ${clips.length ? `<div class="shot-clips">${clips.map((c) => `<a class="shot-clip" href="#/moveset/${esc(c.id)}">
      ${c.media.track ? `<img src="${esc(c.media.track)}" alt="" loading="lazy">` : ''}
      <span><b class="mono">${esc(c.id)}</b><br><span class="faint">${isRejected(c) ? 'rejected by curate · ' : (c.curation ? 'kept · ' : '')}open in moveset</span></span></a>`).join('')}</div>` : ''}
    </div>
    <div>${l.events.length ? `<table class="data">
      <thead><tr><th>Time</th><th>In shot</th><th>Event</th><th>Hitter</th><th class="num">Confidence</th></tr></thead>
      <tbody>${l.events.map((e) => `<tr>
        <td class="mono">${mmss(e.t)}</td><td class="mono faint">+${(e.t - s.start).toFixed(1)} s</td>
        <td><span class="ev ${e.hitter === 'rybakina' ? 'her' : ''}">${esc(e.type)}</span></td>
        <td>${esc(e.hitter)}</td><td class="num">${num(e.confidence, 2)}</td></tr>`).join('')}</tbody>
      </table>` : '<p class="faint">No strokes labelled in this shot.</p>'}</div>
  </div>`;
}

// ------------------------------------------------------------------------------------------------ notes
function renderNotes() {
  const n = D.notes || { limitations: [], next: [] };
  const qa = D.qa_summary?.overall;
  const byStroke = D.qa_summary?.by_stroke || {};
  $('#view-notes').innerHTML = `
    <div class="section-head"><div><h1>Notes</h1>
      <p>Generated from the run folder at ${fmtDate(D.generated)}; rebuild with <code>r3d site</code> after any stage.</p></div></div>
    <div class="notes">
      <div class="card card-pad"><h2 style="margin-bottom:14px">Limitations</h2>
        <ul>${n.limitations.map((t) => `<li>${esc(t)}</li>`).join('')}</ul></div>
      <div class="card card-pad next"><h2 style="margin-bottom:14px">Next steps</h2>
        <ul>${n.next.map((t) => `<li>${esc(t)}</li>`).join('')}</ul></div>
    </div>
    <div class="block"><h2>Stage status</h2>
      <div class="card table-wrap"><table class="data">
        <thead><tr><th>Stage</th><th>Status</th><th>Result</th><th>Updated</th></tr></thead>
        <tbody>${D.pipeline.map((s) => `<tr><td><b>${esc(s.name)}</b></td>
          <td><span class="view-tag status-${s.status}"><span class="status-dot"></span>${esc(s.status)}</span></td>
          <td class="muted">${esc(s.detail)}${breakdown(s)}</td><td class="faint">${s.updated ? fmtDate(s.updated) : '–'}</td></tr>`).join('')}</tbody>
      </table></div>
    </div>
    ${qa ? `<div class="block"><h2>QA summary <small>${qa.n} clips</small></h2>
      <div class="card table-wrap"><table class="data">
        <thead><tr><th>Stroke</th><th class="num">Pass</th><th class="num">Warn</th><th class="num">Fail</th>
          <th class="num">Skate cm/f</th><th class="num">Jitter m/s³</th><th class="num">Reproj px</th><th class="num">|Δt| f</th></tr></thead>
        <tbody>${[['all', qa], ...Object.entries(byStroke)].map(([s, r]) => `<tr><td><b>${esc(cap(s))}</b></td>
          <td class="num">${num(r.PASS)}</td><td class="num">${num(r.WARN)}</td><td class="num">${num(r.FAIL)}</td>
          <td class="num">${num(r.median_skate_cm, 2)}</td><td class="num">${num(r.median_jitter_ms3)}</td>
          <td class="num">${num(r.median_reproj_px, 1)}</td><td class="num">${num(r.median_abs_timing_frames, 1)}</td></tr>`).join('')}</tbody>
      </table></div></div>` : ''}`;
}

// ------------------------------------------------------------------------------------------------ artwork
// `r3d artwork compose`: the character sheet (painted / unpainted 3D renders) and the moveset atlas.
const ART_MODES = [['painted', 'Painted'], ['render', '3D render'], ['split', 'Side by side']];

function artFigure(img, caption, alt) {
  if (!img) return pendingBox(`${caption}: pending`);
  const dims = img.size ? ` · ${img.size[0]}×${img.size[1]} px` : '';
  return `<figure class="art-fig" data-full="${esc(img.url)}" data-caption="${esc(`${caption}${dims} · click the image to zoom`)}">
      <img src="${esc(img.preview)}" alt="${esc(alt)}" loading="lazy">
      <figcaption>${esc(caption)}<span>${esc(dims.replace(' · ', ''))}${isNum(img.size_mb) ? ` · ${num(img.size_mb, 1)} MB PNG` : ''}</span></figcaption>
    </figure>`;
}

function renderArtwork() {
  const A = D.artwork;
  const head = `<div class="section-head"><div>
      <h1>Artwork</h1>
      <p>Every figure on these sheets is the game-ready 3D character, posed by her captured motion: each pose is a
        frame of one of her real strokes, recovered from the broadcast footage and retargeted onto the rig, rendered
        in Blender and laid out like the client's character sheet. The painted sheet repaints those renders in the
        client's style (${esc(artModelName(A?.paint?.model || D.config?.models?.image_paint))}), and a repaint is only
        kept while its silhouette still matches the render, so the poses stay hers; the 3D render sheet shows the
        same figures unpainted. In motion, the same character plays every one of her captured clips.</p>
    </div>${artDownloads(A)}</div>`;
  if (!A || !(A.sheet || A.sheet_render || A.atlas)) {
    $('#view-artwork').innerHTML = `${head}${variantSections(D.variants)}${motionSection(A?.motion)}
      <div class="block"><h2>Character sheet</h2>${pendingBox('Artwork pending: run r3d artwork poses, r3d remote artwork render, r3d artwork paint, r3d artwork compose.')}</div>`;
    wireVariants();
    wireMotion(A?.motion);
    return;
  }
  const p = A.paint;
  const slots = Object.entries(p?.slots || {});
  const fell = slots.filter(([, s]) => !s.accepted).map(([n]) => n);
  const note = [
    `${A.painted.length} figure${A.painted.length === 1 ? '' : 's'} painted`,
    A.rendered.length ? `${A.rendered.length} kept as the render (${A.rendered.map(esc).join(', ')})` : '',
    A.silhouettes.length ? `${A.silhouettes.length} silhouettes cut from the render mattes` : '',
    A.missing.length ? `${A.missing.length} not rendered (${A.missing.map(esc).join(', ')})` : '',
  ].filter(Boolean).join(' · ');
  $('#view-artwork').innerHTML = `${head}${variantSections(D.variants)}${motionSection(A.motion)}
    <div class="block">
      <div class="art-head">
        <h2>Character sheet <small>${esc(note)}</small></h2>
        <div class="seg accent" id="art-mode" role="group" aria-label="Sheet version">${ART_MODES.map(([m, label]) => `
          <button type="button" data-mode="${m}" class="${m === 'painted' ? 'active' : ''}">${label}</button>`).join('')}</div>
      </div>
      <div class="art-sheets" id="art-sheets" data-mode="painted">
        <div class="art-painted">${artFigure(A.sheet, 'Character sheet, painted', 'Character sheet painted in the client style')}</div>
        <div class="art-render">${artFigure(A.sheet_render, 'Character sheet, unpainted 3D renders', 'Character sheet from the raw 3D renders')}</div>
      </div>
      ${slots.length ? `<div class="art-slots">${slots.map(([n, s]) => `<span class="chip${s.accepted ? '' : ' off'}"
        title="${esc(s.error || `${s.attempts} attempt${s.attempts === 1 ? '' : 's'}; silhouette IoU vs the render ${isNum(s.iou) ? num(s.iou, 3) : '–'}${s.accepted ? '' : ': kept the render'}`)}">
        <span class="sw" style="background:${s.accepted ? 'var(--pass)' : 'var(--warn)'}"></span>${esc(n)}
        <span class="faint">${s.error ? 'error' : isNum(s.iou) ? `IoU ${num(s.iou, 2)}` : ''}</span></span>`).join('')}</div>
        <p class="views-note">Paint check: the repaint's silhouette against the render's matte, aligned by bounding box; accepted at
          IoU ≥ ${num(p.accept_iou ?? 0.85, 2)}, else retried once with a stricter prompt (best of three samples), then the render is kept${fell.length ? ` (${fell.map(esc).join(', ')})` : ''}.</p>` : ''}
      ${artSources(A.sources)}
    </div>
    <div class="block">
      <h2>The moveset <small>${A.n_atlas} clips at their contact frame${A.n_flagged ? ` · ${A.n_flagged} marked: motion flagged by curate against its stroke label` : ''}</small></h2>
      ${artFigure(A.atlas, 'Moveset atlas', 'Every captured stroke at contact, grouped by stroke')}
    </div>`;
  $('#art-mode').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-mode]');
    if (!b) return;
    $$('#art-mode button').forEach((x) => x.classList.toggle('active', x === b));
    $('#art-sheets').dataset.mode = b.dataset.mode;
  });
  wireVariants();
  wireMotion(A.motion);
}

// "openai/gpt-image-2.5/sunburst/edit" -> "GPT Image 2.5 Sunburst on fal"
function artModelName(id) {
  const m = /gpt-image-([\d.]+)(?:\/(\w+))?/.exec(id || '');
  if (!m) return id || 'GPT Image';
  const variant = m[2] && m[2] !== 'edit' ? ` ${m[2][0].toUpperCase()}${m[2].slice(1)}` : '';
  return `GPT Image ${m[1]}${variant} on fal`;
}

function artSlotName(slot) {
  const m = /^(sil|action)_(\w+)$/.exec(slot);
  if (!m) return slot.replace('_', ' ');
  return m[1] === 'sil' ? `silhouette ${m[2]}` : `details ${m[2].toUpperCase()}`;
}

// where every posed figure comes from: its clip (linked to the moveset view) and frame relative to contact
function artSources(sources) {
  if (!sources?.length) return '';
  const moment = (s) => {
    const rel = isNum(s.frame) && isNum(s.contact_index) ? s.frame - s.contact_index : null;
    if (rel === null) return '';
    return rel === 0 ? 'at contact' : `${Math.abs(rel)} frame${Math.abs(rel) === 1 ? '' : 's'} ${rel < 0 ? 'before' : 'after'} contact`;
  };
  return `<div class="art-sources"><h3>Where each pose comes from <small>the turnaround, bust and heads use the rest pose</small></h3>
    <ul>${sources.map((s) => `<li><span class="slot">${esc(artSlotName(s.slot))}</span>
      <span><a class="mono" href="#/moveset/${esc(s.clip)}">${esc(s.clip)}</a><br><span class="faint">frame ${esc(s.frame)} · ${s.at === 'start' ? `ready position, ${esc(moment(s))}` : esc(moment(s))}${s.at === 'trophy' ? ' (trophy position)' : ''}</span></span></li>`).join('')}</ul></div>`;
}

function artDownloads(A) {
  const dl = A?.downloads || { glb: D.moveset?.url ? { url: D.moveset.url } : null, fbx: D.moveset?.fbx ? { url: D.moveset.fbx } : null };
  const n = D.moveset?.animations?.length;
  const btn = (f, label) => (f ? `<a class="btn small" href="${esc(f.url)}" download>${label}${isNum(f.size_mb) ? ` <span class="faint">${num(f.size_mb, 1)} MB</span>` : ''}</a>` : '');
  if (!dl.glb && !dl.fbx) return '';
  return `<div class="art-dl"><span class="faint">game-ready character + ${n || 39} animations</span>
    ${btn(dl.glb, 'GLB')}${btn(dl.fbx, 'FBX')}</div>`;
}

// ------------------------------------------------------------------------------------------------ artwork in motion
// `r3d artwork motion-render / motion-paint / motion-compose`: every clip as a video of the game-ready character,
// painted where the per-frame pose check held, and the moveset reel (all clips in a grid, contacts in sync).
// Only what is on screen plays: an IntersectionObserver starts and pauses every video (never 39 at once).
const MO_CMD = 'r3d remote artwork motion-render, then r3d artwork motion-paint and r3d artwork motion-compose';
let MO = null;                      // { M, mode, io, visible: Set<video> }

const moVideo = (c, mode) => (mode === 'painted' && c.painted ? c.painted : c.render);
const moIou = (p) => (isNum(p?.iou_median) ? ` · IoU ${num(p.iou_median, 2)}` : '');
const moTag = (c, mode) => (mode === 'painted' && c.painted ? `painted${moIou(c.painted)}` : '3D render');

function moTagTitle(c, mode) {
  if (mode === 'painted' && c.painted) {
    const p = c.painted;
    return `Painted by ${artVideoModelName(p.model)}; silhouette IoU against the render, frame by frame: median ${num(p.iou_median, 3)}, 10th percentile ${num(p.iou_p10, 3)}${isNum(p.iou_min) ? `, worst ${num(p.iou_min, 3)}` : ''}`;
  }
  if (c.paint && !c.paint.accepted) return `Painted version not kept${c.paint.reason ? `: ${c.paint.reason}` : ''}; the 3D render plays instead`;
  if (mode === 'painted') return 'No painted version yet: the 3D render plays instead';
  return 'The game-ready character rendered in Blender Cycles';
}

// "kling/v4/video-to-video/edit" -> "Kling v4 video-to-video edit"
function artVideoModelName(id) {
  if (!id) return 'a video model';
  const parts = id.replace(/^fal-ai\//, '').split('/');
  return [parts[0][0].toUpperCase() + parts[0].slice(1), ...parts.slice(1)].join(' ');
}

function moFlag(c) {
  if (!c.flagged) return '';
  const why = D.clipById[c.id]?.curation?.why;
  return `<span class="mo-flag" title="${esc(`Flagged by curate: the captured motion does not match the stroke label${why ? ` (${why})` : ''}; animated all the same`)}" aria-label="flagged by curate"></span>`;
}

function moTile(c, mode) {
  if (!c.render) {
    return `<figure class="mo-clip pending"><div class="mo-frame"><span>render pending</span></div>
      <figcaption><span class="mono">${esc(c.id)}</span>${moFlag(c)}</figcaption></figure>`;
  }
  const v = moVideo(c, mode);
  return `<figure class="mo-clip" data-mo-clip="${esc(c.id)}" tabindex="0" role="button" aria-label="${esc(`${c.id}: open the large video`)}">
      <div class="mo-frame"><video class="mo-vid" muted loop playsinline preload="none" disablepictureinpicture
        poster="${esc(v.poster || c.render.poster || '')}" src="${esc(v.tile || v.url)}"></video>
        <span class="mo-tag${mode === 'painted' && c.painted ? ' painted' : ''}" title="${esc(moTagTitle(c, mode))}">${esc(moTag(c, mode))}</span></div>
      <figcaption><span class="mono">${esc(c.id)}</span>${moFlag(c)}</figcaption>
    </figure>`;
}

function motionSection(M) {
  if (!M) {
    return `<div class="block" id="art-motion"><h2>In motion${motionWho()}</h2>
      ${pendingBox(`Motion pending: run ${MO_CMD}.`)}</div>`;
  }
  const mode = M.n_painted ? 'painted' : 'render';
  const reel = mode === 'painted' ? (M.reel || M.reel_render) : (M.reel_render || M.reel);
  const groups = [];
  for (const c of M.clips) {
    const g = groups.find((x) => x.stroke === c.stroke);
    if (g) g.clips.push(c); else groups.push({ stroke: c.stroke, clips: [c] });
  }
  const sub = [
    `${M.n_rendered} of ${M.n_clips} clips rendered`,
    `${M.n_painted} painted`,
    M.n_flagged ? `${M.n_flagged} flagged by curate` : '',
  ].filter(Boolean).join(' · ');
  const ar = reel?.size ? reel.size[0] / reel.size[1] : 0.9;
  const reelHtml = reel
    ? `<figure class="mo-reel" style="--ar:${ar.toFixed(4)}">
        <video id="mo-reel" class="mo-vid" muted loop playsinline preload="none" disablepictureinpicture
          poster="${esc(reel.poster || '')}" src="${esc(reel.url)}" style="aspect-ratio:${reel.size ? `${reel.size[0]} / ${reel.size[1]}` : 'auto'}"></video>
        <figcaption><span id="mo-reel-cap">Moveset reel, ${mode === 'painted' ? 'painted' : '3D render'}</span>
          <span>${reel.size ? `${reel.size[0]}×${reel.size[1]} · ` : ''}${isNum(reel.duration_s) ? `${num(reel.duration_s, 1)} s loop · ` : ''}<a id="mo-reel-full" href="${esc(reel.url)}" target="_blank" rel="noopener">full size ↗</a></span></figcaption>
      </figure>`
    : `<div class="mo-reel">${pendingBox('Moveset reel pending: run r3d artwork motion-compose.')}</div>`;
  return `<div class="block mo" id="art-motion" data-mode="${mode}">
    <div class="art-head">
      <h2>In motion${motionWho()} <small>${esc(sub)}</small></h2>
      <div class="seg accent" id="mo-mode" role="group" aria-label="Video version">
        <button type="button" data-mode="painted" class="${mode === 'painted' ? 'active' : ''}"${M.n_painted ? '' : ' disabled title="No painted clips yet"'}>Painted</button>
        <button type="button" data-mode="render" class="${mode === 'render' ? 'active' : ''}">3D render</button>
      </div>
    </div>
    <p class="block-lede">Every clip of the moveset played by the game-ready character, frame for frame as she moved
      in the broadcast. In the reel all of them hit the ball on the same frame: each clip holds its first frame until
      its swing has to start and its last frame once it is done. A painted clip is kept only while its silhouette
      still matches the render's frame by frame, so the motion stays hers; the others play as the 3D render.</p>
    ${M.reel_stale ? `<p class="mo-stale">The reel predates the latest renders or repaints: run <code>r3d artwork motion-compose</code>.</p>`
      : M.in_reel?.missing?.length ? `<p class="mo-stale">The reel holds ${M.n_clips - M.in_reel.missing.length} of ${M.n_clips} clips: the others are not rendered yet.</p>` : ''}
    <div class="mo-top">${reelHtml}</div>
    <div class="mo-clips">
      <h3>Every clip <small>click one for the large video and its review in the Moveset tab</small></h3>
      ${groups.map((g) => `<div class="mo-group">
        <div class="mo-group-head"><b>${esc(cap(g.stroke))}</b><span class="faint">${g.clips.length} clip${g.clips.length === 1 ? '' : 's'}</span></div>
        <div class="mo-grid">${g.clips.map((c) => moTile(c, mode)).join('')}</div></div>`).join('')}
      ${M.n_flagged ? `<p class="views-note"><span class="mo-flag"></span> flagged by curate: the captured motion does not match the clip's stroke label; animated and kept all the same.</p>` : ''}
    </div>
  </div>`;
}

function wireMotion(M) {
  MO?.io?.disconnect();
  MO = null;
  if (!M) return;
  MO = { M, mode: $('#art-motion')?.dataset.mode || 'render', visible: new Set(), held: false };
  const play = (v) => { const p = v.play(); if (p) p.catch(() => {}); };
  MO.play = play;
  MO.io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      const v = e.target;
      if (e.isIntersecting) {
        MO.visible.add(v);
        if (!MO.held) play(v);
      } else {
        MO.visible.delete(v);
        v.pause();
      }
    }
  }, { threshold: 0.15 });
  for (const v of $$('#art-motion video.mo-vid')) MO.io.observe(v);

  $('#mo-mode')?.addEventListener('click', (e) => {
    const b = e.target.closest('button[data-mode]');
    if (!b || b.disabled || b.dataset.mode === MO.mode) return;
    $$('#mo-mode button').forEach((x) => x.classList.toggle('active', x === b));
    setMotionMode(b.dataset.mode);
  });
  const open = (el) => {
    const c = M.clips.find((x) => x.id === el.dataset.moClip);
    if (c) openMotionClip(c);
  };
  $('#art-motion').addEventListener('click', (e) => {
    const el = e.target.closest('[data-mo-clip]');
    if (el) open(el);
  });
  $('#art-motion').addEventListener('keydown', (e) => {
    const el = e.target.closest('[data-mo-clip]');
    if (el && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); open(el); }
  });
}

/** Leaving the artwork view stops every motion video; coming back re-observes them, so what is on screen plays. */
function motionActive(on) {
  if (!MO) return;
  MO.io.disconnect();
  MO.visible.clear();
  for (const v of $$('#art-motion video.mo-vid')) {
    v.pause();
    if (on) MO.io.observe(v);
  }
  // the reel autoplays on the artwork view only: another view's page load must not fetch it
  const reel = $('#mo-reel');
  if (reel) reel.autoplay = on;
}

/** Swap a video's source, keeping its place in the loop and its play state. */
function moSwap(v, src, poster) {
  if (!v || v.getAttribute('src') === src) return;
  const t = v.currentTime;
  if (poster) v.setAttribute('poster', poster);
  v.setAttribute('src', src);
  if (t > 0) v.addEventListener('loadedmetadata', () => { v.currentTime = Math.min(t, v.duration || t); }, { once: true });
  // a new source re-arms `autoplay` (the reel): only what is on screen may start
  if (MO.visible.has(v) && !MO.held) MO.play(v); else v.pause();
}

function setMotionMode(mode) {
  const M = MO.M;
  MO.mode = mode;
  $('#art-motion').dataset.mode = mode;
  const reel = mode === 'painted' ? (M.reel || M.reel_render) : (M.reel_render || M.reel);
  if (reel) {
    moSwap($('#mo-reel'), reel.url, reel.poster);
    $('#mo-reel-cap').textContent = `Moveset reel, ${mode === 'painted' ? 'painted' : '3D render'}`;
    $('#mo-reel-full').href = reel.url;
  }
  for (const fig of $$('#art-motion [data-mo-clip]')) {
    const c = M.clips.find((x) => x.id === fig.dataset.moClip);
    const v = moVideo(c, mode);
    moSwap($('video', fig), v.tile || v.url, v.poster || c.render.poster);
    const tag = $('.mo-tag', fig);
    tag.textContent = moTag(c, mode);
    tag.title = moTagTitle(c, mode);
    tag.classList.toggle('painted', mode === 'painted' && !!c.painted);
  }
}

function openMotionClip(c) {
  const mode = MO.mode;
  const v = moVideo(c, mode);
  const what = mode === 'painted' && c.painted
    ? `painted by ${artVideoModelName(c.painted.model)}, silhouette IoU median ${num(c.painted.iou_median, 2)}, p10 ${num(c.painted.iou_p10, 2)}`
    : `3D render${mode !== 'painted' ? '' : c.paint ? ' (painted version not kept)' : ' (not painted yet)'}`;
  const frames = isNum(c.render.n_frames) ? ` · ${c.render.n_frames} frames${isNum(c.render.contact_frame) ? `, contact at frame ${c.render.contact_frame}` : ''}` : '';
  openLightboxVideo({
    src: v.url, poster: v === c.painted ? v.poster : (v.poster_full || v.poster),
    caption: `${c.id} · ${cap(c.stroke)}${c.flagged ? ' (flagged by curate)' : ''} · ${what}${frames}`,
    link: D.clipById[c.id] ? { href: `#/moveset/${encodeURIComponent(c.id)}`, text: `Review ${c.id} in the Moveset tab →` } : null,
  });
}

// ------------------------------------------------------------------------------------------------ character variants
// Another character on the same captured motion (variants/<name>/ in the run folder, D.variants), rendered for
// Meta Quest 3 (render profile quest3: 2064×2208 per eye, 60 fps real time) or, until that is composed, its
// quarter-size draft. The reel plays while it is on screen. Every clip is a light tile that plays on hover (a tap on
// touch screens), one at a time, and links to its full-resolution video, which the grid itself never loads.
const VR_LIVE = {
  quest3: ['Quest 3 renders', 'var(--pass)', 'Rendered at the Quest 3 per-eye resolution, 60 fps'],
  draft: ['Draft preview', 'var(--warn)', 'The quarter-size draft of the Quest 3 framing and frame rate, shown until the Quest 3 renders are composed'],
};
let VR = null;                      // { io, visible: Set<reel>, tile: the tile playing, held, pointer }

const playVideo = (v) => { const p = v.play(); if (p) p.catch(() => {}); };
const dimsOf = (s) => (Array.isArray(s) && s.length === 2 ? `${s[0]}×${s[1]}` : '');
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const vrCmd = (V, cmd, profile) => `${V.config ? `R3D_CONFIG=${V.config} ` : ''}r3d ${cmd} --profile ${profile}`;
/** With a variant above it, the In motion block says whose character it shows. */
const motionWho = () => (D.variants?.length && D.athlete?.name ? ` — ${esc(D.athlete.name)}` : '');

function variantSections(list) {
  return (list || []).map(variantSection).join('') + comparisonSection(D.comparisons);
}

/** Side-by-side videos (<run>/compare/): one clip, two characters or settings, same everything else. */
function comparisonSection(list) {
  if (!list?.length) return '';
  return `<div class="block mo vr" id="compare">
    <div class="art-head"><h2>Model comparison <small>${plural(list.length, 'video')}</small></h2></div>
    <div class="cmp-grid">${list.map((c) => {
      const v = c.video || {};
      const ar = v.width && v.height ? (v.width / v.height).toFixed(4) : '0.81';
      return `<figure class="mo-reel cmp" style="--ar:${ar}">
        <video controls muted loop playsinline preload="metadata" poster="${esc(c.poster || '')}" src="${esc(v.url || '')}"
          ${v.width && v.height ? `style="aspect-ratio:${v.width} / ${v.height}"` : ''}></video>
        <figcaption><span>${esc(c.title)}</span><span>${v.width && v.height ? `${dimsOf([v.width, v.height])} · ` : ''}<a href="${esc(v.url || '')}" target="_blank" rel="noopener">full size ↗</a></span></figcaption>
        ${c.note ? `<p class="block-lede">${esc(c.note)}</p>` : ''}
      </figure>`;
    }).join('')}</div>
  </div>`;
}

function variantSection(V) {
  const live = V.profile;
  const size = V.reel?.size || V.size;
  const fps = V.reel?.fps || V.fps;
  const tile = V.clips.find((c) => c.size)?.size || V.size;
  const compose = (p) => vrCmd(V, 'artwork motion-compose', p);
  const sub = [
    live ? `${V.n_rendered} of ${V.n_clips} clips rendered` : 'not rendered yet',
    live === 'draft' && V.n_hires ? `${V.n_hires} at full resolution` : '',
    dimsOf(V.size) ? `${dimsOf(V.size)}${live === 'quest3' ? ' per eye' : ''}` : '',
    isNum(V.fps) ? `${num(V.fps)} fps` : '',
    V.n_flagged ? `${V.n_flagged} flagged by curate` : '',
  ].filter(Boolean).join(' · ');
  const [liveLabel, liveColor, liveTitle] = VR_LIVE[live] || ['Pending', 'var(--text-3)', 'Nothing rendered at the Quest 3 profile yet'];
  const dl = (f, label) => (f ? `<a class="btn small" href="${esc(f.url)}" download>${label}${isNum(f.size_mb) ? ` <span class="faint">${num(f.size_mb, 1)} MB</span>` : ''}</a>` : '');
  const anims = V.downloads?.animations;
  const downloads = V.downloads?.glb || V.downloads?.fbx
    ? `<span class="faint">character + ${anims ? plural(anims, 'animation') : 'moveset'}</span>${dl(V.downloads.glb, 'GLB')}${dl(V.downloads.fbx, 'FBX')}` : '';
  const q3 = V.progress?.quest3 || {};
  const missingTiles = V.n_rendered - V.n_tiles;
  const notes = [
    live === 'draft' && (q3.rendered || V.n_hires) ? ['vr-note', `Quest 3 renders in progress: ${Math.max(q3.rendered || 0, V.n_hires || 0)} of ${V.n_clips} clips at 2064×2208. Clips marked <b>2064×2208 ↗</b> already open their full-resolution video; the reel stays the draft until every clip is rendered.`] : null,
    live === 'quest3' && isNum(V.n_in_reel) && V.n_in_reel < V.n_clips
      ? ['vr-note', `The reel is composed once all ${V.n_clips} clips are rendered (${V.n_rendered} so far${V.n_outdated ? `; ${V.n_outdated} from the previous rig are being re-rendered` : ''}); every rendered clip plays in the grid below.`] : null,
    V.reel_stale && !(live === 'quest3' && isNum(V.n_in_reel) && V.n_in_reel < V.n_clips) ? ['mo-stale', `The reel predates the latest renders${isNum(V.n_in_reel) ? ` (it holds ${V.n_in_reel} of the ${V.n_rendered} rendered clips)` : ''}: run <code>${esc(compose(live))}</code>.`] : null,
    live && !V.composed ? ['vr-note', `Not composed yet: the grid shows every rendered clip's poster; <code>${esc(compose(live))}</code> makes the reel and the playable tiles.`] : null,
    V.composed && missingTiles > 0 ? ['vr-note', `${plural(missingTiles, 'rendered clip')} ${missingTiles === 1 ? 'has' : 'have'} no light tile yet and play${missingTiles === 1 ? 's its' : ' their'} full-resolution video until the next compose.`] : null,
    V.n_tiles_stale ? ['vr-note', `${plural(V.n_tiles_stale, 'tile')} predate${V.n_tiles_stale === 1 ? 's' : ''} a re-render of ${V.n_tiles_stale === 1 ? 'its clip' : 'their clips'}: ${V.n_tiles_stale === 1 ? 'that clip plays its' : 'those clips play their'} new full-resolution video instead.`] : null,
  ].filter(Boolean);
  // a quest3 reel laid out for the whole moveset but holding only part of it is mostly empty paper: until it holds
  // every clip, the grid below is the page and the reel waits
  const partial = live === 'quest3' && isNum(V.n_in_reel) && V.n_in_reel < V.n_clips;
  const reel = partial ? null : V.reel;
  const reelHtml = partial ? '' : reel?.url
    ? `<figure class="mo-reel" style="--ar:${(size ? size[0] / size[1] : 1).toFixed(4)}">
        <video class="vr-reel" muted loop playsinline preload="none" disablepictureinpicture
          poster="${esc(reel.poster || '')}" src="${esc(reel.url)}"${size ? ` style="aspect-ratio:${size[0]} / ${size[1]}"` : ''}></video>
        <figcaption><span>Moveset reel, ${esc(V.name)}</span>
          <span>${[dimsOf(size), isNum(fps) ? `${num(fps)} fps` : '', isNum(reel.loop_s) ? `${num(reel.loop_s, 1)} s loop` : ''].filter(Boolean).map((x) => `${x} · `).join('')}<a href="${esc(reel.url)}" download>download${isNum(reel.size_mb) ? ` ${num(reel.size_mb, 1)} MB` : ''}</a> · <a href="${esc(reel.url)}" target="_blank" rel="noopener">full size ↗</a></span></figcaption>
      </figure>`
    : `<div class="mo-reel">${pendingBox(live
      ? `Reel pending: run ${compose(live)}.`
      : `Renders pending: run ${vrCmd(V, 'remote artwork motion-render', 'quest3')} (--profile draft for a quick preview), then ${compose('quest3')}.`)}</div>`;
  const byId = Object.fromEntries(V.clips.map((c) => [c.id, c]));
  const groups = V.groups.map((g) => ({ stroke: g.stroke, clips: g.clips.map((id) => byId[id]).filter(Boolean) }))
    .filter((g) => g.clips.length);
  const grid = V.clips.some((c) => c.mp4 || c.tile || c.poster)
    ? `<div class="mo-clips">
      <h3>Every clip <small>hover a clip to play it (tap on a touch screen); click it or “full ↗” for the full-resolution video</small></h3>
      ${groups.map((g) => `<div class="mo-group">
        <div class="mo-group-head"><b>${esc(cap(g.stroke))}</b><span class="faint">${plural(g.clips.length, 'clip')}</span></div>
        <div class="mo-grid">${g.clips.map((c) => vrTile(V, c)).join('')}</div></div>`).join('')}
      ${V.n_flagged ? `<p class="views-note"><span class="mo-flag"></span> flagged by curate: the captured motion does not match the clip's stroke label; animated and kept all the same.</p>` : ''}
    </div>` : '';
  return `<div class="block mo vr" id="vr-${esc(V.variant)}" data-variant="${esc(V.variant)}" style="--tile-ar:${(tile ? tile[0] / tile[1] : 1).toFixed(4)}">
    <div class="art-head">
      <h2>${esc(V.name)} — Quest 3 <small>${esc(sub)}</small></h2>
      <div class="art-dl"><span class="chip" title="${esc(liveTitle)}"><span class="sw" style="background:${liveColor}"></span>${esc(liveLabel)}</span>${downloads}</div>
    </div>
    <p class="block-lede">${esc(V.name)} playing ${esc(V.motion_from || 'the captured')}'s moveset: every captured clip retargeted onto
      the ${esc(V.name)} character and rendered in Blender Cycles for Meta Quest 3, at its per-eye resolution of 2064×2208 and
      60 frames per second (the motion is keyed at the broadcast's 25 fps and sampled in between). Render only, no painting.
      In the reel every clip hits the ball on the same frame.${live === 'draft' ? ' What plays here for now is the quarter-size draft of the same framing and frame rate.' : ''}</p>
    ${notes.map(([cls, html]) => `<p class="${cls}">${html}</p>`).join('')}
    <div class="mo-top">${reelHtml}</div>
    ${grid}
  </div>`;
}

function vrTile(V, c) {
  const fps = c.fps || V.fps;
  const res = [dimsOf(c.size || V.size), isNum(fps) ? `${num(fps)} fps` : '', isNum(c.mb) ? `${num(c.mb, 1)} MB` : ''].filter(Boolean).join(', ');
  // a clip whose Quest 3 video exists while the draft is live says so; the others say what they open
  const hi = c.profile === 'quest3' && V.profile !== 'quest3';
  const label = hi ? `${dimsOf(c.size) || 'full'} ↗` : V.profile === 'draft' ? 'draft ↗' : 'full ↗';
  const full = c.mp4 ? `<a class="vr-full${hi ? ' hi' : ''}" href="${esc(c.mp4)}" target="_blank" rel="noopener" title="${esc(`${hi || V.profile !== 'draft' ? 'Full resolution' : 'Draft'}: ${res}`)}">${esc(label)}</a>` : '';
  const caption = `<figcaption><span class="mono">${esc(c.id)}</span>${moFlag(c)}${full}</figcaption>`;
  if (!c.tile && !c.poster && !c.tile_poster) {
    return `<figure class="mo-clip vr-clip pending"><div class="mo-frame"><span>${c.mp4 ? 'poster pending' : c.outdated ? 're-render pending' : 'render pending'}</span></div>${caption}</figure>`;
  }
  // the light tile while it shows the current render; otherwise (no tile yet, or one from an earlier render) the
  // full-resolution video itself plays in the grid
  const src = c.tile && !c.tile_stale ? c.tile : c.mp4 || c.tile;
  const still = c.tile && !c.tile_stale ? c.tile_poster || c.poster : c.poster || c.tile_poster;
  const media = src
    ? `<video class="vr-tile" muted loop playsinline preload="none" disablepictureinpicture poster="${esc(still || '')}" src="${esc(src)}"></video>
       <span class="vr-play" aria-hidden="true">${ICON.play}</span>`
    : `<img src="${esc(still)}" alt="" loading="lazy"><span class="mo-tag">poster · tile pending</span>`;
  return `<figure class="mo-clip vr-clip" data-vr-clip="${esc(c.id)}" tabindex="0" role="button" aria-label="${esc(`${c.id}: open the full-resolution video`)}">
      <div class="mo-frame">${media}</div>${caption}</figure>`;
}

function wireVariants() {
  VR?.io?.disconnect();
  VR = null;
  const secs = $$('#view-artwork .vr[data-variant]');
  if (!secs.length) return;
  VR = { visible: new Set(), tile: null, held: false, pointer: 'mouse' };
  VR.io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        VR.visible.add(e.target);
        if (!VR.held) playVideo(e.target);
      } else {
        VR.visible.delete(e.target);
        e.target.pause();
      }
    }
  }, { threshold: 0.15 });
  for (const v of $$('#view-artwork video.vr-reel')) VR.io.observe(v);
  for (const sec of secs) {
    const V = D.variants.find((x) => x.variant === sec.dataset.variant);
    const clipOf = (fig) => V?.clips.find((c) => c.id === fig.dataset.vrClip);
    sec.addEventListener('pointerdown', (e) => { VR.pointer = e.pointerType; });
    sec.addEventListener('pointerover', (e) => {
      const fig = e.pointerType === 'mouse' && e.target.closest('[data-vr-clip]');
      const v = fig && $('video.vr-tile', fig);
      if (v && v !== VR.tile) vrPlay(v);
    });
    sec.addEventListener('pointerout', (e) => {
      const fig = e.target.closest('[data-vr-clip]');
      if (e.pointerType === 'mouse' && fig && !fig.contains(e.relatedTarget) && VR.tile && fig.contains(VR.tile)) vrStop();
    });
    sec.addEventListener('click', (e) => {
      const fig = e.target.closest('[data-vr-clip]');
      if (!fig || e.target.closest('a')) return;      // "full ↗" opens the full-resolution video itself
      const v = $('video.vr-tile', fig);
      // touch has no hover: a tap plays or pauses the tile; a mouse click opens the full-resolution video
      if (VR.pointer !== 'mouse' && v) {
        if (v === VR.tile && !v.paused) vrStop(); else vrPlay(v);
        return;
      }
      const c = clipOf(fig);
      if (c) openVariantClip(V, c);
    });
    sec.addEventListener('keydown', (e) => {
      const fig = e.target.closest('[data-vr-clip]');
      // Enter on the focused "full ↗" link follows the link, as a click on it does
      if (!fig || e.target.closest('a') || (e.key !== 'Enter' && e.key !== ' ')) return;
      e.preventDefault();
      const c = clipOf(fig);
      if (c) openVariantClip(V, c);
    });
  }
}

/** One tile plays at a time: a 2064×2208 grid's light copies are still dozens of decoders. */
function vrPlay(v) {
  if (!VR || VR.held) return;
  if (VR.tile && VR.tile !== v) vrStop();
  VR.tile = v;
  v.closest('.vr-clip')?.classList.add('playing');
  playVideo(v);
}

function vrStop() {
  const v = VR?.tile;
  if (!v) return;
  v.pause();
  v.closest('.vr-clip')?.classList.remove('playing');
  VR.tile = null;
}

/** Like motionActive: the reels autoplay (on screen) on the artwork view only; leaving it stops every video. */
function variantActive(on) {
  if (!VR) return;
  VR.io.disconnect();
  VR.visible.clear();
  vrStop();
  for (const v of $$('#view-artwork video.vr-reel')) {
    v.pause();
    v.autoplay = on;
    if (on) VR.io.observe(v);
  }
}

function openVariantClip(V, c) {
  const src = c.mp4 || c.tile;
  if (!src) return;
  const fps = c.fps || V.fps;
  const frames = isNum(c.frames) ? ` · ${c.frames} frames${isNum(c.contact_frame) ? `, contact at frame ${c.contact_frame}` : ''}` : '';
  openLightboxVideo({
    src, poster: c.poster || c.tile_poster,
    caption: `${c.id} · ${cap(c.stroke)}${c.flagged ? ' (flagged by curate)' : ''} · ${V.name} · ${dimsOf(c.size || V.size)}${isNum(fps) ? ` · ${num(fps)} fps` : ''}${frames}${c.mp4 ? '' : ' · light tile, the full video is pending'}`,
    link: D.clipById[c.id] ? { href: `#/moveset/${encodeURIComponent(c.id)}`, text: `Review ${c.id} in the Moveset tab →` } : null,
  });
}

// ------------------------------------------------------------------------------------------------ overlays
function openLightboxVideo({ src, poster, caption, link }) {
  const lb = $('#lightbox');
  const video = $('video', lb);
  const a = $('.lightbox-link', lb);
  lb.classList.remove('zoomed');
  $('img', lb).hidden = true;
  video.hidden = false;
  if (poster) video.setAttribute('poster', poster); else video.removeAttribute('poster');
  video.src = src;
  const p = video.play();
  if (p) p.catch(() => {});
  $('figcaption', lb).textContent = caption || '';
  a.hidden = !link;
  if (link) { a.href = link.href; a.textContent = link.text; }
  lb.hidden = false;
  if (MO) { MO.held = true; MO.visible.forEach((v) => v.pause()); }
  if (VR) { VR.held = true; VR.visible.forEach((v) => v.pause()); vrStop(); }
}

function setupLightbox() {
  const lb = $('#lightbox');
  const img = $('img', lb);
  const capEl = $('figcaption', lb);
  const video = $('video', lb);
  const link = $('.lightbox-link', lb);
  const close = () => {
    lb.hidden = true;
    lb.classList.remove('zoomed');
    img.removeAttribute('src');
    video.pause();
    video.removeAttribute('src');
    video.load();
    if (MO?.held) { MO.held = false; MO.visible.forEach((v) => MO.play(v)); }
    if (VR?.held) { VR.held = false; VR.visible.forEach(playVideo); }
  };
  document.addEventListener('click', (e) => {
    const t = e.target.closest('[data-full]');
    if (t && !e.target.closest('a')) {
      lb.classList.remove('zoomed');
      img.hidden = false;
      video.hidden = true;
      link.hidden = true;
      img.src = t.dataset.full;
      capEl.textContent = t.dataset.caption || '';
      lb.hidden = false;
      return;
    }
    if (!lb.hidden && (e.target === lb || e.target.closest('.lightbox-close') || e.target === link)) close();
  });
  // big images (the artwork sheets): click toggles 1:1, centred on the clicked point; scroll to pan
  img.addEventListener('click', (e) => {
    const r = img.getBoundingClientRect();
    const zoom = !lb.classList.contains('zoomed');
    if (zoom && img.naturalWidth <= r.width * 1.05) return;
    const fx = (e.clientX - r.left) / r.width;
    const fy = (e.clientY - r.top) / r.height;
    lb.classList.toggle('zoomed', zoom);
    if (zoom) {
      requestAnimationFrame(() => {
        lb.scrollLeft = fx * img.offsetWidth - lb.clientWidth / 2;
        lb.scrollTop = fy * img.offsetHeight - lb.clientHeight / 2;
      });
    }
  });
  img.addEventListener('load', () => img.classList.toggle('zoomable', img.naturalWidth > img.clientWidth * 1.05));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !lb.hidden) close(); });
}

function setupTooltip() {
  document.addEventListener('mouseover', (e) => {
    const t = e.target.closest('[data-tip]');
    if (t) showTip(e, t.dataset.tip);
  });
  document.addEventListener('mouseout', (e) => { if (e.target.closest('[data-tip]')) hideTip(); });
}

function showTip(e, html) {
  const tip = $('#tooltip');
  tip.innerHTML = html;
  tip.hidden = false;
  const w = tip.offsetWidth;
  tip.style.left = `${clamp(e.clientX + 12, 8, window.innerWidth - w - 8)}px`;
  tip.style.top = `${e.clientY + 16}px`;
}

function hideTip() { $('#tooltip').hidden = true; }
