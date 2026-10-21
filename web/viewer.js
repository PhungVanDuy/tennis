// 3D: a lit stage with orbit controls, the character (GLB, or Tripo's gzip-encoded FBX), and the GVHMR
// joints drawn as a skeleton. Used by the character viewer and by the moveset's synced 3D pane.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';

export { THREE };

const UP = new THREE.Vector3(0, 1, 0);
const KEY_OFFSET = new THREE.Vector3(3.5, 7.5, 4.5);
const GROUND_R = 16;
// Same colours as the GVHMR overlay video: the left chain amber, the right chain and spine blue.
const LEFT_JOINTS = new Set([1, 4, 7, 10, 13, 16, 18, 20, 22]);
const AMBER = new THREE.Color('#ffc850');
const BLUE = new THREE.Color('#5078ff');

// ---------------------------------------------------------------------------------------------- assets
const cache = new Map();

/** Load (once) and cache a model: { root, animations }. kind: 'glb' | 'fbx'. */
export function loadAsset(url, kind = 'glb', onProgress) {
  if (!cache.has(url)) {
    const p = kind === 'fbx' ? loadFbx(url, onProgress) : loadGlb(url, onProgress);
    p.catch(() => cache.delete(url));
    cache.set(url, p);
  }
  return cache.get(url);
}

async function loadGlb(url, onProgress) {
  const gltf = await new GLTFLoader().loadAsync(url, (e) => onProgress?.(e.loaded, e.total));
  return { root: gltf.scene, animations: gltf.animations || [] };
}

async function loadFbx(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  let buf = await res.arrayBuffer();
  onProgress?.(buf.byteLength, buf.byteLength);
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] === 0x1f && head[1] === 0x8b) {
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
    buf = await new Response(stream).arrayBuffer();
  }
  const root = new FBXLoader().parse(buf, url.slice(0, url.lastIndexOf('/') + 1));
  return { root, animations: root.animations || [] };
}

/** An independent copy of a cached asset (skinned meshes need their own skeleton per scene). */
export function instantiate(asset) {
  const obj = SkeletonUtils.clone(asset.root);
  obj.traverse((o) => {
    if (o.isMesh) {
      o.castShadow = true;
      o.receiveShadow = true;
      o.frustumCulled = false; // bounds are bind-pose only; root motion would cull the body
    }
  });
  return obj;
}

/** Triangles / vertices / bones counted on the loaded scene (for files without a server-side report). */
export function measure(obj) {
  let triangles = 0;
  let vertices = 0;
  let meshes = 0;
  const bones = new Set();
  obj.traverse((o) => {
    if (o.isMesh && o.geometry?.attributes?.position) {
      const g = o.geometry;
      meshes += 1;
      vertices += g.attributes.position.count;
      triangles += (g.index ? g.index.count : g.attributes.position.count) / 3;
    }
    if (o.isSkinnedMesh) o.skeleton.bones.forEach((b) => bones.add(b.uuid));
  });
  return { triangles: Math.round(triangles), vertices, joints: bones.size, meshes };
}

/** Rescale models authored in other units (centimetres, unit-box exports) to the character height. */
export function fitUnits(obj, height = 1.84) {
  obj.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(obj, true);
  const h = box.max.y - box.min.y;
  if (h > 0 && (h > height * 1.3 || h < height * 0.75)) {
    obj.scale.multiplyScalar(height / h);
    obj.updateMatrixWorld(true);
    return height;
  }
  return h;
}

/** Centre on the origin, feet on the ground (static display only: animated clips keep their root path). */
export function placeOnGround(obj) {
  obj.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(obj, true);
  const c = box.getCenter(new THREE.Vector3());
  obj.position.x -= c.x;
  obj.position.z -= c.z;
  obj.position.y -= box.min.y;
  obj.updateMatrixWorld(true);
  return box.max.y - box.min.y;
}

/** Hips and the two thigh roots of a humanoid rig (Meshy, Mixamo, SMPL-style names). */
export function findRig(obj) {
  const bones = [];
  obj.traverse((o) => { if (o.isBone) bones.push(o); });
  const pick = (...res) => {
    for (const re of res) {
      const b = bones.find((x) => re.test(x.name));
      if (b) return b;
    }
    return null;
  };
  return {
    bones,
    hips: pick(/^(mixamorig\d*[:_]?)?hips$/i, /pelvis/i, /hips/i, /root/i) || bones[0] || null,
    left: pick(/left.?up.?leg/i, /left.?thigh/i, /left.?hip/i, /thigh.?l$/i, /upperleg.?l$/i),
    right: pick(/right.?up.?leg/i, /right.?thigh/i, /right.?hip/i, /thigh.?r$/i, /upperleg.?r$/i),
  };
}

/** Body frame from pelvis + hip positions: right, up, forward (a body facing +Z has its left hip at +X). */
export function bodyFrame(origin, left, right) {
  const lateral = new THREE.Vector3().subVectors(left, right);
  lateral.y = 0;
  if (lateral.lengthSq() < 1e-10) lateral.set(1, 0, 0);
  lateral.normalize();
  return {
    origin: origin.clone(),
    right: lateral.clone().negate(),
    up: UP.clone(),
    forward: new THREE.Vector3().crossVectors(lateral, UP).normalize(),
  };
}

export function fromBody(frame, [x, y, z]) {
  return new THREE.Vector3()
    .addScaledVector(frame.right, x)
    .addScaledVector(frame.up, y)
    .addScaledVector(frame.forward, z);
}

// ---------------------------------------------------------------------------------------------- stage
export class Stage {
  constructor(el, { background = 0x0f1114 } = {}) {
    this.el = el;
    const r = (this.renderer = new THREE.WebGLRenderer({ antialias: true }));
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.05;
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    el.prepend(r.domElement);

    const scene = (this.scene = new THREE.Scene());
    scene.background = new THREE.Color(background);
    const pmrem = new THREE.PMREMGenerator(r);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environmentIntensity = 0.55;

    scene.add(new THREE.HemisphereLight(0xdfe7ff, 0x24272c, 0.85));
    const key = (this.key = new THREE.DirectionalLight(0xffffff, 2.3));
    key.position.copy(KEY_OFFSET);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    Object.assign(key.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.5, far: 30 });
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.02;
    scene.add(key, key.target);
    const rim = new THREE.DirectionalLight(0xc4d6ff, 1.1);
    rim.position.set(-4, 3.5, -5);
    scene.add(rim);

    // a floor that fades into the background instead of ending on a hard horizon
    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(GROUND_R, 72),
      new THREE.MeshStandardMaterial({
        color: 0x1b1f24, roughness: 0.95, metalness: 0, transparent: true, depthWrite: false,
        alphaMap: radialTexture(),
      }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);
    this.grid = fadingGrid(GROUND_R * 0.8);
    scene.add(this.grid);

    const cam = (this.camera = new THREE.PerspectiveCamera(35, 1, 0.05, 300));
    cam.position.set(2.6, 1.7, 4.6);
    const ctl = (this.controls = new OrbitControls(cam, r.domElement));
    ctl.enableDamping = true;
    ctl.dampingFactor = 0.09;
    ctl.target.set(0, 1, 0);
    ctl.maxPolarAngle = Math.PI * 0.495;
    ctl.minDistance = 0.6;
    ctl.maxDistance = 60;
    ctl.autoRotateSpeed = 1.2;
    ctl.update();

    this.hooks = [];
    this.lineMaterials = new Set();
    this.follow = null;
    this._fy = null;
    this.tween = null;
    this.active = false;
    this.clock = new THREE.Clock();
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(el);
    this.resize();
  }

  resize() {
    const w = this.el.clientWidth;
    const h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    for (const m of this.lineMaterials) m.resolution.set(w, h);
  }

  setActive(on) {
    if (on === this.active) return;
    this.active = on;
    if (on) {
      this.clock.getDelta();
      this.resize();
    }
    this.renderer.setAnimationLoop(on ? () => this.frame() : null);
  }

  onFrame(fn) { this.hooks.push(fn); }

  frame() {
    const dt = Math.min(this.clock.getDelta(), 0.1);
    for (const fn of this.hooks) fn(dt);
    this._follow(dt);
    this._tween(dt);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  /** Keep the orbit target on a moving point (camera translated with it, orbit offset kept). */
  setFollow(fn) {
    this.follow = fn;
    this._fy = null;
  }

  _follow(dt) {
    const p = this.follow?.();
    if (!p) return;
    const tgt = this.controls.target;
    // horizontal tracking is exact; height is eased so the hip bob of a stroke does not shake the view
    this._fy = this._fy == null ? p.y : this._fy + (p.y - this._fy) * Math.min(1, dt * 5);
    const d = new THREE.Vector3(p.x - tgt.x, this._fy - tgt.y, p.z - tgt.z);
    tgt.add(d);
    this.camera.position.add(d);
    this.key.position.copy(tgt).add(KEY_OFFSET);
    this.key.target.position.copy(tgt);
  }

  /** Move the camera to target + offset (and fov) with a short eased orbit, or instantly. */
  flyTo(target, offset, fov = this.camera.fov, instant = false) {
    if (instant) {
      this.tween = null;
      this.controls.target.copy(target);
      this.camera.position.copy(target).add(offset);
      this.camera.fov = fov;
      this.camera.updateProjectionMatrix();
      this._fy = target.y;
      return;
    }
    const from = this.camera.position.clone().sub(this.controls.target);
    this.tween = { from, to: offset.clone(), fov0: this.camera.fov, fov1: fov, t: 0, dur: 0.55 };
  }

  _tween(dt) {
    const tw = this.tween;
    if (!tw) return;
    tw.t = Math.min(tw.dur, tw.t + dt);
    const k = 1 - (1 - tw.t / tw.dur) ** 3;
    const a = tw.from.clone().normalize();
    const b = tw.to.clone().normalize();
    const q = new THREE.Quaternion().setFromUnitVectors(a, b);
    const dir = a.applyQuaternion(new THREE.Quaternion().slerp(q, k));
    const len = tw.from.length() + (tw.to.length() - tw.from.length()) * k;
    this.camera.position.copy(this.controls.target).addScaledVector(dir, len);
    this.camera.fov = tw.fov0 + (tw.fov1 - tw.fov0) * k;
    this.camera.updateProjectionMatrix();
    if (tw.t >= tw.dur) this.tween = null;
  }

  add(obj) { this.scene.add(obj); }

  remove(obj) {
    if (!obj) return;
    this.scene.remove(obj);
  }

  registerLines(material) {
    this.lineMaterials.add(material);
    material.resolution.set(this.el.clientWidth || 1, this.el.clientHeight || 1);
  }
}

function radialTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, '#fff');
  grad.addColorStop(0.45, '#fff');
  grad.addColorStop(1, '#000');
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 256);
  return new THREE.CanvasTexture(c);
}

/** 1 m grid (5 m lines brighter) whose lines fade out with distance from the origin. */
function fadingGrid(radius) {
  const pos = [];
  const col = [];
  const n = Math.ceil(radius);
  const alpha = (x, z) => Math.max(0, 1 - Math.hypot(x, z) / radius) ** 1.4;
  for (let i = -n; i <= n; i++) {
    const shade = i % 5 === 0 ? 0.36 : 0.22;
    for (let j = -n; j < n; j++) {
      for (const [x0, z0, x1, z1] of [[i, j, i, j + 1], [j, i, j + 1, i]]) {
        pos.push(x0, 0, z0, x1, 0, z1);
        col.push(shade, shade, shade + 0.03, alpha(x0, z0), shade, shade, shade + 0.03, alpha(x1, z1));
      }
    }
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geom.setAttribute('color', new THREE.Float32BufferAttribute(col, 4));
  const grid = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, depthWrite: false,
  }));
  grid.position.y = 0.003;
  return grid;
}

// ---------------------------------------------------------------------------------------------- skeleton
/** GVHMR world joints (data/motion/<id>.json) as an animated stick figure. */
export class JointSkeleton {
  constructor(data, stage) {
    this.fps = data.fps;
    this.n = data.n;
    this.parents = data.parents;
    this.nj = data.parents.length;
    this.J = Float32Array.from(data.joints);
    this.pairs = [];
    this.parents.forEach((p, j) => { if (p >= 0) this.pairs.push([p, j]); });
    const colors = [];
    for (const [, j] of this.pairs) {
      const c = LEFT_JOINTS.has(j) ? AMBER : BLUE;
      colors.push(c.r, c.g, c.b, c.r, c.g, c.b);
    }
    this.pos = new Float32Array(this.pairs.length * 6);
    this.geom = new LineSegmentsGeometry();
    this.geom.setPositions(this.pos);
    this.geom.setColors(colors);
    this.material = new LineMaterial({
      linewidth: 3, vertexColors: true, depthTest: false, transparent: true, opacity: 0.95,
    });
    stage?.registerLines(this.material);
    this.lines = new LineSegments2(this.geom, this.material);
    this.lines.renderOrder = 10;
    this.lines.frustumCulled = false;
    this.dots = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.024, 12, 8),
      new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true }),
      this.nj,
    );
    this.dots.renderOrder = 11;
    this.dots.frustumCulled = false;
    this.group = new THREE.Group();
    this.group.add(this.lines, this.dots);
    this.cur = new Float32Array(this.nj * 3);
    this._m = new THREE.Matrix4();
    this.setFrame(0);
  }

  joint(f, j, out = new THREE.Vector3()) {
    const i = (clampInt(f, 0, this.n - 1) * this.nj + j) * 3;
    return out.set(this.J[i], this.J[i + 1], this.J[i + 2]);
  }

  setFrame(f) {
    const a = clampInt(Math.floor(f), 0, this.n - 1);
    const b = Math.min(a + 1, this.n - 1);
    const w = Math.min(1, Math.max(0, f - a));
    const A = a * this.nj * 3;
    const B = b * this.nj * 3;
    for (let k = 0; k < this.nj * 3; k++) this.cur[k] = this.J[A + k] * (1 - w) + this.J[B + k] * w;
    this.pairs.forEach(([p, j], s) => {
      this.pos.set(this.cur.subarray(p * 3, p * 3 + 3), s * 6);
      this.pos.set(this.cur.subarray(j * 3, j * 3 + 3), s * 6 + 3);
    });
    const ib = this.geom.attributes.instanceStart.data;
    ib.array.set(this.pos);
    ib.needsUpdate = true;
    for (let j = 0; j < this.nj; j++) {
      this.dots.setMatrixAt(j, this._m.makeTranslation(this.cur[j * 3], this.cur[j * 3 + 1], this.cur[j * 3 + 2]));
    }
    this.dots.instanceMatrix.needsUpdate = true;
  }

  /** Pelvis position now, in world space (after the alignment transform on the group). */
  pelvisWorld(out = new THREE.Vector3()) {
    this.group.updateMatrixWorld(true);
    return out.set(this.cur[0], this.cur[1], this.cur[2]).applyMatrix4(this.group.matrixWorld);
  }

  /** Body frame of frame f in skeleton space (pelvis = 0, hips = 1 / 2). */
  frameAt(f) {
    return bodyFrame(this.joint(f, 0), this.joint(f, 1), this.joint(f, 2));
  }

  dispose() {
    this.geom.dispose();
    this.material.dispose();
    this.dots.geometry.dispose();
    this.dots.material.dispose();
  }
}

function clampInt(x, lo, hi) { return Math.min(hi, Math.max(lo, x | 0)); }
