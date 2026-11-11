import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { cardVertex, cardFragment, quadVertex, dofFragment } from './shaders.js';

const query = new URLSearchParams(location.search);
const STILL = query.has('still');                       // freeze idle motion (for comparing with concept.png)
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;

const canvas = document.getElementById('c');
const idxEl = document.getElementById('idx');

// Textures baked in Blender: the card as Cycles lights it (open), how that light fades inside the slot
// between cards (lut), the card under its own edge light (glow), and the tileable veins (frost).
const TEX_NAMES = ['open_front', 'open_end', 'open_top', 'lut_front', 'lut_far', 'lut_end', 'lut_top', 'lut_top_back', 'glow_front', 'glow_end', 'glow_top', 'frost'];
const texLoader = new THREE.TextureLoader();
const [P, gltf, ...texList] = await Promise.all([
  fetch('assets/params.json').then((r) => r.json()),
  new GLTFLoader().loadAsync('assets/card.glb'),
  ...TEX_NAMES.map((n) => texLoader.loadAsync(`assets/tex/${n}.png`)),
]);
const tex = Object.fromEntries(TEX_NAMES.map((n, k) => [n, texList[k]]));

// ----------------------------------------------------------------------------- layout
// Blender (x, y, z)  ->  three (x, z, -y):  x = card length, y = up, -z = stacking axis
const { length: L, height: H, thickness: T } = P.card;
const PITCH = P.pitchCard, ROW_PITCH = P.pitchRow;
const [R0, R1] = P.rows, [C0, C1] = P.cols;
const NR = R1 - R0 + 1, NC = C1 - C0 + 1, COUNT = NR * NC;
const F = P.field;
const rowScale = new Map(Object.entries(F.rowScale).map(([k, v]) => [Number(k), v]));
const b2t = (v) => new THREE.Vector3(v[0], v[2], -v[1]);

// ----------------------------------------------------------------------------- height field
// Same composition as blender/build_scene.py, generalised to a continuous focus (fi, fj) so the
// ridge can slide across the field.
const tent = (x) => Math.max(0, 1 - Math.abs(x));
const smooth = (x) => x * x * (3 - 2 * x);
function ridge(d) {
  const front = Math.max(0, F.front0 - F.frontSlope * (-d - 1));
  const back = Math.max(0, F.back0 - F.backSlope * (d - 1));
  if (d <= -1) return front;
  if (d >= 1) return back;
  return d < 0 ? THREE.MathUtils.lerp(F.front0, front, -d) : THREE.MathUtils.lerp(F.front0, back, d);
}
function prevRow(d) {
  const h = Math.max(0, F.prevTop - (d < 0 ? F.prevFrontSlope : F.prevBackSlope) * Math.abs(d));
  return Math.min(h, ridge(d));
}
function ripple(i, j, t) {
  const a = Math.sin(j * 0.21 + i * 1.3 + t * 0.5) * 0.5 + 0.5;
  const b = Math.sin(j * 0.057 - i * 0.6 + 1.0 - t * 0.23) * 0.5 + 0.5;
  return F.ripple * (0.6 * a + 0.8 * b);
}
function fieldAt(i, j, fi, fj, t) {
  const d = j - fj, r = i - fi;
  let h = 0;
  const w0 = tent(r);
  if (w0 > 0) h += w0 * (ridge(d) + (F.peak - F.front0) * smooth(tent(d)));
  const wp = tent(r + 1);
  if (wp > 0) h += wp * prevRow(d);
  for (const [k, s] of rowScale) {
    const w = tent(r - k);
    if (w > 0) h += w * s * ridge(d);
  }
  return h + ripple(i, j, t);
}
function glowAt(i, j, fi, fj) {
  const d = j - fj, r = i - fi;
  return smooth(tent(d)) * (tent(r) + 0.75 * tent(r + 1));
}

// ----------------------------------------------------------------------------- renderer
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
renderer.toneMapping = THREE.NeutralToneMapping;        // same highlight roll-off as Blender's Khronos PBR Neutral view

for (const [name, t] of Object.entries(tex)) {
  if (name === 'frost') {
    t.colorSpace = THREE.NoColorSpace;
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
  } else {
    t.colorSpace = THREE.SRGBColorSpace;
    if (name.startsWith('lut')) { t.generateMipmaps = false; t.minFilter = THREE.LinearFilter; }
    else t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  }
  t.needsUpdate = true;
}

const HAZE = new THREE.Color(...P.haze.color);
const scene = new THREE.Scene();
scene.background = HAZE;

// camera fitted to concept.png (see params.json)
const K = P.camera;
const BASE_ASPECT = 1672 / 941;
const BASE_HFOV = 2 * Math.atan(K.sensorMm / 2 / K.lensMm);
const camera = new THREE.PerspectiveCamera(10, BASE_ASPECT, 4, 90);
const homeTarget = b2t(K.aim);
const az = THREE.MathUtils.degToRad(K.azimuthDeg), el = THREE.MathUtils.degToRad(K.elevationDeg);
const viewDir = b2t([Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), -Math.sin(el)]);
const homePos = homeTarget.clone().addScaledVector(viewDir, -K.distance);
camera.position.copy(homePos);

const controls = new OrbitControls(camera, canvas);
controls.target.copy(homeTarget);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.enablePan = false;
controls.rotateSpeed = 0.28;
controls.zoomSpeed = 0.6;
controls.minDistance = K.distance * 0.55;
controls.maxDistance = K.distance * 1.45;
const homeSph = new THREE.Spherical().setFromVector3(homePos.clone().sub(homeTarget));
controls.minPolarAngle = homeSph.phi - 0.42;
controls.maxPolarAngle = Math.min(homeSph.phi + 0.2, 1.4);
controls.minAzimuthAngle = homeSph.theta - 0.55;
controls.maxAzimuthAngle = homeSph.theta + 0.55;
controls.update();

// ----------------------------------------------------------------------------- cards
const M = P.material, BK = P.bake;
const uniforms = {
  tOpenFront: { value: tex.open_front }, tOpenEnd: { value: tex.open_end }, tOpenTop: { value: tex.open_top },
  tLutFront: { value: tex.lut_front }, tLutFar: { value: tex.lut_far }, tLutEnd: { value: tex.lut_end }, tLutTop: { value: tex.lut_top }, tLutTopBack: { value: tex.lut_top_back },
  tGlowFront: { value: tex.glow_front }, tGlowEnd: { value: tex.glow_end }, tGlowTop: { value: tex.glow_top },
  tFrost: { value: tex.frost },
  uDims: { value: new THREE.Vector3(L, H, T) },
  uLut: { value: new THREE.Vector4(BK.stepRaise, BK.topRange[0], BK.topRange[1], BK.lutScale) },
  uOpenGain: { value: 1 / BK.openScale },
  uVeinDark: { value: new THREE.Vector3(...M.veinDark) },
  uVeinLight: { value: new THREE.Vector3(...M.veinLight) },
  uVeinAcross: { value: M.veinAcross },
  uVeinGain: { value: P.app.veinGain },
  uGlowColor: { value: new THREE.Vector3(...M.glow) },
  uGlowGain: { value: new THREE.Vector2(P.app.glowGain / BK.glowScale, P.app.spillGain) },
  uSpot: { value: new THREE.Vector4(0, 0, P.app.spotRadius, P.spot.gain * P.app.spotGain) },
  uSpotColor: { value: new THREE.Vector3(...P.spot.color).multiplyScalar(1.04) },
  uSpotDir: { value: b2t(P.spot.from).normalize() },
  uHazeColor: { value: HAZE },
  uHaze: { value: new THREE.Vector3(K.distance + P.haze.start, P.haze.depth, P.haze.base) },
};

const matrices = new THREE.InstancedBufferAttribute(new Float32Array(COUNT * 16), 16).setUsage(THREE.DynamicDrawUsage);
const nbrAttr = new THREE.InstancedBufferAttribute(new Float32Array(COUNT * 4), 4).setUsage(THREE.DynamicDrawUsage);
const glowAttr = new THREE.InstancedBufferAttribute(new Float32Array(COUNT * 2), 2).setUsage(THREE.DynamicDrawUsage);
const idx = (i, j) => (i - R0) * NC + (j - C0);
for (let i = R0; i <= R1; i++) {
  for (let j = C0; j <= C1; j++) {
    const o = idx(i, j) * 16;
    matrices.array[o] = matrices.array[o + 5] = matrices.array[o + 10] = matrices.array[o + 15] = 1;
    matrices.array[o + 12] = i * ROW_PITCH;
    matrices.array[o + 14] = -j * PITCH;
  }
}

const MAT_INDEX = { Acrylic: 0, Rivet: 1 };
gltf.scene.traverse((src) => {
  if (!src.isMesh) return;
  const geometry = src.geometry;
  geometry.setAttribute('aNbr', nbrAttr);
  geometry.setAttribute('aGlow', glowAttr);
  const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: cardVertex,
    fragmentShader: cardFragment,
    defines: { MAT: MAT_INDEX[src.material.name] ?? 0 },
  });
  const mesh = new THREE.InstancedMesh(geometry, material, COUNT);
  mesh.instanceMatrix = matrices;
  mesh.frustumCulled = false;
  scene.add(mesh);
});

// warm floor: any sliver seen between cards reads as light, never as a void
const floor = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), new THREE.MeshBasicMaterial({ color: new THREE.Color(0.62, 0.36, 0.17) }));
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.05;
scene.add(floor);

// ----------------------------------------------------------------------------- post: DOF -> bloom -> output
class SceneDofPass extends Pass {
  constructor() {
    super();
    this.rt = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4, depthTexture: new THREE.DepthTexture(1, 1) });
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        tColor: { value: this.rt.texture }, tDepth: { value: this.rt.depthTexture }, uTexel: { value: new THREE.Vector2() },
        uNear: { value: camera.near }, uFar: { value: camera.far }, uFocus: { value: K.distance }, uCocScale: { value: 0 }, uMaxCoc: { value: 20 },
      },
      vertexShader: quadVertex, fragmentShader: dofFragment, depthTest: false, depthWrite: false,
    });
    this.quad = new FullScreenQuad(this.material);
  }
  setSize(w, h) {
    this.rt.setSize(w, h);
    this.material.uniforms.uTexel.value.set(1 / w, 1 / h);
    this.material.uniforms.uMaxCoc.value = h * 0.021;
  }
  render(renderer, writeBuffer) {
    renderer.setRenderTarget(this.rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer);
    this.quad.render(renderer);
  }
}
const composer = new EffectComposer(renderer);
const dofPass = new SceneDofPass();
const bloomPass = new UnrealBloomPass(new THREE.Vector2(1, 1), P.app.bloom, 0.8, 0.9);
composer.addPass(dofPass);
composer.addPass(bloomPass);
composer.addPass(new OutputPass());

// The image is soft by design (depth of field), so cap the shaded pixel count instead of following the
// device pixel ratio blindly; the budget shrinks further if the GPU cannot hold the frame rate.
let pixelBudget = 5.5e6;
function resize() {
  const w = innerWidth, h = innerHeight, aspect = w / h;
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2, Math.sqrt(pixelBudget / (w * h))));
  renderer.setSize(w, h, false);
  composer.setPixelRatio(renderer.getPixelRatio());
  composer.setSize(w, h);
  camera.aspect = aspect;
  // keep the concept framing: lock the horizontal field on narrow windows, the vertical one on wide windows
  const baseVfov = 2 * Math.atan(Math.tan(BASE_HFOV / 2) / BASE_ASPECT);
  const vfov = aspect >= BASE_ASPECT ? baseVfov : 2 * Math.atan(Math.tan(BASE_HFOV / 2) / aspect);
  camera.fov = THREE.MathUtils.radToDeg(vfov);
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();

// ----------------------------------------------------------------------------- state
const off = new Float32Array(COUNT), vel = new Float32Array(COUNT), glow = new Float32Array(COUNT);
const focus = { i: 0, j: 0, ti: 0, tj: 0 };          // eased / target focus card
const hover = { i: 0, j: 0, on: false, amt: 0 };
let time = 0;
for (let i = R0; i <= R1; i++) for (let j = C0; j <= C1; j++) {
  off[idx(i, j)] = fieldAt(i, j, 0, 0, 0);
  glow[idx(i, j)] = glowAt(i, j, 0, 0);
}

function setFocus(i, j) {
  focus.ti = THREE.MathUtils.clamp(Math.round(i), R0 + 2, R1 - 3);
  focus.tj = THREE.MathUtils.clamp(Math.round(j), C0 + 22, C1 - 14);
  const pad = (n, w) => (n < 0 ? '-' : '') + String(Math.abs(n)).padStart(w, '0');
  idxEl.textContent = `${pad(focus.ti, 2)} · ${pad(focus.tj, 3)}`;
}
setFocus(0, 0);

// ----------------------------------------------------------------------------- picking
// March the pointer ray through the stack: first card whose front face it meets below its top edge.
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
function pick(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const { origin: o, direction: d } = raycaster.ray;
  if (Math.abs(d.z) < 1e-4) return null;
  const step = d.z < 0 ? 1 : -1;                       // d.z < 0: ray travels towards larger j
  for (let j = step > 0 ? C0 : C1; j >= C0 && j <= C1; j += step) {
    const zFace = -j * PITCH + (step > 0 ? T / 2 : -T / 2);
    const s = (zFace - o.z) / d.z;
    if (s <= 0) continue;
    const x = o.x + d.x * s, y = o.y + d.y * s;
    const i = Math.round(x / ROW_PITCH);
    if (i < R0 || i > R1 || Math.abs(x - i * ROW_PITCH) > L / 2) continue;
    const base = off[idx(i, j)];
    if (y <= base + H && y >= base - 0.05) return { i, j };
  }
  return null;
}

let down = null, dragging = false;
canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; dragging = false; });
canvas.addEventListener('pointermove', (e) => {
  if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) { dragging = true; canvas.classList.add('dragging'); }
  const hit = dragging ? null : pick(e.clientX, e.clientY);
  hover.on = !!hit;
  if (hit) { hover.i = hit.i; hover.j = hit.j; }
});
canvas.addEventListener('pointerleave', () => { hover.on = false; });
addEventListener('pointerup', (e) => {
  canvas.classList.remove('dragging');
  if (down && !dragging && e.target === canvas) {
    const hit = pick(e.clientX, e.clientY);
    if (hit) (hit.i === focus.ti && hit.j === focus.tj) ? setFocus(0, 0) : setFocus(hit.i, hit.j);
  }
  down = null; dragging = false;
});

// camera reset
let resetAnim = null;
function resetView() {
  resetAnim = { t: 0, p0: camera.position.clone(), t0: controls.target.clone() };
  setFocus(0, 0);
}
canvas.addEventListener('dblclick', resetView);
addEventListener('keydown', (e) => {
  const k = e.key;
  if (k === 'r' || k === 'R' || k === 'Escape') resetView();
  else if (k === 'ArrowLeft') setFocus(focus.ti, focus.tj + 1);
  else if (k === 'ArrowRight') setFocus(focus.ti, focus.tj - 1);
  else if (k === 'ArrowUp') setFocus(focus.ti + 1, focus.tj);
  else if (k === 'ArrowDown') setFocus(focus.ti - 1, focus.tj);
  else return;
  e.preventDefault();
});

// ----------------------------------------------------------------------------- frame
const STIFF = 90, DAMP = 13;
const focusWorld = new THREE.Vector3(), tmp = new THREE.Vector3();
let focusDepth = K.distance;
function update(dt) {
  if (!STILL && !REDUCED) time += dt;
  const kf = 1 - Math.exp(-dt * 5.5);
  focus.i += (focus.ti - focus.i) * kf;
  focus.j += (focus.tj - focus.j) * kf;
  hover.amt += ((hover.on ? 1 : 0) - hover.amt) * (1 - Math.exp(-dt * 10));
  const kg = 1 - Math.exp(-dt * 7);

  const m = matrices.array, g = glowAttr.array;
  for (let i = R0; i <= R1; i++) {
    for (let j = C0; j <= C1; j++) {
      const k = idx(i, j);
      let target = fieldAt(i, j, focus.i, focus.j, time);
      let gt = glowAt(i, j, focus.i, focus.j);
      if (hover.amt > 0.001 && i === hover.i) {
        const dj = j - hover.j;
        if (dj > -4 && dj < 4) {
          const w = Math.exp(-dj * dj / 1.1) * hover.amt;
          target += 0.16 * w;
          gt = Math.max(gt, 0.22 * w * (dj === 0 ? 1 : 0.3));
        }
      }
      vel[k] += (STIFF * (target - off[k]) - DAMP * vel[k]) * dt;
      off[k] += vel[k] * dt;
      glow[k] += (gt - glow[k]) * kg;
      m[k * 16 + 13] = off[k];
      g[k * 2] = glow[k];
    }
  }
  // neighbour tops drive the analytic occlusion in the shader
  const nb = nbrAttr.array;
  for (let i = R0; i <= R1; i++) {
    for (let j = C0; j <= C1; j++) {
      const k = idx(i, j), o = k * 4;
      nb[o] = (j > C0 ? off[k - 1] : off[k]) + H;            // front (towards camera, j - 1)
      nb[o + 1] = (j < C1 ? off[k + 1] : off[k]) + H;        // back
      nb[o + 2] = (i > R0 ? off[k - NC] : off[k]) + H;       // left row
      nb[o + 3] = (i < R1 ? off[k + NC] : off[k]) + H;       // right row
      // warm light spilled onto the neighbours of a lit card
      g[k * 2 + 1] = Math.max(j > C0 ? glow[k - 1] : 0, j < C1 ? glow[k + 1] : 0, 0.5 * (j > C0 + 1 ? glow[k - 2] : 0), 0.5 * (j < C1 - 1 ? glow[k + 2] : 0));
    }
  }
  matrices.needsUpdate = nbrAttr.needsUpdate = glowAttr.needsUpdate = true;

  if (resetAnim) {
    resetAnim.t = Math.min(1, resetAnim.t + dt / 0.9);
    const e = smooth(resetAnim.t);
    camera.position.lerpVectors(resetAnim.p0, homePos, e);
    controls.target.lerpVectors(resetAnim.t0, homeTarget, e);
    if (resetAnim.t >= 1) resetAnim = null;
  }
  controls.update();
  uniforms.uSpot.value.x = focus.i * ROW_PITCH + P.spot.offset[0];
  uniforms.uSpot.value.y = -(focus.j * PITCH + P.spot.offset[1]);

  // focus the lens where the Blender camera does: a couple of cards in front of the pulled-out one
  const [fr, fc] = K.focusCard;
  focusWorld.set((focus.i + fr) * ROW_PITCH - L / 2, H + fieldAt(Math.round(focus.i) + fr, Math.round(focus.j) + fc, focus.i, focus.j, time), -(focus.j + fc) * PITCH);
  camera.getWorldDirection(tmp);
  const depth = focusWorld.sub(camera.position).dot(tmp);
  focusDepth += (depth - focusDepth) * (1 - Math.exp(-dt * 4));
  const u = dofPass.material.uniforms;
  const f = K.lensMm / 1000;
  u.uFocus.value = focusDepth;
  // thin lens: coc[px] = f^2 / (N (zf - f)) * |z - zf| / z / sensor * width
  u.uCocScale.value = 0.55 * (f * f) / (K.fstop * (focusDepth - f)) / (K.sensorMm / 1000) * dofPass.rt.width * (BASE_HFOV / (2 * Math.atan(Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * camera.aspect)));
}

const stats = { cpuMs: 0, rawMs: 0 };
let last = performance.now(), slow = 0;
function frame(now) {
  const raw = (now - last) / 1000;
  const dt = Math.min(raw, 1 / 20);
  last = now;
  slow = raw > 1 / 40 && raw < 0.25 ? slow + 1 : Math.max(0, slow - 2);
  if (slow > 45 && pixelBudget > 1.0e6) { pixelBudget *= 0.75; slow = 0; resize(); }
  update(dt);
  composer.render();
  stats.cpuMs = performance.now() - now; stats.rawMs = raw * 1000;
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
document.getElementById('loading').classList.add('done');

if (import.meta.env?.DEV) window.__app = { step: (n = 60, dt = 1 / 60) => { for (let k = 0; k < n; k++) update(dt); composer.render(); }, stats, P, camera, controls, uniforms, focus, setFocus, dofPass, bloomPass, renderer };
