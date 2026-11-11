// Shading for the card field. Cards only ever translate vertically, so every fragment can work out
// how deep it sits inside the slot between its neighbours (aNbr = top heights of the four neighbours)
// and derive occlusion, key-light shadow and warm inter-reflection analytically — no shadow maps.

export const cardVertex = /* glsl */ `
attribute vec4 aNbr;     // world-space top height of: front (towards camera), back, left row, right row
attribute vec2 aGlow;    // x: edge light of this card (0..1), y: light spilled from glowing neighbours
varying vec3 vLocal;
varying vec3 vN;
varying vec3 vWorld;
varying vec4 vNbr;
varying vec2 vGlow;
varying float vViewZ;
varying float vRnd;
void main() {
  vLocal = position;
  vN = normal;
  vec4 wp = modelMatrix * instanceMatrix * vec4(position, 1.0);
  vWorld = wp.xyz;
  vNbr = aNbr;
  vGlow = aGlow;
  vec4 mv = viewMatrix * wp;
  vViewZ = -mv.z;
  vRnd = fract(sin(dot(instanceMatrix[3].xz, vec2(12.9898, 78.233))) * 43758.5453);
  gl_Position = projectionMatrix * mv;
}`;

export const cardFragment = /* glsl */ `
precision highp float;
uniform vec3 uKeyDir;
uniform vec3 uKeyColor;
uniform vec3 uSkyColor;
uniform vec3 uHazeColor;
uniform vec2 uHaze;        // start, depth (view distance)
uniform vec3 uDims;        // card length, height, thickness
uniform vec2 uGaps;        // slot between cards in a row, slot between rows
varying vec3 vLocal;
varying vec3 vN;
varying vec3 vWorld;
varying vec4 vNbr;
varying vec2 vGlow;
varying float vViewZ;
varying float vRnd;

float hash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float vnoise(vec3 x) {
  vec3 i = floor(x), f = fract(x);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(hash(i), hash(i + vec3(1, 0, 0)), f.x), mix(hash(i + vec3(0, 1, 0)), hash(i + vec3(1, 1, 0)), f.x), f.y),
             mix(mix(hash(i + vec3(0, 0, 1)), hash(i + vec3(1, 0, 1)), f.x), mix(hash(i + vec3(0, 1, 1)), hash(i + vec3(1, 1, 1)), f.x), f.y), f.z);
}
// how much sky a point sees from depth d inside a slot of width g
float slot(float d, float g) { return g / (g + 3.4 * max(d + 0.04, 0.0)); }

void main() {
  vec3 n = normalize(vN);
  vec3 V = normalize(cameraPosition - vWorld);
  float hl = uDims.x * 0.5, H = uDims.y, T = uDims.z;
  float y = vWorld.y;

  // ---- material ---------------------------------------------------------------------------
  vec3 albedo; float gloss = 0.0; float milk = 0.0; vec3 emissive = vec3(0.0);
  float fall = exp(-(vLocal.x + hl) / 0.35);                 // LED sits at the block end
#if MAT == 0
  // frosted acrylic: warm tan body with long soft streaks running along the card
  float cloud = smoothstep(0.25, 0.75, vnoise(vec3(vLocal.x * 1.1, vLocal.y * 5.0, vRnd * 37.0)) * 0.6
                                     + vnoise(vec3(vLocal.x * 2.7 + 9.0, vLocal.y * 12.0, vRnd * 11.0)) * 0.4);
  albedo = mix(vec3(0.84, 0.63, 0.43), vec3(0.88, 0.68, 0.49), cloud);
  // inner spacer frame seen through the face (card.png): a paler band along the top and the block end
  float frame = max(smoothstep(H - 0.24, H - 0.20, vLocal.y), 1.0 - smoothstep(-hl + 0.24, -hl + 0.28, vLocal.x));
  albedo = mix(albedo, vec3(0.90, 0.72, 0.54), frame * 0.6);
  gloss = 0.16;
  milk = 1.0;
  emissive = albedo * vec3(1.0, 0.66, 0.36) * (vGlow.x * (0.16 + 2.4 * fall) + vGlow.y * (0.05 + 0.6 * fall));
#elif MAT == 1
  albedo = vec3(0.90, 0.74, 0.56);
  milk = 1.0;
  emissive = vec3(1.0, 0.80, 0.55) * 0.05 + vec3(1.0, 0.72, 0.44) * vGlow.x * (0.8 + 8.0 * fall);
#elif MAT == 2
  albedo = vec3(0.96, 0.87, 0.75);                           // pale satin block, flush with the face
  gloss = 0.3;
#else
  albedo = vec3(0.95, 0.93, 0.90);                           // polished eyelet rivet
#endif

  // ---- where am I inside the stack? --------------------------------------------------------
  float wT = max(n.y, 0.0), wF = max(n.z, 0.0), wB = max(-n.z, 0.0), wL = max(-n.x, 0.0), wR = max(n.x, 0.0), wD = max(-n.y, 0.0);
  float wSum = wT + wF + wB + wL + wR + wD + 1e-4;
  float dF = vNbr.x - y, dB = vNbr.y - y, dL = vNbr.z - y, dR = vNbr.w - y;
  float vis = (wT * slot(max(dF, 0.0) + max(dB, 0.0), 2.0 * uGaps.x + T)
             + wF * slot(dF, uGaps.x) + wB * slot(dB, uGaps.x)
             + wL * slot(dL, uGaps.y) + wR * slot(dR, uGaps.y) + wD * 0.15) / wSum;

  // ---- key light with analytic slot shadows --------------------------------------------------
  vec3 L = uKeyDir;
  float ndl = max(dot(n, L), 0.0);
  float lz = max(abs(L.z), 1e-3);
  bool fromFront = L.z >= 0.0;
  float dS = fromFront ? dF : dB;                             // the neighbour standing between me and the light
  float reachZ = uGaps.x * L.y / lz;                          // how far down a face the key reaches
  float reachL = uGaps.y * L.y / max(-L.x, 1e-3);
  float shZ = 1.0 - smoothstep(reachZ - 0.05, reachZ + 0.10, dS);
  float shL = 1.0 - smoothstep(reachL - 0.05, reachL + 0.12, dL);
  float edge = fromFront ? (T * 0.5 - vLocal.z) : (vLocal.z + T * 0.5);
  float shT = 1.0 - smoothstep(0.0, T, dS * lz / L.y - uGaps.x - edge);
  float sh = (wT * shT + (wF + wB) * shZ + wL * shL + (wR + wD)) / wSum;

  // ---- compose -----------------------------------------------------------------------------
  vec3 sky = uSkyColor * (0.86 + 0.14 * n.y);
  vec3 col = albedo * (sky * vis + uKeyColor * ndl * sh);
  // light that has bounced between the warm faces: darker and more saturated the deeper we are
  col += pow(albedo, vec3(1.5)) * vec3(1.0, 0.97, 0.92) * 0.62 * (1.0 - vis);
  // translucency: wherever a card stands clear of the one behind it, light passes through the frosted
  // sheet and the face turns milky; inside the stack it stays tan
  float open = (wF * (1.0 - smoothstep(-0.05, 0.14, dB)) + wB * (1.0 - smoothstep(-0.05, 0.14, dF))
              + wL * slot(dL, uGaps.y) * 0.45 + wR * slot(dR, uGaps.y) * 0.45 + wT * 0.12) / wSum;
  col = mix(col, vec3(0.93, 0.84, 0.75) * (0.9 + 0.1 * vis), milk * 0.82 * open);
  // frosted edges scatter light: thin bright rims along the top and the ends
  float rim = exp(-max(H - vLocal.y, 0.0) / 0.02) + exp(-max(hl - abs(vLocal.x), 0.0) / 0.018);
  col += albedo * uSkyColor * 0.2 * min(rim, 1.0) * (wF + wB + wL + wR) * milk;
  // soft sheen
  vec3 Hh = normalize(L + V);
  col += uKeyColor * gloss * pow(max(dot(n, Hh), 0.0), 40.0) * sh;
  float fres = pow(1.0 - max(dot(n, V), 0.0), 5.0);
  col += uSkyColor * fres * 0.16 * vis * step(0.01, gloss);
#if MAT == 3
  vec3 R = reflect(-V, n);
  vec3 env = mix(vec3(0.36, 0.26, 0.17), vec3(1.3, 1.24, 1.15), smoothstep(-0.25, 0.55, R.y - 0.35 * R.x));
  col = env * albedo * (0.4 + 0.6 * vis);
#endif
  col += emissive;

  float haze = clamp((vViewZ - uHaze.x) / uHaze.y, 0.0, 1.0);
  col = mix(col, uHazeColor, haze * haze);
  gl_FragColor = vec4(col, 1.0);
}`;

export const quadVertex = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

// Gather depth of field: circle of confusion from the real thin-lens formula, golden-angle spiral taps.
export const dofFragment = /* glsl */ `
precision highp float;
#define TAPS 44
uniform sampler2D tColor;
uniform sampler2D tDepth;
uniform vec2 uTexel;
uniform float uNear, uFar, uFocus, uCocScale, uMaxCoc;
varying vec2 vUv;
float viewZ(vec2 uv) { float d = texture2D(tDepth, uv).x; return (uNear * uFar) / (uFar - d * (uFar - uNear)); }
float coc(float z) { return min(uCocScale * abs(z - uFocus) / z, uMaxCoc); }
void main() {
  float zc = viewZ(vUv);
  float cc = coc(zc);
  vec3 acc = texture2D(tColor, vUv).rgb;
  float wsum = 1.0;
  float rot = 6.2831853 * fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  for (int k = 1; k < TAPS; k++) {
    float fk = float(k);
    float r = sqrt(fk / float(TAPS)) * uMaxCoc;
    float a = fk * 2.39996323 + rot;
    vec2 uv = vUv + vec2(cos(a), sin(a)) * r * uTexel;
    float zs = viewZ(uv);
    float cs = coc(zs);
    if (zs > zc) cs = min(cs, cc * 2.0);        // background must not bleed over sharper foreground
    float w = smoothstep(r - 1.0, r + 1.0, cs);
    acc += texture2D(tColor, uv).rgb * w;
    wsum += w;
  }
  gl_FragColor = vec4(acc / wsum, 1.0);
}`;
