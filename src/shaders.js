// Shading for the card field, driven by textures baked in Blender (blender/build_scene.py --bake).
//
//   open_*   what Cycles sees, looking along the scene camera, on each face of a card that stands clear
//            of the stack: the frosted see-through sheets, the seam, the inlaid block, the rivet
//   lut_*    the same light measured on a card that sits in the stack, divided by the unoccluded card.
//            One axis of each map is depth below the neighbours' tops, so it answers
//            "how much of that light is left this far down the slot" for any height difference.
//            The sheets are see-through, so the cards in front (lut_front) and behind (lut_far) each
//            get their own map
//   glow_*   the card lit only by its own edge light
//   frost    tileable veins; every card slides the map by its own offset
//
// Cards only ever move vertically, so each fragment knows how deep it sits between its neighbours
// (aNbr = their top heights) and the baked light can follow the motion exactly, without shadow maps.

export const cardVertex = /* glsl */ `
attribute vec4 aNbr;     // world-space top height of: front (towards camera), back, left row, right row
attribute vec2 aGlow;    // x: edge light of this card (0..1), y: light spilled from glowing neighbours
varying vec3 vLocal;
varying vec3 vN;
varying vec3 vWorld;
varying vec4 vNbr;
varying vec2 vGlow;
varying float vViewZ;
varying vec2 vRnd;
void main() {
  vLocal = position;
  vN = normal;
  vec4 wp = modelMatrix * instanceMatrix * vec4(position, 1.0);
  vWorld = wp.xyz;
  vNbr = aNbr;
  vGlow = aGlow;
  vec4 mv = viewMatrix * wp;
  vViewZ = -mv.z;
  float r = fract(sin(dot(instanceMatrix[3].xz, vec2(12.9898, 78.233))) * 43758.5453);
  vRnd = vec2(r * 7.13, r * 3.71);
  gl_Position = projectionMatrix * mv;
}`;

export const cardFragment = /* glsl */ `
precision highp float;
uniform sampler2D tOpenFront, tOpenEnd, tOpenTop;
uniform sampler2D tLutFront, tLutFar, tLutEnd, tLutTop, tLutTopBack;
uniform sampler2D tGlowFront, tGlowEnd, tGlowTop;
uniform sampler2D tFrost;
uniform vec3 uDims;        // card length, height, thickness
uniform vec4 uLut;         // step raise of the face bake, top-wedge range (d0, d1), ratio scale
uniform float uOpenGain;   // undoes the headroom scale the open_* maps were stored with
uniform vec3 uVeinDark, uVeinLight;
uniform float uVeinAcross, uVeinGain;
uniform vec3 uGlowColor;
uniform vec2 uGlowGain;    // own edge light, spill on neighbours
uniform vec4 uSpot;        // xz centre, radius, gain: pool of light around the focused card
uniform vec3 uSpotDir;     // where that light comes from
uniform vec3 uSpotColor;
uniform vec3 uHazeColor;
uniform vec3 uHaze;        // start, depth (view distance), faint veil over everything
varying vec3 vLocal;
varying vec3 vN;
varying vec3 vWorld;
varying vec4 vNbr;
varying vec2 vGlow;
varying float vViewZ;
varying vec2 vRnd;

// slot-light ratio at depth d below the neighbour's top (negative = above it)
vec3 slotFace(sampler2D lut, float u, float d) {
  float v = (uDims.y - uLut.x - d) / uDims.y;
  // below ~0.35 the bake is close to the floor under the card; deeper slots just stay as dark as that
  return texture2D(lut, vec2(u, clamp(v, 0.16, 0.996))).rgb * uLut.w;
}
vec3 slotTop(sampler2D lut, float d, float v) {
  float u = (d - uLut.y) / (uLut.z - uLut.y);
  return texture2D(lut, vec2(clamp(u, 0.004, 0.996), v)).rgb * uLut.w;
}

void main() {
  vec3 n = normalize(vN);
  vec3 an = abs(n);
  float L = uDims.x, H = uDims.y, T = uDims.z;
  float y = vWorld.y;
  float dF = vNbr.x - y, dB = vNbr.y - y, dL = vNbr.z - y, dR = vNbr.w - y;
  vec3 px = dFdx(vLocal), py = dFdy(vLocal);     // continuous across the bevels, unlike the projected uv

  vec3 base, light, glow;
#if MAT == 1
  bool front = true, top = false;                // rivet: seen as in the front view
#else
  bool top = an.y > max(an.x, an.z);
  bool front = !top && an.z >= an.x;
#endif
  if (front) {
    vec2 uv = vec2(vLocal.x / L + 0.5, vLocal.y / H);
    vec2 gx = vec2(px.x / L, px.y / H), gy = vec2(py.x / L, py.y / H);
    base = textureGrad(tOpenFront, uv, gx, gy).rgb;
    glow = textureGrad(tGlowFront, uv, gx, gy).rgb;
    float dNear = n.z >= 0.0 ? dF : dB, dFar = n.z >= 0.0 ? dB : dF;
    // how deep the face sits behind the card in front of it, times what shows through from behind.
    // open_front was shot with the card behind level with this one, i.e. this deep below its top:
    float dBase = H - vLocal.y;
    light = slotFace(tLutFront, uv.x, dNear) * slotFace(tLutFar, uv.x, dFar) / max(slotFace(tLutFar, uv.x, dBase), vec3(0.05));
  } else if (!top) {
    vec2 uv = vec2(0.5 + vLocal.z / T, vLocal.y / H);
    vec2 gx = vec2(px.z / T, px.y / H), gy = vec2(py.z / T, py.y / H);
    base = textureGrad(tOpenEnd, uv, gx, gy).rgb;
    glow = textureGrad(tGlowEnd, uv, gx, gy).rgb;
    light = slotFace(tLutEnd, uv.x, n.x < 0.0 ? dL : dR);
  } else {
    vec2 uv = vec2(vLocal.x / L + 0.5, 0.5 - vLocal.z / T);
    vec2 gx = vec2(px.x / L, -px.z / T), gy = vec2(py.x / L, -py.z / T);
    base = textureGrad(tOpenTop, uv, gx, gy).rgb;
    glow = textureGrad(tGlowTop, uv, gx, gy).rgb;
    light = slotTop(tLutTop, dF, uv.y) * slotTop(tLutTopBack, dB, uv.y);
    if (n.y < 0.0) light *= 0.25;                // underside
  }

#if MAT == 0
  // veins: u along the card, v up it and drifting through the thickness (same mapping as in Blender)
  float c = texture2D(tFrost, vec2(vLocal.x / L, vLocal.y / H - vLocal.z * uVeinAcross) + vRnd).r;
  base *= mix(vec3(1.0), mix(uVeinDark, uVeinLight, c) / mix(uVeinDark, uVeinLight, 0.5), uVeinGain);
#endif

  // pool of light around the focused card
  vec2 sp = vWorld.xz - uSpot.xy;
  float spot = uSpot.w * (0.5 + 0.5 * max(dot(n, uSpotDir), 0.0)) * exp(-dot(sp, sp) / (uSpot.z * uSpot.z));

  base *= uOpenGain;
  vec3 col = base * light;
  // the pool is white light on a pale sheet, while the baked light is already warm from its trip through
  // the stack: add it mostly as its own colour rather than as more of the same
  col += mix(col, vec3(dot(col, vec3(0.2126, 0.7152, 0.0722))) * uSpotColor, 0.7) * spot;
  float fall = exp(-(vLocal.x + 0.5 * L) / 0.5);
  col += glow * uGlowGain.x * vGlow.x;
  col += base * light * uGlowColor * uGlowGain.y * vGlow.y * (0.25 + fall);

  float haze = clamp((vViewZ - uHaze.x) / uHaze.y, 0.0, 1.0);
  col = mix(col, uHazeColor, uHaze.z + (1.0 - uHaze.z) * haze * haze);
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
