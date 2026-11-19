import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const params = new URLSearchParams(window.location.search);
const $ = (sel) => document.querySelector(sel);

if (params.has("clean")) document.body.classList.add("clean");

const stage = document.querySelector(".stage");
const canvas = $("#viewer");
const loader = $("#loader");
const errorPanel = $("#errorPanel");
const rotateBtn = $("#rotateBtn");
const wireBtn = $("#wireBtn");

let renderer;
try {
  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
    powerPreference: "high-performance",
    preserveDrawingBuffer: false,
  });
} catch (error) {
  console.error(error);
  showError(error);
  throw error;
}

renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0xf7f6f1, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = null;

// A soft white-to-warm-white backdrop gives the clear glass an edge without
// introducing props or colour into the asset itself.
function createBackdropTexture() {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 1024;
  const ctx = c.getContext("2d");
  const g = ctx.createLinearGradient(0, 0, 0, 1024);
  g.addColorStop(0.0, "#ffffff");
  g.addColorStop(0.54, "#ffffff");
  g.addColorStop(0.82, "#fbfaf7");
  g.addColorStop(1.0, "#f4f3ee");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 512, 1024);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
const usePureWhite = params.get("white") === "1";
if (usePureWhite) {
  renderer.setClearColor(0xffffff, 1);
  document.body.style.background = "#ffffff";
}
const backdrop = new THREE.Mesh(
  new THREE.PlaneGeometry(8, 8),
  new THREE.MeshBasicMaterial({
    map: createBackdropTexture(),
    depthWrite: false,
    depthTest: false,
    toneMapped: false,
  })
);
backdrop.position.set(0, 1.7, -2.5);
backdrop.renderOrder = -100;
backdrop.visible = !usePureWhite;
scene.add(backdrop);

const camera = new THREE.PerspectiveCamera(26, 1, 0.03, 20);
camera.position.set(0.31, 0.285, 1.16);

const controls = new OrbitControls(camera, canvas);
controls.target.set(0.0, 0.208, 0.0);
controls.enableDamping = true;
controls.dampingFactor = 0.065;
controls.enablePan = false;
controls.minDistance = 0.48;
controls.maxDistance = 2.4;
controls.minPolarAngle = Math.PI * 0.24;
controls.maxPolarAngle = Math.PI * 0.525;
controls.rotateSpeed = 0.65;
controls.zoomSpeed = 0.8;
controls.update();

const turntable = new THREE.Group();
scene.add(turntable);

let petalGlowMaterials = [];
let pollenGroup = null;
let pollenMaterial = null;
let glowHalo = null;

// ---------------------------------------------------------------------------
// environment and light
// ---------------------------------------------------------------------------

const pmrem = new THREE.PMREMGenerator(renderer);
const room = new RoomEnvironment(renderer);
const envTexture = pmrem.fromScene(room, 0.045).texture;
scene.environment = envTexture;
pmrem.dispose();
room.dispose?.();

const hemi = new THREE.HemisphereLight(0xfffdf7, 0xb9c0b3, 0.60);
scene.add(hemi);

const key = new THREE.DirectionalLight(0xfffbf2, 1.25);
key.position.set(-1.35, 2.75, 1.55);
key.target.position.set(0.0, 0.21, 0.0);
key.castShadow = true;
key.shadow.mapSize.set(1536, 1536);
key.shadow.camera.left = -0.55;
key.shadow.camera.right = 0.55;
key.shadow.camera.top = 0.7;
key.shadow.camera.bottom = -0.05;
key.shadow.camera.near = 0.2;
key.shadow.camera.far = 5.0;
key.shadow.bias = -0.00035;
key.shadow.normalBias = 0.0012;
key.shadow.radius = 16;
scene.add(key, key.target);

const fill = new THREE.DirectionalLight(0xeaf1e8, 0.38);
fill.position.set(1.6, 1.45, 0.9);
scene.add(fill);

const rim = new THREE.DirectionalLight(0xffffff, 0.48);
rim.position.set(-1.2, 1.8, -1.4);
scene.add(rim);

// A very soft warm edge so the petals read as gently self-lit.
const glowRim = new THREE.DirectionalLight(0xffdca0, 0.58);
glowRim.position.set(0.9, 1.25, -1.1);
scene.add(glowRim);

// Invisible white studio floor: only the shadow is visible.
const shadowFloor = new THREE.Mesh(
  new THREE.PlaneGeometry(4.0, 4.0),
  new THREE.ShadowMaterial({ color: 0x29322b, opacity: 0.045 })
);
shadowFloor.rotation.x = -Math.PI / 2;
shadowFloor.position.y = -0.0008;
shadowFloor.receiveShadow = true;
scene.add(shadowFloor);

function createContactShadow() {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 512;
  const ctx = c.getContext("2d");
  const grd = ctx.createRadialGradient(256, 256, 25, 256, 256, 250);
  grd.addColorStop(0.0, "rgba(50, 48, 39, 0.34)");
  grd.addColorStop(0.32, "rgba(50, 48, 39, 0.17)");
  grd.addColorStop(0.68, "rgba(50, 48, 39, 0.045)");
  grd.addColorStop(1.0, "rgba(50, 48, 39, 0)");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, 512, 512);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

const contact = new THREE.Mesh(
  new THREE.PlaneGeometry(0.34, 0.27),
  new THREE.MeshBasicMaterial({
    map: createContactShadow(),
    transparent: true,
    opacity: 0.34,
    depthWrite: false,
    toneMapped: false,
  })
);
contact.rotation.x = -Math.PI / 2;
contact.position.set(0.0, 0.0002, 0.0);
scene.add(contact);

// ---------------------------------------------------------------------------
// subtle floral glow
// ---------------------------------------------------------------------------

function createGlowHalo() {
  const c = document.createElement("canvas");
  c.width = c.height = 512;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(256, 256, 8, 256, 256, 250);
  g.addColorStop(0.0, "rgba(255, 211, 91, 0.52)");
  g.addColorStop(0.32, "rgba(255, 208, 92, 0.20)");
  g.addColorStop(0.72, "rgba(255, 215, 124, 0.055)");
  g.addColorStop(1.0, "rgba(255, 218, 140, 0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 512, 512);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: tex,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.NormalBlending,
      opacity: 0.98,
    })
  );
  sprite.position.set(0.0, 0.25, -0.32);
  sprite.scale.set(0.60, 0.78, 1);
  sprite.renderOrder = -6;
  return sprite;
}

function mulberry32(seed) {
  return function random() {
    let t = (seed += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createPollen() {
  const rng = mulberry32(7319);
  const count = 380;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const phases = new Float32Array(count);
  const alphas = new Float32Array(count);
  const gold = new THREE.Color(0xf0c45a);
  const cream = new THREE.Color(0xffe6a6);
  for (let i = 0; i < count; i += 1) {
    let x;
    let y;
    let z;
    if (i < 205) {
      const a = rng() * Math.PI * 2;
      const r = 0.035 + Math.pow(rng(), 0.62) * 0.145;
      x = Math.cos(a) * r;
      z = Math.sin(a) * r * 0.66;
      y = 0.255 + rng() * 0.19;
    } else {
      x = (rng() - 0.5) * 0.30;
      y = 0.13 + rng() * 0.30;
      z = (rng() - 0.5) * 0.18;
    }
    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;
    const col = gold.clone().lerp(cream, rng() * 0.65);
    col.multiplyScalar(0.88 + rng() * 0.18);
    colors[i * 3] = col.r;
    colors[i * 3 + 1] = col.g;
    colors[i * 3 + 2] = col.b;
    sizes[i] = 2.0 + Math.pow(rng(), 1.45) * 5.2;
    phases[i] = rng() * Math.PI * 2;
    alphas[i] = 0.30 + rng() * 0.36;
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("aColor", new THREE.BufferAttribute(colors, 3));
  geom.setAttribute("aSize", new THREE.BufferAttribute(sizes, 1));
  geom.setAttribute("aPhase", new THREE.BufferAttribute(phases, 1));
  geom.setAttribute("aAlpha", new THREE.BufferAttribute(alphas, 1));
  pollenMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uPixelRatio: { value: renderer.getPixelRatio() },
    },
    transparent: true,
    depthWrite: false,
    depthTest: false,
    blending: THREE.NormalBlending,
    vertexShader: `
      uniform float uTime;
      uniform float uPixelRatio;
      attribute vec3 aColor;
      attribute float aSize;
      attribute float aPhase;
      attribute float aAlpha;
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        vec3 p = position;
        float t = uTime * 0.34 + aPhase;
        p.x += sin(t * 1.1) * 0.0022;
        p.y += sin(t * 0.72 + aPhase) * 0.0018;
        p.z += cos(t * 0.87) * 0.0018;
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_Position = projectionMatrix * mv;
        float twinkle = 0.42 + 0.58 * pow(0.5 + 0.5 * sin(uTime * 1.15 + aPhase), 2.0);
        gl_PointSize = aSize * uPixelRatio * (1.0 / max(0.22, -mv.z));
        vColor = aColor;
        vAlpha = aAlpha * (0.48 + 0.52 * twinkle);
      }
    `,
    fragmentShader: `
      precision highp float;
      varying vec3 vColor;
      varying float vAlpha;
      void main() {
        vec2 c = gl_PointCoord - vec2(0.5);
        float d = dot(c, c);
        if (d > 0.25) discard;
        float core = 1.0 - smoothstep(0.0, 0.075, d);
        float halo = 1.0 - smoothstep(0.01, 0.25, d);
        vec3 col = mix(vColor, vec3(1.0, 0.96, 0.78), core * 0.35);
        gl_FragColor = vec4(col, (core * 0.60 + halo * 0.30) * vAlpha);
      }
    `,
  });
  const pts = new THREE.Points(geom, pollenMaterial);
  pts.frustumCulled = false;
  pts.renderOrder = 3;
  const group = new THREE.Group();
  group.add(pts);
  return group;
}

const glowHaloSprite = createGlowHalo();
glowHalo = glowHaloSprite;
scene.add(glowHalo);
pollenGroup = createPollen();
turntable.add(pollenGroup);

function createFlowerGlowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(64, 64, 1, 64, 64, 63);
  g.addColorStop(0, "rgba(255, 228, 138, 0.52)");
  g.addColorStop(0.28, "rgba(255, 216, 111, 0.20)");
  g.addColorStop(0.68, "rgba(255, 207, 92, 0.035)");
  g.addColorStop(1, "rgba(255, 207, 92, 0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function addFlowerGlows(root) {
  const tex = createFlowerGlowTexture();
  root.traverse((node) => {
    if (!node.name || !node.name.startsWith("Flower_")) return;
    const mat = new THREE.SpriteMaterial({
      map: tex,
      color: 0xffd675,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.NormalBlending,
      opacity: 0.085,
    });
    const sprite = new THREE.Sprite(mat);
    sprite.scale.set(0.043, 0.043, 1);
    sprite.renderOrder = 1;
    node.add(sprite);
  });
}

// ---------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------

const loader3d = new GLTFLoader();
const highURL = new URL("../model/rapeseed_vase_high.glb", import.meta.url).href;
const wireURL = new URL("../model/rapeseed_vase_wireframe.glb", import.meta.url).href;

let beautyRoot = null;
let wireRoot = null;
let wirePromise = null;
let wireActive = params.get("wire") === "1";
let autoRotate = params.get("still") !== "1";
let rotation = THREE.MathUtils.degToRad(parseFloat(params.get("a") || "-6"));
if (!Number.isFinite(rotation)) rotation = 0;
turntable.rotation.y = rotation;

function showError(error) {
  console.error(error);
  loader?.classList.add("is-done");
  if (errorPanel) errorPanel.hidden = false;
}

function setMaterialQuality(root) {
  root.traverse((child) => {
    if (!child.isMesh) return;
    child.frustumCulled = true;
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    let isTransmissive = false;
    for (const material of materials) {
      if (!material) continue;
      if (material.isMeshPhysicalMaterial) {
        isTransmissive = isTransmissive || material.transmission > 0.01;
        material.envMapIntensity = material.transmission > 0.01 ? 1.05 : 0.50;
      } else if (material.isMeshStandardMaterial) {
        material.envMapIntensity = 0.50;
      }
      if (material.name && material.name.startsWith("Petals_")) {
        material.emissive = new THREE.Color(0xc99b3e);
        material.emissiveIntensity = 0.060;
        material.needsUpdate = true;
        if (!petalGlowMaterials.includes(material)) petalGlowMaterials.push(material);
      }
      material.shadowSide = THREE.FrontSide;
      if (material.name === "Water_Surface" || material.name === "Water_Line") {
        material.transparent = true;
        material.depthTest = false;
        material.depthWrite = false;
        material.renderOrder = 2;
        material.needsUpdate = true;
      }
    }
    child.castShadow = !isTransmissive;
    child.receiveShadow = false;
    child.renderOrder = isTransmissive ? 1 : 0;
  });
}

async function loadBeauty() {
  const gltf = await loader3d.loadAsync(highURL);
  beautyRoot = gltf.scene;
  beautyRoot.name = "Rapeseed_High";
  setMaterialQuality(beautyRoot);
  addFlowerGlows(beautyRoot);
  turntable.add(beautyRoot);
  if (params.has("wire")) {
    await ensureWire();
  }
}

async function ensureWire() {
  if (wireRoot) return wireRoot;
  if (!wirePromise) {
    wirePromise = loader3d.loadAsync(wireURL).then((gltf) => {
      const root = gltf.scene;
      root.name = "Rapeseed_Wire";
      root.traverse((child) => {
        if (!child.isMesh) return;
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        const base = materials.find((m) => m?.name === "Glass_Clear") ? 0.13 : 0.44;
        child.material = new THREE.MeshBasicMaterial({
          color: 0x4d696c,
          wireframe: true,
          transparent: true,
          opacity: base,
          depthWrite: false,
          toneMapped: false,
        });
        child.castShadow = false;
        child.receiveShadow = false;
      });
      wireRoot = root;
      wireRoot.visible = false;
      turntable.add(wireRoot);
      return wireRoot;
    });
  }
  return wirePromise;
}

function applyView() {
  const view = params.get("view");
  if (view === "detail") {
    camera.position.set(0.19, 0.33, 0.68);
    controls.target.set(0.0, 0.31, 0.0);
    controls.minDistance = 0.35;
  } else if (view === "vase") {
    camera.position.set(0.15, 0.15, 0.52);
    controls.target.set(0.0, 0.105, 0.0);
    controls.minDistance = 0.28;
  } else {
    camera.position.set(0.31, 0.285, 1.16);
    controls.target.set(0.0, 0.208, 0.0);
  }
  controls.update();
}

function resize() {
  const width = stage.clientWidth || window.innerWidth;
  const height = stage.clientHeight || window.innerHeight;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(width, height, false);
  camera.aspect = width / Math.max(1, height);
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function setWire(active, userInitiated = true) {
  wireActive = !!active;
  wireBtn?.setAttribute("aria-pressed", String(wireActive));
  document.body.classList.toggle("wire", wireActive);
  if (!beautyRoot) return;
  if (wireActive) {
    ensureWire().then(() => {
      if (!wireRoot) return;
      beautyRoot.visible = false;
      wireRoot.visible = true;
      wireRoot.rotation.copy(beautyRoot.rotation);
      if (pollenGroup) pollenGroup.visible = false;
      if (glowHalo) glowHalo.visible = false;
      renderer.toneMapping = THREE.NoToneMapping;
      shadowFloor.visible = false;
      contact.visible = false;
    }).catch(showError);
  } else {
    if (wireRoot) wireRoot.visible = false;
    beautyRoot.visible = true;
    if (pollenGroup) pollenGroup.visible = true;
    if (glowHalo) glowHalo.visible = true;
    renderer.toneMapping = THREE.NoToneMapping;
    shadowFloor.visible = true;
    contact.visible = true;
  }
  if (userInitiated) {
    // A small visual acknowledgement in console for automation without UI.
    console.info(`[rapeseed] wireframe: ${wireActive ? "on" : "off"}`);
  }
}

function setAutoRotate(active) {
  autoRotate = !!active;
  rotateBtn?.setAttribute("aria-pressed", String(autoRotate));
}

rotateBtn?.addEventListener("click", () => setAutoRotate(!autoRotate));
wireBtn?.addEventListener("click", () => setWire(!wireActive));

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) return;
  if (event.key.toLowerCase() === "w") setWire(!wireActive);
  if (event.key.toLowerCase() === "r") setAutoRotate(!autoRotate);
});

setAutoRotate(autoRotate);

// ---------------------------------------------------------------------------
// animation loop
// ---------------------------------------------------------------------------

const clock = new THREE.Clock();
let firstFrame = true;
let interacting = false;

controls.addEventListener("start", () => { interacting = true; });
controls.addEventListener("end", () => { interacting = false; });

function animate() {
  const delta = Math.min(clock.getDelta(), 0.05);
  if (autoRotate && !interacting) {
    rotation += delta * 0.105;
    turntable.rotation.y = rotation;
  }
  controls.update();
  const t = clock.elapsedTime;
  if (pollenMaterial) {
    pollenMaterial.uniforms.uTime.value = t;
  }
  if (glowHalo?.material) {
    glowHalo.material.opacity = 0.82 + 0.08 * Math.sin(t * 0.42);
  }
  for (let i = 0; i < petalGlowMaterials.length; i += 1) {
    petalGlowMaterials[i].emissiveIntensity = 0.052 + 0.018 * Math.sin(t * 0.65 + i * 0.7);
  }
  // The wire model shares the same turntable transform but keeps a separate
  // root; mirror rotation so toggling is seamless.
  if (wireRoot && beautyRoot) wireRoot.rotation.copy(beautyRoot.rotation);
  renderer.render(scene, camera);
  if (firstFrame) {
    firstFrame = false;
    loader?.classList.add("is-done");
    document.body.classList.add("ready");
  }
  requestAnimationFrame(animate);
}

async function boot() {
  loader3d.manager.onError = (url) => console.error("[rapeseed] failed to load", url);
  try {
    await loadBeauty();
    applyView();
    if (wireActive) {
      setWire(true, false);
    }
    requestAnimationFrame(animate);
    window.__rapeseedReady = true;
    console.info("[rapeseed] ready", {
      objects: 235,
      triangles: 323828,
      trianglesLOD: 111748,
      camera: camera.position.toArray(),
    });
  } catch (error) {
    showError(error);
  }
}

window.__rapeseed = {
  scene,
  camera,
  controls,
  renderer,
  turntable,
  setWire,
  setAutoRotate,
  setRotationDegrees: (degrees) => {
    rotation = THREE.MathUtils.degToRad(degrees);
    turntable.rotation.y = rotation;
    if (wireRoot && beautyRoot) wireRoot.rotation.copy(beautyRoot.rotation);
  },
  get ready() { return window.__rapeseedReady === true; },
};

boot();
