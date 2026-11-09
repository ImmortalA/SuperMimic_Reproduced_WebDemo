import "./style.css";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import loadMujoco from "@mujoco/mujoco";
import { loadPolicy } from "./policy.js";
import { createController } from "./control.js";
import { createStage } from "./stage.js";

const BASE = import.meta.env.BASE_URL;

const statusEl = document.getElementById("status");
const setStatus = (t) => { statusEl.textContent = t; };

function addVisualMeshes(mujoco, model, scene) {
  const MESH = mujoco.mjtGeom.mjGEOM_MESH.value;
  const meshes = [];
  const verts = model.mesh_vert;
  const faces = model.mesh_face;
  const rgba = model.geom_rgba;
  for (let g = 0; g < model.ngeom; g++) {
    if (model.geom_type[g] !== MESH) continue;
    const mid = model.geom_dataid[g];
    const vadr = model.mesh_vertadr[mid];
    const vnum = model.mesh_vertnum[mid];
    const fadr = model.mesh_faceadr[mid];
    const fnum = model.mesh_facenum[mid];
    const pos = new Float32Array(vnum * 3);
    for (let i = 0; i < vnum * 3; i++) pos[i] = verts[(vadr * 3) + i];
    const idx = new Uint32Array(fnum * 3);
    for (let i = 0; i < fnum * 3; i++) idx[i] = faces[(fadr * 3) + i];
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgba[g * 4], rgba[g * 4 + 1], rgba[g * 4 + 2]),
      roughness: 0.45,
      metalness: 0.12,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.userData.gid = g;
    scene.add(mesh);
    meshes.push(mesh);
  }
  return meshes;
}

async function main() {
  const meta = await fetch(`${BASE}meta.json`).then((r) => r.json());
  setStatus("Loading policy and scene…");
  const [policy, mujoco] = await Promise.all([
    loadPolicy(`${BASE}policy.bin`),
    loadMujoco(),
  ]);

  const sceneDir = `${BASE}scene/`;
  const xml = await fetch(`${sceneDir}g1.xml`).then((r) => r.text());
  const meshNames = [...xml.matchAll(/file="([^"]+)"/g)].map((m) => m[1]);
  mujoco.FS.mkdir("/working");
  mujoco.FS.mkdir("/working/assets");
  mujoco.FS.writeFile("/working/g1.xml", xml);
  await Promise.all(meshNames.map(async (name) => {
    const buf = await fetch(`${sceneDir}assets/${name}`).then((r) => r.arrayBuffer());
    mujoco.FS.writeFile(`/working/assets/${name}`, new Uint8Array(buf));
  }));
  const model = mujoco.MjModel.from_xml_path("/working/g1.xml");
  if (!model) throw new Error("MjModel.from_xml_path failed");
  const data = new mujoco.MjData(model);

  const clips = {};
  for (const c of meta.clips) {
    const raw = await fetch(`${BASE}${c.file}`).then((r) => r.arrayBuffer());
    clips[c.name] = { ...c, feat: new Float32Array(raw) };
  }

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.02;
  document.body.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x090a0e);
  scene.fog = new THREE.Fog(0x090a0e, 12, 26);
  const camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.05, 40);
  camera.up.set(0, 0, 1);
  camera.position.set(0.15, -6.6, 3.15);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 1.1, 0.95);
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.update();
  scene.add(new THREE.HemisphereLight(0xfff6ea, 0x2a241c, 0.42));
  const fill = new THREE.DirectionalLight(0xfff8f0, 0.55);
  fill.position.set(4, -5, 8);
  scene.add(fill);
  const { stage, key } = createStage();
  scene.add(stage);

  const meshes = addVisualMeshes(mujoco, model, scene);
  const ctrl = createController({ policy, meta, clips });
  const pelvis = model.body("pelvis").id;
  const follow = new THREE.Vector3(0, 0, 1);
  const camOffset = new THREE.Vector3().copy(camera.position).sub(controls.target);
  let dragging = false;
  renderer.domElement.addEventListener("pointerdown", () => { dragging = true; });
  window.addEventListener("pointerup", () => {
    if (!dragging) return;
    dragging = false;
    camOffset.copy(camera.position).sub(controls.target);
  });

  function syncMeshes(dt) {
    const xpos = data.geom_xpos;
    const xmat = data.geom_xmat;
    for (const mesh of meshes) {
      const i = mesh.userData.gid;
      mesh.position.set(xpos[i * 3], xpos[i * 3 + 1], xpos[i * 3 + 2]);
      const m = new THREE.Matrix4();
      const r = xmat.subarray(i * 9, i * 9 + 9);
      m.set(r[0], r[1], r[2], 0, r[3], r[4], r[5], 0, r[6], r[7], r[8], 0, 0, 0, 0, 1);
      mesh.quaternion.setFromRotationMatrix(m);
    }
    const p = data.xpos;
    const px = p[pelvis * 3];
    const py = p[pelvis * 3 + 1];
    const k = 1 - Math.exp(-2.2 * Math.max(dt, 0));
    follow.x += (px - follow.x) * k;
    follow.y += (py - follow.y) * k;
    if (!dragging) {
      controls.target.set(follow.x, follow.y, 1);
      camera.position.set(follow.x + camOffset.x, follow.y + camOffset.y, 1 + camOffset.z);
    }
    key.position.set(follow.x + 0.2, follow.y - 3.2, 5.2);
    key.target.position.set(follow.x, follow.y + 0.15, 0.9);
  }

  function reset() {
    ctrl.reset(data);
    mujoco.mj_forward(model, data);
    ctrl.seed(data);
    const p = data.xpos;
    follow.set(p[pelvis * 3], p[pelvis * 3 + 1], 1);
  }

  function setMode(name) {
    ctrl.setMode(name);
    for (const b of document.querySelectorAll("#modes button[data-mode]")) {
      b.classList.toggle("on", b.dataset.mode === name);
    }
    reset();
  }

  document.querySelectorAll("#modes button[data-mode]").forEach((b) => {
    b.onclick = () => setMode(b.dataset.mode);
  });
  document.getElementById("reset").onclick = () => reset();
  const driveKey = (e) => {
    const k = e.key.toLowerCase();
    if (k === "w" || k === "a" || k === "s" || k === "d") return k;
    if (k === "arrowup" || k === "arrowdown" || k === "arrowleft" || k === "arrowright") return k;
    if (e.key === "Shift") return "shift";
    return null;
  };
  window.addEventListener("keydown", (e) => {
    const n = driveKey(e);
    if (n) {
      e.preventDefault();
      ctrl.setKey(n, true);
      return;
    }
    if (e.key === "1") setMode("stand");
    if (e.key === "2") setMode("walk");
    if (e.key === "3") setMode("run");
    if (e.key === "r" || e.key === "R") reset();
  });
  window.addEventListener("keyup", (e) => {
    const n = driveKey(e);
    if (n) {
      e.preventDefault();
      ctrl.setKey(n, false);
    }
  });
  window.addEventListener("blur", () => ctrl.clearKeys());
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  reset();
  setStatus(`checkpoint iter ${meta.iteration} · stand`);
  let acc = 0;
  let last = performance.now();
  const controlDt = meta.physics_dt * meta.decimation;

  function tick(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    acc += dt;
    let steps = 0;
    while (acc >= controlDt && steps < 2) {
      ctrl.step(mujoco, model, data);
      acc -= controlDt;
      steps += 1;
    }
    syncMeshes(dt);
    renderer.render(scene, camera);
    setStatus(`${ctrl.mode}  vx ${ctrl.vx.toFixed(2)}  yaw ${ctrl.yawRate.toFixed(2)}`);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

main().catch((err) => {
  console.error(err);
  setStatus(String(err.message || err));
});
