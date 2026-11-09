import * as THREE from "three";

function woodTexture() {
  const c = document.createElement("canvas");
  c.width = 512;
  c.height = 512;
  const g = c.getContext("2d");
  g.fillStyle = "#8a6242";
  g.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 6; i++) {
    const x = i * 86;
    g.fillStyle = i % 2 === 0 ? "#a07850" : "#8d6844";
    g.fillRect(x, 0, 84, 512);
    g.strokeStyle = "rgba(40, 22, 10, 0.35)";
    g.beginPath();
    g.moveTo(x + 84, 0);
    g.lineTo(x + 84, 512);
    g.stroke();
    for (let k = 0; k < 10; k++) {
      const y = (k * 61 + i * 17) % 512;
      g.strokeStyle = "rgba(255, 230, 200, 0.06)";
      g.beginPath();
      g.moveTo(x + 6, y);
      g.lineTo(x + 78, y + 4);
      g.stroke();
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.anisotropy = 8;
  tex.repeat.set(4, 3);
  return tex;
}

function addSpot(stage, color, intensity, position, target, castShadow) {
  const spot = new THREE.SpotLight(color, intensity, 14, Math.PI / 9, 0.45, 1.1);
  spot.position.set(position[0], position[1], position[2]);
  spot.target.position.set(target[0], target[1], target[2]);
  spot.castShadow = castShadow;
  if (castShadow) {
    spot.shadow.mapSize.set(1024, 1024);
    spot.shadow.camera.near = 0.4;
    spot.shadow.camera.far = 18;
    spot.shadow.bias = -0.0004;
  }
  stage.add(spot);
  stage.add(spot.target);
  return spot;
}

/**
 * Playable deck is 8 m wide and 6 m deep.
 * Inner faces: x = ±4, y = -1.2 (front) and y = 4.8 (back).
 */
export function createStage() {
  const stage = new THREE.Group();
  stage.name = "stage";

  const deck = new THREE.Mesh(
    new THREE.BoxGeometry(7.9, 5.9, 0.18),
    new THREE.MeshStandardMaterial({
      map: woodTexture(),
      color: 0xfff4e8,
      roughness: 0.62,
      metalness: 0.02,
    }),
  );
  deck.position.set(0, 1.8, -0.09);
  deck.receiveShadow = true;
  stage.add(deck);

  const skirt = new THREE.MeshStandardMaterial({ color: 0x121318, roughness: 0.7, metalness: 0.15 });
  const apron = new THREE.Mesh(new THREE.BoxGeometry(8.15, 6.15, 0.16), skirt);
  apron.position.set(0, 1.8, -0.2);
  stage.add(apron);

  const flat = new THREE.MeshStandardMaterial({ color: 0x16181e, roughness: 0.86, metalness: 0.08 });
  const walls = [
    { size: [0.16, 6.2, 2.4], pos: [-4.12, 1.8, 1.2] },
    { size: [0.16, 6.2, 2.4], pos: [4.12, 1.8, 1.2] },
    { size: [8.4, 0.16, 2.4], pos: [0, 4.96, 1.2] },
  ];
  for (const w of walls) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...w.size), flat);
    mesh.position.set(...w.pos);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    stage.add(mesh);
  }

  const trimMat = new THREE.MeshStandardMaterial({
    color: 0xffd2a4,
    emissive: 0xffa35a,
    emissiveIntensity: 0.65,
    roughness: 0.4,
  });
  const trims = [
    { size: [0.04, 6.05, 0.025], pos: [-4.0, 1.8, 2.38] },
    { size: [0.04, 6.05, 0.025], pos: [4.0, 1.8, 2.38] },
    { size: [7.95, 0.04, 0.025], pos: [0, 4.84, 2.38] },
  ];
  for (const t of trims) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...t.size), trimMat);
    mesh.position.set(...t.pos);
    stage.add(mesh);
  }

  const cyc = document.createElement("canvas");
  cyc.width = 256;
  cyc.height = 256;
  const cg = cyc.getContext("2d");
  const grad = cg.createLinearGradient(0, 0, 0, 256);
  grad.addColorStop(0, "#1c2438");
  grad.addColorStop(0.55, "#121722");
  grad.addColorStop(1, "#0c0e14");
  cg.fillStyle = grad;
  cg.fillRect(0, 0, 256, 256);
  const cycTex = new THREE.CanvasTexture(cyc);
  cycTex.colorSpace = THREE.SRGBColorSpace;
  const backdrop = new THREE.Mesh(
    new THREE.PlaneGeometry(7.7, 2.2),
    new THREE.MeshStandardMaterial({ map: cycTex, roughness: 1, metalness: 0 }),
  );
  backdrop.rotation.x = Math.PI / 2;
  backdrop.position.set(0, 4.82, 1.15);
  backdrop.receiveShadow = true;
  stage.add(backdrop);

  const lipMat = new THREE.MeshStandardMaterial({ color: 0xd5dbe3, roughness: 0.32, metalness: 0.65 });
  const lip = new THREE.Mesh(new THREE.BoxGeometry(8.05, 0.06, 0.04), lipMat);
  lip.position.set(0, -1.26, 0.04);
  stage.add(lip);

  const house = new THREE.Mesh(
    new THREE.PlaneGeometry(30, 30),
    new THREE.MeshStandardMaterial({ color: 0x0b0c10, roughness: 1 }),
  );
  house.position.set(0, 1.8, -0.36);
  house.receiveShadow = true;
  stage.add(house);

  const key = addSpot(stage, 0xfff4e6, 42, [0.2, -2.4, 5.4], [0, 0.8, 1.0], true);
  addSpot(stage, 0xd5e4ff, 14, [-2.4, 0.2, 4.2], [0, 1.2, 1.0], false);
  addSpot(stage, 0xffc9a0, 12, [2.6, 0.4, 4.0], [0, 1.2, 1.0], false);
  addSpot(stage, 0xc9d6ee, 10, [0, 2.4, 3.4], [0, 4.6, 1.3], false);

  return { stage, key };
}
