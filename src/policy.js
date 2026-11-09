/** SiLU MLP matching SuperMimic's MotionEncoder + Decoder. */

function silu(x) {
  return x / (1 + Math.exp(-x));
}

function readLayers(view, offset) {
  const n = view.getUint32(offset, true);
  offset += 4;
  const layers = [];
  for (let i = 0; i < n; i++) {
    const inn = view.getUint32(offset, true);
    const out = view.getUint32(offset + 4, true);
    offset += 8;
    const w = new Float32Array(view.buffer, view.byteOffset + offset, out * inn);
    offset += out * inn * 4;
    const b = new Float32Array(view.buffer, view.byteOffset + offset, out);
    offset += out * 4;
    layers.push({ w: w.slice(), b: b.slice(), inn, out });
  }
  return { layers, offset };
}

function apply(layers, x, activateLast) {
  let h = x;
  for (let li = 0; li < layers.length; li++) {
    const { w, b, inn, out } = layers[li];
    const y = new Float32Array(out);
    for (let r = 0; r < out; r++) {
      let s = b[r];
      const row = r * inn;
      for (let c = 0; c < inn; c++) s += w[row + c] * h[c];
      const last = li === layers.length - 1;
      y[r] = !last || activateLast ? silu(s) : s;
    }
    h = y;
  }
  return h;
}

export function policyFromBuffer(buf) {
  const view = new DataView(buf);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== "POL1") throw new Error(`bad policy magic ${magic}`);
  let offset = 4;
  const enc = readLayers(view, offset);
  offset = enc.offset;
  const dec = readLayers(view, offset);
  offset = dec.offset;
  const obsScale = new Float32Array(buf, offset, 930).slice();
  offset += 930 * 4;
  const goalScale = new Float32Array(buf, offset, 680).slice();
  return {
    forward(obs, goal) {
      const g = new Float32Array(680);
      const o = new Float32Array(930);
      for (let i = 0; i < 680; i++) g[i] = goal[i] * goalScale[i];
      for (let i = 0; i < 930; i++) o[i] = obs[i] * obsScale[i];
      const z = apply(enc.layers, g, false);
      for (let i = 0; i < z.length; i++) z[i] = Math.tanh(z[i]);
      const cat = new Float32Array(z.length + o.length);
      cat.set(z, 0);
      cat.set(o, z.length);
      return apply(dec.layers, cat, false);
    },
  };
}

export async function loadPolicy(url) {
  const buf = await fetch(url).then((r) => {
    if (!r.ok) throw new Error(`policy fetch ${r.status}`);
    return r.arrayBuffer();
  });
  return policyFromBuffer(buf);
}
