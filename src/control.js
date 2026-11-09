/** 50 Hz step. WASD commands a gait the same way the interactive driver does:
 *  mocap joints from a clip, root velocity from the stick. */

const HIST = 10;
const OBS = 93;
const TERMS = [29, 29, 3, 3, 29];
const DT = 0.02;

export function quatApplyInverse(q, v) {
  const x = q[1], y = q[2], z = q[3], w = q[0];
  const vx = v[0], vy = v[1], vz = v[2];
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx - w * tx + (y * tz - z * ty),
    vy - w * ty + (z * tx - x * tz),
    vz - w * tz + (x * ty - y * tx),
  ];
}

export function flattenHistory(hist) {
  const out = new Float32Array(HIST * OBS);
  let dst = 0;
  let start = 0;
  for (const d of TERMS) {
    for (let t = 0; t < HIST; t++) {
      out.set(hist[t].subarray(start, start + d), dst);
      dst += d;
    }
    start += d;
  }
  return out;
}

function approach(cur, target, maxDelta) {
  const d = target - cur;
  if (Math.abs(d) <= maxDelta) return target;
  return cur + Math.sign(d) * maxDelta;
}

function meanForward(clip) {
  let s = 0;
  for (let i = 0; i < clip.frames; i++) s += Math.abs(clip.feat[i * 68 + 64]);
  return s / Math.max(clip.frames, 1);
}

function sample(clip, frame) {
  const T = clip.frames;
  let f = frame % T;
  if (f < 0) f += T;
  const i0 = Math.floor(f);
  const a = f - i0;
  const i1 = (i0 + 1) % T;
  const out = new Float32Array(68);
  const A = i0 * 68;
  const B = i1 * 68;
  for (let i = 0; i < 68; i++) out[i] = clip.feat[A + i] * (1 - a) + clip.feat[B + i] * a;
  return out;
}

export function createController({ policy, meta, clips }) {
  const nJ = meta.joints.length;
  const scale = Float32Array.from(meta.action_scale);
  const q0 = Float32Array.from(meta.default_q);
  const meanWalk = Math.max(meanForward(clips.walk), 0.2);
  const meanRun = Math.max(meanForward(clips.run), 0.4);
  const keys = new Set();
  let latched = "stand";
  let vx = 0;
  let yawRate = 0;
  let phase = 0;
  let lastAction = new Float32Array(nJ);
  let hist = [];

  function command() {
    const w = keys.has("w") || keys.has("arrowup");
    const s = keys.has("s") || keys.has("arrowdown");
    const a = keys.has("a") || keys.has("arrowleft");
    const d = keys.has("d") || keys.has("arrowright");
    const runHeld = keys.has("shift") || latched === "run";
    if (w || s || a || d) {
      const speed = runHeld ? 1.05 : 0.55;
      return {
        tv: ((w ? 1 : 0) - (s ? 1 : 0)) * speed,
        ty: ((a ? 1 : 0) - (d ? 1 : 0)) * 1.0,
        run: runHeld,
      };
    }
    if (latched === "walk") return { tv: 0.55, ty: 0, run: false };
    if (latched === "run") return { tv: 1.05, ty: 0, run: true };
    return { tv: 0, ty: 0, run: false };
  }

  const STAND_Z = 0.784;

  function standGoal() {
    const g = new Float32Array(680);
    for (let k = 0; k < 10; k++) {
      const o = k * 68;
      g[o + 60] = -1;
      g[o + 67] = STAND_Z;
    }
    return g;
  }

  function goalAt(clip, rate, dir) {
    const g = new Float32Array(680);
    const speedScale = dir * rate;
    for (let k = 0; k < 10; k++) {
      const feat = sample(clip, phase + speedScale * k);
      if (Math.abs(speedScale - 1) > 1e-3) {
        for (let i = 29; i < 58; i++) feat[i] *= speedScale;
      }
      feat[58] = 0;
      feat[59] = 0;
      feat[60] = -1;
      feat[61] = 0;
      feat[62] = 0;
      feat[63] = yawRate;
      feat[64] = vx;
      feat[65] = 0;
      feat[66] = 0;
      g.set(feat, k * 68);
    }
    return g;
  }

  return {
    get mode() {
      if (latched) return latched;
      if (Math.abs(vx) > 0.75) return "run";
      if (Math.abs(vx) > 0.05 || Math.abs(yawRate) > 0.05) return "walk";
      return "stand";
    },
    get frame() { return Math.floor(Math.abs(phase)); },
    get vx() { return vx; },
    get yawRate() { return yawRate; },
    setKey(name, down) {
      if (down) keys.add(name);
      else keys.delete(name);
    },
    clearKeys() { keys.clear(); },
    setMode(name) {
      latched = name;
      keys.clear();
    },
    reset(data) {
      data.qpos[0] = 0;
      data.qpos[1] = 0;
      data.qpos[2] = STAND_Z;
      data.qpos[3] = 1;
      data.qpos[4] = 0;
      data.qpos[5] = 0;
      data.qpos[6] = 0;
      for (let i = 0; i < nJ; i++) data.qpos[7 + i] = q0[i];
      data.qvel.fill(0);
      phase = 0;
      vx = 0;
      yawRate = 0;
      lastAction = new Float32Array(nJ);
    },
    seed(data) {
      const obs = observe(data.qpos, data.qvel);
      hist = Array.from({ length: HIST }, () => obs.slice());
    },
    step(mujoco, model, data) {
      const { tv, ty, run } = command();
      const stopping = Math.abs(tv) < 1e-6 && Math.abs(ty) < 1e-6;
      vx = approach(vx, tv, (stopping ? 3 : 2) * DT);
      yawRate = approach(yawRate, ty, 3 * DT);
      const moving = Math.abs(vx) > 0.05 || Math.abs(yawRate) > 0.05;
      const clip = !moving ? clips.stand : (run || Math.abs(vx) > 0.75 ? clips.run : clips.walk);
      const mean = clip === clips.run ? meanRun : meanWalk;
      let rate = 0.25;
      let dir = 1;
      if (moving) {
        dir = vx < -0.05 ? -1 : 1;
        rate = Math.abs(vx) > 0.05 ? Math.max(0.5, Math.min(1.5, Math.abs(vx) / mean)) : 1;
      }
      const obs = observe(data.qpos, data.qvel);
      hist.push(obs);
      if (hist.length > HIST) hist.shift();
      const goal = moving ? goalAt(clip, rate, dir) : standGoal();
      const action = policy.forward(flattenHistory(hist), goal);
      lastAction = action;
      for (let i = 0; i < nJ; i++) data.ctrl[i] = q0[i] + action[i] * scale[i];
      for (let s = 0; s < meta.decimation; s++) mujoco.mj_step(model, data);
      phase += dir * rate;
    },
  };

  function observe(qpos, qvel) {
    const o = new Float32Array(OBS);
    let k = 0;
    for (let i = 0; i < nJ; i++) o[k++] = qpos[7 + i] - q0[i];
    for (let i = 0; i < nJ; i++) o[k++] = qvel[6 + i];
    o[k++] = qvel[3];
    o[k++] = qvel[4];
    o[k++] = qvel[5];
    const g = quatApplyInverse([qpos[3], qpos[4], qpos[5], qpos[6]], [0, 0, -1]);
    o[k++] = g[0];
    o[k++] = g[1];
    o[k++] = g[2];
    for (let i = 0; i < nJ; i++) o[k++] = lastAction[i];
    return o;
  }
}
