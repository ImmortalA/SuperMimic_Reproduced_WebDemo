# SuperMimic playground

Browser demo of a SuperMimic tracking policy on the Unitree G1. Physics is [MuJoCo WASM](https://github.com/google-deepmind/mujoco/tree/main/wasm) (`@mujoco/mujoco`). The policy is the same motion encoder + decoder as training, run in JavaScript. Buttons (or keys 1 / 2 / 3) switch the reference clip: stand, walk, run.

This folder is the root of its own repository. The browser app does not import the training package. The SuperMimic repo gitignores `playground/public`, `node_modules`, and `dist`, so copy this folder out (or `git init` here) before publishing. Include `public/` in the demo repo: that is the policy, the G1 meshes, and the clips.

## Export assets

From the SuperMimic repo, with the training environment:

```bash
env -u PYTHONPATH uv run python playground/export_assets.py \
  --ckpt logs/stage_a_ft/ckpt_latest.pt \
  --npz-dir data/npz/stage_b
```

That writes `playground/public/`:

- `policy.bin` — network weights
- `meta.json` — joint order, action scale, default pose, clip index
- `scene/g1.xml` and `scene/assets/` — G1 MJCF plus meshes, with the same position-actuator gains as training
- `clips/*.bin` — precomputed 68-d goal features

## Run locally

```bash
cd playground
npm install
npm run dev
```

Open the URL Vite prints (usually http://localhost:5173).

## Publish

```bash
cd playground
npm run build
```

`dist/` is a static site. For GitHub Pages, push this folder as the repository root and deploy `dist/`, or commit `dist` on a `gh-pages` branch.

```bash
cd playground
git init
gh repo create supermimic-playground --public --source=. --remote=origin
# after npm run build, publish dist with your usual Pages setup
```

The G1 meshes come from the Unitree MJCF shipped with mjlab. Keep that license with the assets if you redistribute them.

## What the demo does

Each control tick (50 Hz) builds the same observation the trainer uses: 10-step history of joint position error, joint velocity, base angular velocity, projected gravity, and the previous action, plus 10 future reference frames. The network outputs a 29-d joint target. MuJoCo integrates four 5 ms physics steps per control tick.
