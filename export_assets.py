"""Export a browser demo bundle: G1 scene, policy weights, and tracking clips.

Run from the SuperMimic repo (needs the training environment):

  env -u PYTHONPATH uv run python playground/export_assets.py \\
      --ckpt logs/stage_a_ft/ckpt_latest.pt
"""

from __future__ import annotations

import json
import re
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    ACTUATOR_4010,
    ACTUATOR_5020,
    ACTUATOR_7520_14,
    ACTUATOR_7520_22,
    DAMPING_4010,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    G1_XML,
    HOME_KEYFRAME,
    STIFFNESS_4010,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
)
from supermimic.envs.mdp.motion_library import goal_features
from supermimic.envs.mdp.motion_library import RefState
from supermimic.policy.nets import Policy, PolicyCfg

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"

# (joint-name regexes, stiffness, damping, effort, armature)
_ACT = (
    ((".*_elbow_joint", ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_wrist_roll_joint"),
     STIFFNESS_5020, DAMPING_5020, ACTUATOR_5020.effort_limit, ACTUATOR_5020.reflected_inertia),
    ((".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"),
     STIFFNESS_7520_14, DAMPING_7520_14, ACTUATOR_7520_14.effort_limit, ACTUATOR_7520_14.reflected_inertia),
    ((".*_hip_roll_joint", ".*_knee_joint"),
     STIFFNESS_7520_22, DAMPING_7520_22, ACTUATOR_7520_22.effort_limit, ACTUATOR_7520_22.reflected_inertia),
    ((".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
     STIFFNESS_4010, DAMPING_4010, ACTUATOR_4010.effort_limit, ACTUATOR_4010.reflected_inertia),
    (("waist_pitch_joint", "waist_roll_joint"),
     STIFFNESS_5020 * 2, DAMPING_5020 * 2, ACTUATOR_5020.effort_limit * 2, ACTUATOR_5020.reflected_inertia * 2),
    ((".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
     STIFFNESS_5020 * 2, DAMPING_5020 * 2, ACTUATOR_5020.effort_limit * 2, ACTUATOR_5020.reflected_inertia * 2),
)


def _re(name: str, pat: str) -> bool:
    return re.fullmatch(pat, name) is not None


def _match_act(name: str) -> tuple[float, float, float, float]:
    for globs, kp, kd, effort, arm in _ACT:
        if any(_re(name, g) for g in globs):
            return float(kp), float(kd), float(effort), float(arm)
    raise KeyError(name)


def _match_scale(name: str) -> float:
    for pat, scale in G1_ACTION_SCALE.items():
        if _re(name, pat):
            return float(scale)
    raise KeyError(name)


def _default_q(name: str) -> float:
    for pat, val in HOME_KEYFRAME.joint_pos.items():
        if _re(name, pat):
            return float(val)
    return 0.0


def _scene_xml() -> tuple[str, list[str]]:
    xml = G1_XML.read_text()
    names = re.findall(r'<joint name="([^"]+)"', xml)
    actuators = ["  <actuator>"]
    for name in names:
        kp, kd, effort, arm = _match_act(name)
        xml = xml.replace(
            f'<joint name="{name}"',
            f'<joint name="{name}" armature="{arm:.8g}"',
            1,
        )
        actuators.append(
            f'    <position name="{name}" joint="{name}" kp="{kp:.8g}" kv="{kd:.8g}" '
            f'forcerange="{-effort:.8g} {effort:.8g}"/>'
        )
    actuators.append("  </actuator>")
    floor = """
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.16 0.16 0.18 1" friction="1 0.005 0.0001"/>
    <geom name="wall_left" type="box" pos="-4.12 1.8 1.2" size="0.08 3.1 1.2" rgba="0.09 0.09 0.11 1" friction="1 0.005 0.0001"/>
    <geom name="wall_right" type="box" pos="4.12 1.8 1.2" size="0.08 3.1 1.2" rgba="0.09 0.09 0.11 1" friction="1 0.005 0.0001"/>
    <geom name="wall_front" type="box" pos="0 -1.3 0.52" size="4.12 0.06 0.52" rgba="0.12 0.13 0.16 1" friction="1 0.005 0.0001"/>
    <geom name="wall_back" type="box" pos="0 4.96 1.2" size="4.2 0.08 1.2" rgba="0.09 0.09 0.11 1" friction="1 0.005 0.0001"/>
    <light pos="0 0 8" dir="0 0 -1" directional="true" diffuse="1 1 1"/>
"""
    if "<option" not in xml:
        xml = xml.replace("<compiler", '<option timestep="0.005" gravity="0 0 -9.81"/>\n  <compiler', 1)
    xml = xml.replace("</worldbody>", floor + "  </worldbody>", 1)
    xml = xml.replace("</mujoco>", "\n".join(actuators) + "\n</mujoco>")
    return xml, names


def _write_mlp(buf: bytearray, state: dict, prefix: str) -> None:
    """Dump Linear weights in Sequential order. SiLU modules have no parameters."""
    layers = []
    i = 0
    while True:
        wk = f"{prefix}.{i}.weight"
        if wk not in state:
            if f"{prefix}.{i+1}.weight" in state:
                i += 1
                continue
            break
        w = state[wk].detach().cpu().numpy().astype(np.float32)
        b = state[f"{prefix}.{i}.bias"].detach().cpu().numpy().astype(np.float32)
        layers.append((w, b))
        i += 1
    if not layers:
        raise RuntimeError(f"no linear layers under {prefix}")
    buf.extend(struct.pack("<I", len(layers)))
    for w, b in layers:
        out, inn = w.shape
        buf.extend(struct.pack("<II", inn, out))
        buf.extend(w.tobytes(order="C"))
        buf.extend(b.tobytes(order="C"))


def _clip_features(npz: Path, default_q: np.ndarray, start: int, max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz) as d:
        n_all = int(d["joint_pos"].shape[0])
        i0 = min(max(0, start), max(0, n_all - 2))
        i1 = min(n_all, i0 + max_frames)
        sl = slice(i0, i1)
        names = [str(x) for x in d["body_names"]]
        root = names.index("pelvis")
        ref = RefState(
            root_pos_w=torch.tensor(d["body_pos_w"][sl, root], dtype=torch.float32),
            root_quat_w=torch.tensor(d["body_quat_w"][sl, root], dtype=torch.float32),
            joint_pos=torch.tensor(d["joint_pos"][sl], dtype=torch.float32),
            root_lin_vel_w=torch.tensor(d["body_lin_vel_w"][sl, root], dtype=torch.float32),
            root_ang_vel_w=torch.tensor(d["body_ang_vel_w"][sl, root], dtype=torch.float32),
            joint_vel=torch.tensor(d["joint_vel"][sl], dtype=torch.float32),
        )
        feat = goal_features(ref, torch.tensor(default_q, dtype=torch.float32)).numpy().astype(np.float32)
        qpos = np.concatenate(
            [d["body_pos_w"][sl, root], d["body_quat_w"][sl, root], d["joint_pos"][sl]], axis=1
        ).astype(np.float32)
    return feat, qpos


def main(
    ckpt: Path = Path("logs/stage_a_ft/ckpt_latest.pt"),
    npz_dir: Path = Path("data/npz/stage_b"),
) -> None:
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint {ckpt}")
    PUBLIC.mkdir(parents=True, exist_ok=True)
    scene = PUBLIC / "scene"
    if scene.exists():
        shutil.rmtree(scene)
    (scene / "assets").mkdir(parents=True)
    xml, joint_names = _scene_xml()
    (scene / "g1.xml").write_text(xml)
    assets = G1_XML.parent / "assets"
    nmesh = 0
    for src in assets.iterdir():
        if src.is_file():
            shutil.copy2(src, scene / "assets" / src.name)
            nmesh += 1

    scales = np.array([_match_scale(n) for n in joint_names], dtype=np.float32)
    default_q = np.array([_default_q(n) for n in joint_names], dtype=np.float32)

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    policy = Policy(PolicyCfg())
    policy.load_state_dict(blob["policy"])
    state = policy.state_dict()
    buf = bytearray()
    buf.extend(b"POL1")
    _write_mlp(buf, state, "encoders.motion.mlp.net")
    _write_mlp(buf, state, "decoder.mlp.net")
    obs_scale = state["decoder.scale"].numpy().astype(np.float32)
    goal_scale = state["encoders.motion.scale"].numpy().astype(np.float32)
    assert obs_scale.shape == (930,) and goal_scale.shape == (680,)
    buf.extend(obs_scale.tobytes())
    buf.extend(goal_scale.tobytes())
    (PUBLIC / "policy.bin").write_bytes(buf)

    clips_out = PUBLIC / "clips"
    if clips_out.exists():
        shutil.rmtree(clips_out)
    clips_out.mkdir()
    wanted = [
        ("stand", "standing_00_dance1_subject1.npz", 0, 250),
        ("walk", "walk1_subject1.npz", 1500, 400),
        ("run", "run_run1_subject2.npz", 9475, 400),
    ]
    meta_clips = []
    for name, fname, start, n in wanted:
        src = npz_dir / fname
        if not src.exists():
            # stage_a standing lives in stage_b copies; walk may be the raw lafan name
            alt = list(npz_dir.glob(f"*{fname}"))
            if not alt:
                raise SystemExit(f"missing clip {src}")
            src = alt[0]
        feat, qpos = _clip_features(src, default_q, start, n)
        (clips_out / f"{name}.bin").write_bytes(feat.tobytes())
        qpos0 = qpos[0].copy()
        qpos0[0] = 0.0
        qpos0[1] = 0.0
        meta_clips.append({
            "name": name,
            "frames": int(feat.shape[0]),
            "file": f"clips/{name}.bin",
            "qpos0": qpos0.tolist(),
        })

    meta = {
        "iteration": int(blob.get("iteration", -1)),
        "joints": joint_names,
        "action_scale": scales.tolist(),
        "default_q": default_q.tolist(),
        "physics_dt": 0.005,
        "decimation": 4,
        "history": 10,
        "obs_dim": 93,
        "goal_dim": 680,
        "clips": meta_clips,
        "meshes": nmesh,
    }
    (PUBLIC / "meta.json").write_text(json.dumps(meta))
    print(f"wrote {PUBLIC}  policy={len(buf)/1e6:.1f}MB  meshes={nmesh}  clips={len(meta_clips)}  iter={meta['iteration']}")


if __name__ == "__main__":
    tyro.cli(main)
