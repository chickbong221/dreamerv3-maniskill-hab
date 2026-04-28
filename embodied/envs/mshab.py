import functools

import elements
import embodied
import numpy as np


class MSHab(embodied.Env):
  """DreamerV3 adapter for ManiSkill-HAB subtask training environments.

  Instantiate via DreamerV3's task convention:
    --task mshab_tidy_house_pick  (suite=mshab, task=tidy_house_pick)

  The task string must end with one of the four subtasks:
    pick | place | open | close

  Observation space produced:
    depth_head    uint8   (128, 128, 1)  head camera depth, normalised to [0,255]
    depth_hand    uint8   (128, 128, 1)  hand camera depth, normalised to [0,255]
    proprio       float32 (D,)           flattened agent + extra state from MS-HAB
    log/success   float32 ()             1.0 when the current subtask succeeds
    log/fail      float32 ()             1.0 when the episode is failed
    reward        float32 ()
    is_first      bool    ()
    is_last       bool    ()
    is_terminal   bool    ()

  Action space:
    action  float32  (A,)  in [-1, 1]   pd_joint_delta_pos
    reset   bool     ()
  """

  _SUBTASKS = frozenset({'pick', 'place', 'open', 'close'})

  # Depth clipping range (metres).  Objects beyond this map to 255.
  DEPTH_MAX_M = 10.0

  def __init__(
      self,
      task,
      split='train',
      max_episode_steps=200,
      stationary_base=False,
      stationary_torso=False,
      stationary_head=True,
      sim_backend='gpu',
  ):
    # ---- parse "tidy_house_pick" → task_name + subtask ------------------
    parts = task.rsplit('_', 1)
    if len(parts) != 2 or parts[1] not in self._SUBTASKS:
      raise ValueError(
          f"task='{task}' must end with a subtask name "
          f"({sorted(self._SUBTASKS)}), e.g. 'tidy_house_pick'.")
    task_name, subtask = parts

    # ---- build ManiSkill-HAB gymnasium env ------------------------------
    import gymnasium as gym
    from mani_skill import ASSET_DIR
    from mshab.envs.planner import plan_data_from_file
    from mshab.envs.wrappers import FetchActionWrapper, FetchDepthObservationWrapper
    import mshab.envs  # noqa: registers env IDs with gymnasium

    rearrange_dir = (
        ASSET_DIR / 'scene_datasets/replica_cad_dataset/rearrange')
    plan_data = plan_data_from_file(
        rearrange_dir / 'task_plans' / task_name / subtask / split / 'all.json')
    spawn_data_fp = (
        rearrange_dir / 'spawn_data' / task_name / subtask / split / 'spawn_data.pt')

    env = gym.make(
        f'{subtask.capitalize()}SubtaskTrain-v0',
        num_envs=1,
        obs_mode='depth',
        reward_mode='normalized_dense',
        control_mode='pd_joint_delta_pos',
        render_mode='rgb_array',
        shader_dir='minimal',
        robot_uids='fetch',
        sim_backend=sim_backend,
        max_episode_steps=max_episode_steps,
        task_plans=plan_data.plans,
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=spawn_data_fp,
        require_build_configs_repeated_equally_across_envs=False,
    )

    # Depth obs wrapper: raw sensor → {state: (1,D), fetch_head_depth: (1,C,H,W), ...}
    env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)

    # Action wrapper: zero out head (and optionally base/torso) joints
    env = FetchActionWrapper(
        env,
        stationary_base=stationary_base,
        stationary_torso=stationary_torso,
        stationary_head=stationary_head,
    )

    self._env = env
    self._done = True

    # ---- shape discovery via an initial reset ---------------------------
    raw_obs, _ = env.reset()
    state_np = self._to_np(raw_obs['state'])             # (1, D)
    head_np  = self._to_np(raw_obs['fetch_head_depth'])  # (1, C, H, W)

    self._state_dim  = int(state_np.shape[-1])
    self._action_dim = int(env.action_space.shape[-1])
    _, c, h, w = head_np.shape
    self._depth_hwc  = (h, w, c)  # e.g. (128, 128, 1)

    # Reuse the cached reset obs so the first step() doesn't double-reset
    self._cached_obs = raw_obs

  # ------------------------------------------------------------------ spaces

  @functools.cached_property
  def obs_space(self):
    return {
        'depth_head':    elements.Space(np.uint8, self._depth_hwc),
        'depth_hand':    elements.Space(np.uint8, self._depth_hwc),
        'proprio':       elements.Space(np.float32, (self._state_dim,)),
        'log/success':   elements.Space(np.float32),
        'log/fail':      elements.Space(np.float32),
        'reward':        elements.Space(np.float32),
        'is_first':      elements.Space(bool),
        'is_last':       elements.Space(bool),
        'is_terminal':   elements.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    return {
        'action': elements.Space(np.float32, (self._action_dim,), -1.0, 1.0),
        'reset':  elements.Space(bool),
    }

  # ------------------------------------------------------------------ step

  def step(self, action):
    if action['reset'] or self._done:
      if self._cached_obs is not None:
        raw_obs, self._cached_obs = self._cached_obs, None
      else:
        raw_obs, _ = self._env.reset()
      self._done = False
      return self._make_obs(raw_obs, 0.0, success=0.0, fail=0.0, is_first=True)

    # ManiSkill expects a batched action: (1, action_dim)
    act = np.asarray(action['action'], dtype=np.float32)[None]
    raw_obs, rew, term, trunc, info = self._env.step(act)

    terminated = bool(self._to_np(term)[0])
    truncated  = bool(self._to_np(trunc)[0])
    self._done = terminated or truncated

    # MS-HAB info dict contains 'success' and 'fail' boolean tensors (shape (1,))
    success = float(self._to_np(info.get('success', [0.0]))[0])
    fail    = float(self._to_np(info.get('fail',    [0.0]))[0])

    return self._make_obs(
        raw_obs,
        float(self._to_np(rew)[0]),
        success=success,
        fail=fail,
        is_last=self._done,
        is_terminal=terminated,
    )

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  # ------------------------------------------------------------------ helpers

  def _make_obs(self, raw, reward, success=0.0, fail=0.0,
                is_first=False, is_last=False, is_terminal=False):
    state = self._to_np(raw['state'])[0]              # (D,)
    head  = self._to_np(raw['fetch_head_depth'])[0]   # (C, H, W)
    hand  = self._to_np(raw['fetch_hand_depth'])[0]   # (C, H, W)
    return {
        'depth_head':  self._depth_to_uint8(head.transpose(1, 2, 0)),
        'depth_hand':  self._depth_to_uint8(hand.transpose(1, 2, 0)),
        'proprio':     state.astype(np.float32),
        'log/success': np.float32(success),
        'log/fail':    np.float32(fail),
        'reward':      np.float32(reward),
        'is_first':    np.bool_(is_first),
        'is_last':     np.bool_(is_last),
        'is_terminal': np.bool_(is_terminal),
    }

  def _depth_to_uint8(self, depth_hwc):
    """Clip depth (metres) to [0, DEPTH_MAX_M] and quantise to uint8."""
    return (np.clip(depth_hwc, 0.0, self.DEPTH_MAX_M)
            / self.DEPTH_MAX_M * 255.0).astype(np.uint8)

  @staticmethod
  def _to_np(x):
    """Convert a PyTorch tensor (GPU or CPU) or array to float32 numpy."""
    try:
      return x.detach().cpu().float().numpy()
    except AttributeError:
      return np.asarray(x, dtype=np.float32)
