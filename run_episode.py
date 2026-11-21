"""Run one Dream-1 task in the sim and record it as an episode, judged against the simulator's ground truth.

Usage:
    python run_episode.py "put apple into the bowl" [--seconds 90] [--view] [--no-early-stop] [--model beta]

--view opens the MuJoCo viewer window; without it the run is headless (faster). The episode stops early once the
judge sees the task done and settled, unless --no-early-stop.
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore")
import dreamfix  # noqa: E402  (sim on sys.path, connection and render fixes)
import dream  # noqa: E402
import dream_control  # noqa: E402

from recorder import EpisodeRecorder, RecordingEnv  # noqa: E402


def run(task, seconds=90, view=False, early_stop=True, model=dream_control.BETA_MODEL, actions=48, quiet=True,
        seed=None, batch_id=None):
    """Run, record and judge one Dream-1 episode.
    @param seed When set, the objects' start poses are jittered with this seed (RecordingEnv.jitter).
    @param batch_id Batch this episode belongs to, stored in meta.json.
    @return (episode folder, outcome dict).
    """
    env = RecordingEnv()
    offsets = env.jitter(seed) if seed is not None else None
    rec = EpisodeRecorder(env, task, model=model, source="dream-1", stop_on_success=early_stop,
                          extra={"actions_per_request": actions, "max_seconds": seconds, "seed": seed,
                                 "jitter": offsets, "batch_id": batch_id})
    env.recorder = rec

    fixed_capture = dream_control.capture

    def capture_and_record(env_, windows, keep=True):
        images = fixed_capture(env_, windows, keep)
        if keep:
            rec.on_frame(images)
        return images

    timed_get_action = dream.DreamClient.get_action

    def get_action_timed(self, *args, **kwargs):
        started = time.monotonic()
        reply = timed_get_action(self, *args, **kwargs)
        rec.rtts.append(time.monotonic() - started)
        return reply

    dream_control.capture = capture_and_record
    dream.DreamClient.get_action = get_action_timed
    retries_before = dreamfix.STATS["retries"]
    error = None
    argv = ["--seconds", str(seconds), "--task", task, "--model", model, "--actions", str(actions)]
    argv += ["--quiet"] if quiet else []
    argv += [] if view else ["--headless"]
    try:
        dream_control.main(argv, shared_env=env, stop_event=rec.stop_event)
    except KeyboardInterrupt as e:
        error = e
    except Exception as e:  # noqa: BLE001  (record the failure as part of the episode)
        error = e
    finally:
        dream_control.capture = fixed_capture
        dream.DreamClient.get_action = timed_get_action
    rec.extra["connection_retries"] = dreamfix.STATS["retries"] - retries_before
    outcome = rec.close(error)
    env.close()
    return rec.dir, outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", nargs="?", default="put apple into the bowl")
    parser.add_argument("--seconds", type=float, default=90)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--model", type=dream_control.resolve_model, default=dream_control.BETA_MODEL)
    parser.add_argument("--actions", type=int, default=48)
    parser.add_argument("--seed", type=int, help="Jitter the objects' start poses with this seed")
    parser.add_argument("--batch-id")
    args = parser.parse_args()
    folder, outcome = run(args.task, args.seconds, args.view, not args.no_early_stop, args.model, args.actions,
                          seed=args.seed, batch_id=args.batch_id)
    m = outcome["metrics"]
    print(f"\nEpisode {folder.name}: {outcome['label'].upper()} ({outcome['reason']})")
    if outcome["failure_time_s"] is not None:
        print(f"  failure at {outcome['failure_time_s']} s")
    if outcome["success_time_s"] is not None:
        print(f"  placed at {outcome['success_time_s']} s")
    print(f"  {m.get('object')} moved {m.get('object_moved_m')} m, ends {m.get('final_distance_to_target_m')} m from the {m.get('target')}")
    for w in outcome["warnings"]:
        print(f"  warning: {w}")
    print(f"  saved to {folder}")
    print(f"EPISODE_DIR={folder}", flush=True)  # read by batch.py


if __name__ == "__main__":
    main()
