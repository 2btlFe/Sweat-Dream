"""A walkthrough video of the app (docs/assets/videos/demo.mp4): sign in, sections, a rule-based run in the Collect
page, its result, the dataset popup, the Train page with a price, and the Test page, with a caption and a spoken
narration on each step (macOS `say`, made on this computer).

    ~/dream-env/bin/python tools/demo_video.py

It runs the app on a spare port with the key from ~/.dreamlabs/credentials (as signing in does; the key is never shown),
records Playwright's headless Chromium at 1280 x 800, hides the key's last characters and the account name while it
records, and converts the recording to H.264 MP4 with PyAV. Nothing is charged: the run is rule-based only, no dataset is
saved and no training is started. The room setting the demo changes is put back afterwards. Needs Playwright (see
tools/screenshots.py).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # the code is in src/
os.chdir(ROOT)

import subprocess  # noqa: E402

import av  # noqa: E402
import numpy as np  # noqa: E402
import app  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = ROOT / "docs" / "assets" / "videos" / "demo.mp4"
PORT = 8018
SIZE = {"width": 1280, "height": 800}
VOICE = "Samantha"
RATE = 44100
SKIP_S = 0.6  # the recording's first blank moment, cut from video and narration alike
# Each step: the caption on screen, and what the narrator says.
STEPS = {
    "intro": ("Sweet Dream: diverse data for custom Dream-1 fine-tuning",
              "Sweet Dream collects diverse robot data in simulation, and fine-tunes a custom Dream-1 on it."),
    "signin": ("Sign in with your Dream API key", "First, sign in with your Dream API key."),
    "sections": ("Pick a section: Collect, Train or Test", "Then pick a section: collect, train, or test."),
    "collect": ("1. Collect: set up the scene", "In Collect, you set up the scene for the robot."),
    "room": ("Change the room, light or camera: the preview follows",
             "Change the room, the light, or the camera, and the preview follows."),
    "mode": ("Rule-based only: the script does the task, no credits",
             "Rule-based only lets a script do the whole task, with no credits."),
    "run": ("Run: the episode plays in real time",
            "Press Run. The episode plays in real time, and every step is recorded."),
    "watch": ("The script grasps the apple, carries it and sets it in the bowl",
              "Here the script reaches for the apple, lifts it, carries it over, and sets it down in the bowl. "
              "The camera views are the same ones Dream-1 sees."),
    "result": ("Every run is judged and recorded: here, a success",
               "Every run is judged by success rules. This one succeeded."),
    "dataset": ("Pick the good runs (+) and save them as a dataset",
                "Pick the good runs, and save them as a dataset in the Dream Labs format."),
    "train": ("2. Train: upload a saved dataset to Dream Labs", "In Train, upload a saved dataset to Dream Labs."),
    "price": ("Pick the steps and see the price before training",
              "Pick the training steps, and see the price before you start."),
    "test": ("3. Test: run your fine-tune on its own",
             "In Test, run your fine-tuned model on its own, and compare it with Dream-1."),
    "end": ("Sweet Dream: Diverse Data for Custom Dream-1 Fine-Tuning", "Sweet Dream. Diverse data for custom Dream-1 fine-tuning."),
}

# Runs in every page before its own scripts: hides the key's last characters, the account name, a fine-tune's
# account id and the home folder in local paths (in text and in input values) as they appear, and offers
# window.__caption(text) for the step captions.
INIT = r"""
(() => {
  const fix = s => s.replace(/byeongcheol\.lee/gi, "your-account").replace(/…[A-Za-z0-9_-]{4}\b/g, "…abcd")
                    .replace(/dream-1-ft:[0-9a-f]{8}-/g, "dream-1-ft:xxxxxxxx-")
                    .replace(/\/Users\/[^/\s<"']+/g, "~");  // a home folder shows as ~
  // Values the page puts into inputs (the save popup's folder) never reach the text nodes: fix them as they are set.
  const desc = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
  Object.defineProperty(HTMLInputElement.prototype, "value", {
    configurable: true, enumerable: desc.enumerable,
    get() { return desc.get.call(this); },
    set(v) { desc.set.call(this, typeof v === "string" ? fix(v) : v); },
  });
  const clean = root => {
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n; (n = w.nextNode());) { const v = fix(n.nodeValue); if (v !== n.nodeValue) n.nodeValue = v; }
  };
  const start = () => {
    clean(document.body);
    new MutationObserver(ms => ms.forEach(m => {
      if (m.type === "characterData") { const v = fix(m.target.nodeValue); if (v !== m.target.nodeValue) m.target.nodeValue = v; }
      m.addedNodes.forEach(n => n.nodeType === 3 ? (n.nodeValue = fix(n.nodeValue)) : n.nodeType === 1 && clean(n));
    })).observe(document.body, { childList: true, subtree: true, characterData: true });
  };
  document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", start) : start();
  window.__caption = text => {
    let el = document.getElementById("__caption");
    if (!el) {
      el = document.createElement("div"); el.id = "__caption";
      el.style.cssText = "position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:99999;max-width:86%;" +
        "padding:12px 22px;border-radius:999px;background:rgba(28,28,28,.88);color:#fff;font:500 20px/1.3 Figtree,system-ui,sans-serif;" +
        "box-shadow:0 8px 24px rgba(0,0,0,.25);text-align:center;transition:opacity .25s";
      document.body.appendChild(el);
    }
    el.textContent = text; el.style.opacity = text ? "1" : "0";
  };
})();
"""


def key_from_credentials():
    text = Path("~/.dreamlabs/credentials").expanduser().read_text()
    return re.search(r"^DREAM_API_KEY=(\S+)", text, re.M).group(1)


def speak(text, path):
    """The narration of one step as mono float samples at RATE (macOS `say`; nothing leaves this computer)."""
    subprocess.run(["say", "-v", VOICE, "-o", str(path), text], check=True)
    resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)
    chunks = []
    with av.open(str(path)) as c:
        for frame in c.decode(audio=0):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
    for f in resampler.resample(None):
        chunks.append(f.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32)


def to_mp4(webm, mp4, voice, skip_s=SKIP_S):
    """H.264 MP4 at 25 fps from Playwright's WebM, without its first blank moments, with the narration (a mono float
    track at RATE) as AAC."""
    with av.open(str(webm)) as src, av.open(str(mp4), "w") as dst:
        out = dst.add_stream("libx264", rate=25)
        out.width, out.height, out.pix_fmt = SIZE["width"], SIZE["height"], "yuv420p"
        out.options = {"crf": "23", "preset": "medium"}
        sound = dst.add_stream("aac", rate=RATE, layout="mono")  # both streams before anything is written
        stream = src.streams.video[0]
        start, step, t_next = None, 1 / 25, 0.0
        last = None
        for frame in src.decode(stream):
            t = float(frame.pts * stream.time_base)
            if start is None:
                start = t
            t -= start + skip_s
            if t < 0:
                continue
            img = frame.to_ndarray(format="rgb24")
            while t_next <= t:  # constant frame rate: repeat the last frame across gaps
                f = av.VideoFrame.from_ndarray(last if last is not None else img, format="rgb24")
                for packet in out.encode(f):
                    dst.mux(packet)
                t_next += step
            last = img
        for packet in out.encode():
            dst.mux(packet)
        voice = voice[: int(t_next * RATE)]
        for i in range(0, len(voice), 1024):
            f = av.AudioFrame.from_ndarray(voice[None, i:i + 1024], format="flt", layout="mono")
            f.sample_rate = RATE
            for packet in sound.encode(f):
                dst.mux(packet)
        for packet in sound.encode():
            dst.mux(packet)


def main():
    settings = app.SETTINGS
    backup = settings.read_bytes() if settings.exists() else None
    app.PAGE_KEY["value"] = key_from_credentials()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), app.Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{PORT}"
    work = Path(tempfile.mkdtemp(prefix="sweet-dream-demo-"))
    lines = {k: speak(said, work / f"{k}.aiff") for k, (_, said) in STEPS.items()}
    placed = []  # (seconds into the recording, step)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(viewport=SIZE, record_video_dir=str(work), record_video_size=SIZE)
            context.add_init_script(INIT)
            page = context.new_page()
            t0 = time.monotonic()
            pause = lambda s: page.wait_for_timeout(int(s * 1000))  # noqa: E731

            def say(key, at_least=0.0):
                """Show the step's caption and start its narration; wait while it is spoken (or at_least)."""
                page.evaluate("t => window.__caption(t)", STEPS[key][0])
                placed.append((time.monotonic() - t0, key))
                pause(max(at_least, len(lines[key]) / RATE + 0.5))

            # 1. Sign in.
            page.goto(base + "/")
            page.wait_for_function("() => !document.getElementById('signedIn').hidden || !document.getElementById('loginForm').hidden")
            pause(0.5)
            page.evaluate("() => { if (typeof show === 'function') show(null); }")  # the sign-in box, not the signed-in view
            say("intro")
            page.evaluate("() => window.__caption('Sign in with your Dream API key')")
            placed.append((time.monotonic() - t0, "signin"))
            page.type("#apiKey", "dl_live_demo_key", delay=70)  # a password field: it shows dots; nothing is sent
            pause(max(0.5, len(lines["signin"]) / RATE - 1.0))

            # 2. Sections.
            page.goto(base + "/sections")
            say("sections", 3)
            page.hover(".card.collect")
            pause(0.8)
            page.click(".card.collect")

            # 3. Collect: the scene, a rule-based run, its result, the dataset popup.
            page.wait_for_selector("#runs tr[data-id]", timeout=20000)
            say("collect", 2.5)
            page.select_option("#room", "kitchen")
            say("room", 4.5)
            page.check('input[name="runMode"][value="script"]')
            say("mode", 2.5)
            page.click("#run")
            page.evaluate("() => window.scrollTo({top: 0, behavior: 'smooth'})")  # the live camera view, top right
            say("run")
            page.wait_for_function("() => fetch('/api/status').then(r => r.json()).then(s => s.running)", timeout=20000)
            started = time.monotonic()
            pause(4)
            say("watch")  # while the run goes on
            while time.monotonic() - started < 90:
                if not page.evaluate("() => fetch('/api/status').then(r => r.json()).then(s => s.running)"):
                    break
                pause(1)
            pause(2)
            row = page.locator("#runs tr[data-id]").first
            row.scroll_into_view_if_needed()
            row.click()
            page.evaluate("() => { const d = document.getElementById('detail'); window.scrollTo({top: d.getBoundingClientRect().top + scrollY - 20, behavior: 'smooth'}); }")
            say("result", 4.5)
            page.evaluate("() => { const r = document.getElementById('runsTable'); window.scrollTo({top: r.getBoundingClientRect().top + scrollY - 200, behavior: 'smooth'}); }")
            pause(1)
            page.locator("#runs tr[data-id] .pick").first.click()
            page.click("#saveDataset")
            say("dataset", 4)
            page.click("#saveCancel")  # the demo saves nothing
            pause(0.8)

            # 4. Train.
            page.goto(base + "/train")
            page.wait_for_function("() => !document.getElementById('localRows').textContent.includes('Loading')", timeout=60000)
            say("train", 3.5)
            page.evaluate("() => { const t = document.getElementById('trainTitle'); window.scrollTo({top: t.getBoundingClientRect().top + scrollY - 80, behavior: 'smooth'}); }")
            pause(1)
            page.type("#suffix", "banana-plate-v1", delay=50)
            page.fill("#steps", "500")
            page.dispatch_event("#steps", "input")
            say("price", 4)

            # 5. Test.
            page.goto(base + "/test")
            say("test", 4)
            page.goto(base + "/sections")
            say("end", 3)

            video = page.video.path()
            context.close()
            browser.close()
        total = max(t for t, _ in placed) + 30
        voice = np.zeros(int(total * RATE), dtype=np.float32)
        for t, key in placed:
            i = max(0, int((t - SKIP_S) * RATE))
            clip = lines[key][: len(voice) - i]
            voice[i:i + len(clip)] += clip
        OUT.parent.mkdir(parents=True, exist_ok=True)
        to_mp4(video, OUT, np.clip(voice, -1, 1))
        print(f"saved {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)", flush=True)
    finally:
        server.shutdown()
        if backup is not None:
            settings.write_bytes(backup)  # the room the demo picked is not kept
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
