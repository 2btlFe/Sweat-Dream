"""Screenshots of the app's pages for the README, the usage guide and the project page (docs/assets/screens/).

    ~/dream-env/bin/python tools/screenshots.py

It starts the app on a spare port with the key from ~/.dreamlabs/credentials (as signing in does; the key is never
typed or printed), opens each page in Playwright's headless Chromium at 1280 x 800, and saves PNGs. Before each shot it
hides what should not be published: the key's last characters and the Dream Labs account name. Needs Playwright:
    ~/dream-env/bin/pip install playwright && ~/dream-env/bin/python -m playwright install chromium
"""

from __future__ import annotations

import os
import re
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import app  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = ROOT / "docs" / "assets" / "screens"
PORT = 8017
SIZE = {"width": 1280, "height": 800}
# A run with a recording to show in the run detail (a success in a hand-set scene); the newest success otherwise.
DETAIL_RUN = os.environ.get("SHOT_RUN", "20261122-111638-9099")

FIX = r"""s => s.replace(/byeongcheol\.lee/gi, "your-account").replace(/…[A-Za-z0-9_-]{4}\b/g, "…abcd")
                .replace(/dream-1-ft:[0-9a-f]{8}-/g, "dream-1-ft:xxxxxxxx-")"""
# Text the page writes later (the key hint arrives after the page loads) is masked as it appears, not only before a shot.
KEEP_MASKED = """(() => {
  const fix = %s;
  const mask = root => {
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n; (n = w.nextNode());) { const v = fix(n.nodeValue); if (v !== n.nodeValue) n.nodeValue = v; }
  };
  document.addEventListener("DOMContentLoaded", () => {
    mask(document.body);
    new MutationObserver(ms => { for (const m of ms) mask(m.target.nodeType === 3 ? m.target.parentNode : m.target); })
      .observe(document.body, { subtree: true, childList: true, characterData: true });
  });
})()""" % FIX

MASK = """() => {
  const fix = s => s.replace(/byeongcheol\\.lee/gi, "your-account").replace(/…[A-Za-z0-9_-]{4}\\b/g, "…abcd")
                    .replace(/dream-1-ft:[0-9a-f]{8}-/g, "dream-1-ft:xxxxxxxx-");
  const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let n; (n = w.nextNode());) n.nodeValue = fix(n.nodeValue);
  for (const i of document.querySelectorAll("input")) i.value = fix(i.value);
  for (const o of document.querySelectorAll("option")) o.textContent = fix(o.textContent);
}"""


def key_from_credentials():
    text = Path("~/.dreamlabs/credentials").expanduser().read_text()
    return re.search(r"^DREAM_API_KEY=(\S+)", text, re.M).group(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    app.PAGE_KEY["value"] = key_from_credentials()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), app.Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{PORT}"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=SIZE, device_scale_factor=1)
        page.add_init_script(KEEP_MASKED)

        def shot(name, settle=1500, clip=None):
            page.mouse.move(2, 2)  # no hover highlight on a button under the cursor
            page.wait_for_timeout(settle)
            page.evaluate(MASK)
            page.wait_for_timeout(200)
            path = OUT / f"{name}.png"
            page.screenshot(path=str(path), clip=clip)
            print(f"saved {path.relative_to(ROOT)}", flush=True)

        # 1. Start: the sign-in box.
        page.goto(base + "/")
        page.wait_for_timeout(800)
        page.evaluate("() => { if (typeof show === 'function') show(null); document.activeElement && document.activeElement.blur(); }")
        shot("01-start", 600)

        # 2. Sections.
        page.goto(base + "/sections")
        shot("02-sections")

        # 3. Collect: run setup and the camera view.
        page.goto(base + "/collect")
        page.wait_for_selector("#runs tr[data-id]", timeout=20000)
        shot("03-collect", 3500)

        # 5. Collect: a run's detail, with its video (taken here, before the batch view, but numbered after it).
        run = DETAIL_RUN if page.locator(f'#runs tr[data-id="{DETAIL_RUN}"]').count() else None
        row = page.locator(f'#runs tr[data-id="{run}"]') if run else page.locator("#runs tr[data-id]").first
        row.scroll_into_view_if_needed()
        row.click()
        page.wait_for_timeout(1500)
        page.evaluate("() => { const d = document.getElementById('detail'); window.scrollTo(0, d.getBoundingClientRect().top + scrollY - 20); }")
        page.wait_for_timeout(500)
        box = page.locator("#detail").bounding_box()  # the run's detail only (the left column is empty there)
        shot("05-collect-run", 2500, clip={"x": box["x"] - 12, "y": max(0, box["y"] - 12), "width": box["width"] + 24,
                                           "height": min(780, SIZE["height"] - max(0, box["y"] - 12))})

        # 4. Collect: a batch, with its settings.
        # The batch settings are in the setup's third step, Run settings.
        page.evaluate("() => window.scrollTo(0, 0)")
        page.click("#stageNav button:nth-child(3)")
        page.click("#modeBatch")
        page.evaluate("() => { const w = document.getElementById('wizard'); window.scrollTo(0, w.getBoundingClientRect().top + scrollY - 24); }")
        shot("04-collect-batch", 1200)

        # 6-7. Train: the saved datasets, then the training form and jobs.
        page.goto(base + "/train")
        page.wait_for_function("() => !document.getElementById('localRows').textContent.includes('Loading')", timeout=60000)
        shot("06-train", 800)
        page.evaluate("() => { const t = document.getElementById('trainTitle'); window.scrollTo(0, t.getBoundingClientRect().top + scrollY - 70); }")
        shot("07-train-model", 800)

        # 8. Test.
        page.goto(base + "/test")
        shot("08-test", 3500)

        browser.close()
    server.shutdown()


if __name__ == "__main__":
    main()
