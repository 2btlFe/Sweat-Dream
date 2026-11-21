"""Run one Dream-1 task in the MuJoCo viewer window, with the fixes in dreamfix.

Usage: python live_view.py [task] [seconds]
"""

import sys
import warnings

warnings.filterwarnings("ignore")
import dreamfix  # noqa: E402  (must come before dream_control is used)
import dream_control  # noqa: E402

task = sys.argv[1] if len(sys.argv) > 1 else "put apple into the bowl"
seconds = sys.argv[2] if len(sys.argv) > 2 else "90"
try:
    dream_control.main(["--quiet", "--seconds", seconds, "--task", task])
except KeyboardInterrupt:
    print("Stopped.")
print(f"Done. Connection retries: {dreamfix.STATS['retries']}")
input("Press Enter to close.")
