"""Service identity surfaced by the /health endpoint.

VERSION is overwritten at deploy time via the APP_VERSION env var (see
__main__.py) so the binary running in prod can report the released semver
without rebuilding the image.
"""

import os

SERVICE = "Prog Strength Agent"
VERSION = os.environ.get("APP_VERSION", "dev")
