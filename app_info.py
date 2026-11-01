"""Name, version and branding, in one place: the window title, the splash screen and
build_exe.py all read from here. No heavy imports -- the launcher imports this first."""

APP_NAME = "Lite Classify"
APP_VERSION = "1.2"
APP_TAGLINE = "Object Based Image Classification"
APP_AUTHOR = "LeoBar"
APP_ID = "LeoBar.LiteClassify"          # Windows AppUserModelID: groups the taskbar icon
ICON_FILE = "logo_liteclassify.ico"     # bundled at the root of the build (see build_exe.py)
BRAND_ORANGE = "#f65244"             # from the icon's background
BRAND_ORANGE_LIGHT = "#f86c41"

WINDOW_TITLE = f"{APP_NAME} v{APP_VERSION} - {APP_TAGLINE}"
