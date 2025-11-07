import os
THIS_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def find_sim_per_platform():
    import platform
    import os

    system = platform.system()
    if system == "Linux":
        sim_path = os.path.join(THIS_FILE_PATH, "DonkeySimLinux/donkey_sim.x86_64")
    elif system == "Windows":
        sim_path = os.path.join(THIS_FILE_PATH, "DonkeySimWindows/donkey_sim.exe")
    elif system == "Darwin":
        sim_path = os.path.join(THIS_FILE_PATH, "DonkeySimMac/donkey_sim.app/Contents/MacOS/donkey_sim")
    else:
        raise Exception(f"Unsupported platform: {system}")
    return sim_path