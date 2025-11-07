#!/usr/bin/env python3
"""
Sandbox Server Debug Client
Interactive command-line tool for debugging and testing all Sandbox Server APIs.

Usage:
    python debug_client.py [--host HOST] [--port PORT]

Examples:
    # Connect to default server
    python debug_client.py

    # Connect to specific server
    python debug_client.py --host 127.0.0.1 --port 9091

    # Interactive commands:
    > menu get_scene_names
    > menu load_scene warehouse
    > track control 0.5 0.3 0.0
    > track get_protocol_version
    > track reset_car
    > track car_config car01 255 0 0 MyCar
    > track cam_config 120 0 0 640 480 3 JPG
    > track set_position 0 0 0
    > help
    > quit
"""

import json
import time
import sys
import argparse
from io import BytesIO
import base64
from PIL import Image

import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from gym_donkeycar.core.sim_client import SDClient


class DebugClient(SDClient):
    """Debug client that prints all received messages and provides command interface."""
    
    def __init__(self, address, verbose=True):
        super().__init__(*address, poll_socket_sleep_time=0.01)
        self.verbose = verbose
        self.last_telemetry = None
        self.last_image = None
        self.car_loaded = False
        self.scene_names = []
        self.protocol_version = None
        
    def on_msg_recv(self, json_packet):
        """Handle all incoming messages."""
        msg_type = json_packet.get('msg_type', 'unknown')
        
        if msg_type == 'car_loaded':
            self.car_loaded = True
            print(f"\n✓ Car loaded successfully!")
            
        elif msg_type == 'telemetry':
            self.last_telemetry = json_packet.copy()
            # Save image if present
            if 'image' in json_packet:
                try:
                    img_string = json_packet['image']
                    image = Image.open(BytesIO(base64.b64decode(img_string)))
                    self.last_image = image
                    if self.verbose:
                        print(f"\n📷 Telemetry received:")
                        print(f"   Image: {image.size[0]}x{image.size[1]}")
                except Exception as e:
                    print(f"   Warning: Could not decode image: {e}")
                # Remove image from display to reduce clutter
                del json_packet['image']
            
            if self.verbose:
                print(f"\n📊 Telemetry:")
                for key, value in json_packet.items():
                    if key not in ['imageb', 'lidar']:  # Skip large binary data
                        print(f"   {key}: {value}")
                        
        elif msg_type == 'scene_names':
            self.scene_names = json_packet.get('scene_names', [])
            print(f"\n✓ Available scenes ({len(self.scene_names)}):")
            for i, scene in enumerate(self.scene_names, 1):
                print(f"   {i}. {scene}")
                
        elif msg_type == 'scene_selection_ready':
            print(f"\n✓ Scene selection ready")
            
        elif msg_type == 'protocol_version':
            self.protocol_version = json_packet.get('version', 'unknown')
            print(f"\n✓ Protocol version: {self.protocol_version}")
            
        elif msg_type == 'connected':
            print(f"\n✓ Connected to server")
            
        else:
            if self.verbose:
                print(f"\n📨 Received: {json.dumps(json_packet, indent=2)}")
            else:
                print(f"\n📨 {msg_type}: {json_packet}")


class SandboxDebugger:
    """Interactive debugger for Sandbox Server."""
    
    def __init__(self, host='127.0.0.1', port=9091):
        self.host = host
        self.port = port
        self.client = None
        self.api_mode = None  # 'menu' or 'track'
        
    def connect(self):
        """Connect to the server."""
        try:
            print(f"Connecting to {self.host}:{self.port}...")
            self.client = DebugClient((self.host, self.port), verbose=True)
            time.sleep(0.5)  # Wait for connection
            if self.client.aborted:
                print("❌ Connection failed!")
                return False
            print("✓ Connected successfully!")
            return True
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def send_menu_command(self, cmd, *args):
        """Send Menu API command."""
        if cmd == 'get_scene_names':
            msg = {'msg_type': 'get_scene_names'}
            self.client.send_now(json.dumps(msg))
            print("→ Sent: get_scene_names")
            
        elif cmd == 'load_scene':
            if not args:
                print("❌ Usage: menu load_scene <scene_name>")
                return
            scene_name = args[0]
            msg = {'msg_type': 'load_scene', 'scene_name': scene_name}
            self.client.send_now(json.dumps(msg))
            print(f"→ Sent: load_scene {scene_name}")
            
        elif cmd == 'get_protocol_version':
            msg = {'msg_type': 'get_protocol_version'}
            self.client.send_now(json.dumps(msg))
            print("→ Sent: get_protocol_version")
            
        elif cmd == 'quit_app':
            msg = {'msg_type': 'quit_app'}
            self.client.send_now(json.dumps(msg))
            print("→ Sent: quit_app")
            
        elif cmd == 'connected':
            msg = {'msg_type': 'connected'}
            self.client.send_now(json.dumps(msg))
            print("→ Sent: connected")
            
        else:
            print(f"❌ Unknown menu command: {cmd}")
            print("Available: get_scene_names, load_scene, get_protocol_version, quit_app, connected")
    
    def send_track_command(self, cmd, *args):
        """Send Track API command."""
        if cmd == 'control':
            try:
                steering = float(args[0]) if len(args) > 0 else 0.0
                throttle = float(args[1]) if len(args) > 1 else 0.0
                brake = float(args[2]) if len(args) > 2 else 0.0
            except (ValueError, IndexError):
                print("❌ Usage: track control <steering> <throttle> [brake]")
                print("   Example: track control 0.5 0.3 0.0")
                return
            msg = {
                'msg_type': 'control',
                'steering': str(steering),
                'throttle': str(throttle),
                'brake': str(brake)
            }
            self.client.send_now(json.dumps(msg))
            print(f"→ Sent: control (steering={steering}, throttle={throttle}, brake={brake})")
            
        elif cmd == 'reset_car':
            msg = {'msg_type': 'reset_car'}
            self.client.send_now(json.dumps(msg))
            print("→ Sent: reset_car")
            
        elif cmd == 'exit_scene':
            msg = {'msg_type': 'exit_scene'}
            self.client.send_now(json.dumps(msg))
            print("→ Sent: exit_scene")
            
        elif cmd == 'get_protocol_version':
            msg = {'msg_type': 'get_protocol_version'}
            self.client.send_now(json.dumps(msg))
            print("→ Sent: get_protocol_version")
            
        elif cmd == 'step_mode':
            if len(args) < 2:
                print("❌ Usage: track step_mode <mode> <time_step>")
                print("   mode: 'synchronous' or 'asynchronous'")
                print("   Example: track step_mode synchronous 0.1")
                return
            mode = args[0]
            time_step = float(args[1])
            msg = {
                'msg_type': 'step_mode',
                'step_mode': mode,
                'time_step': str(time_step)
            }
            self.client.send_now(json.dumps(msg))
            print(f"→ Sent: step_mode (mode={mode}, time_step={time_step})")
            
        elif cmd == 'regen_road':
            if len(args) < 3:
                print("❌ Usage: track regen_road <road_style> <rand_seed> <turn_increment>")
                return
            try:
                road_style = int(args[0])
                rand_seed = int(args[1])
                turn_increment = float(args[2])
            except ValueError:
                print("❌ Invalid arguments")
                return
            msg = {
                'msg_type': 'regen_road',
                'road_style': str(road_style),
                'rand_seed': str(rand_seed),
                'turn_increment': str(turn_increment)
            }
            self.client.send_now(json.dumps(msg))
            print(f"→ Sent: regen_road")
            
        elif cmd == 'car_config':
            if len(args) < 5:
                print("❌ Usage: track car_config <body_style> <r> <g> <b> <car_name> [font_size]")
                print("   Example: track car_config car01 255 0 0 MyCar 100")
                return
            try:
                body_style = args[0]
                r = int(args[1])
                g = int(args[2])
                b = int(args[3])
                car_name = args[4]
                font_size = int(args[5]) if len(args) > 5 else 100
            except (ValueError, IndexError):
                print("❌ Invalid arguments")
                return
            msg = {
                'msg_type': 'car_config',
                'body_style': body_style,
                'body_r': str(r),
                'body_g': str(g),
                'body_b': str(b),
                'car_name': car_name,
                'font_size': str(font_size)
            }
            self.client.send_now(json.dumps(msg))
            print(f"→ Sent: car_config")
            
        elif cmd == 'cam_config':
            if len(args) < 8:
                print("❌ Usage: track cam_config <fov> <offset_x> <offset_y> <offset_z> <img_w> <img_h> <img_d> <img_enc> [rot_x] [rot_y] [rot_z] [fish_eye_x] [fish_eye_y]")
                print("   Example: track cam_config 120 0 3 0 640 480 3 JPG 90")
                return
            try:
                fov = float(args[0])
                offset_x = float(args[1])
                offset_y = float(args[2])
                offset_z = float(args[3])
                img_w = int(args[4])
                img_h = int(args[5])
                img_d = int(args[6])
                img_enc = args[7]
                rot_x = float(args[8]) if len(args) > 8 else 0.0
                rot_y = float(args[9]) if len(args) > 9 else 0.0
                rot_z = float(args[10]) if len(args) > 10 else 0.0
                fish_eye_x = float(args[11]) if len(args) > 11 else 0.0
                fish_eye_y = float(args[12]) if len(args) > 12 else 0.0
            except (ValueError, IndexError):
                print("❌ Invalid arguments")
                return
            msg = {
                'msg_type': 'cam_config',
                'fov': str(fov),
                'offset_x': str(offset_x),
                'offset_y': str(offset_y),
                'offset_z': str(offset_z),
                'img_w': str(img_w),
                'img_h': str(img_h),
                'img_d': str(img_d),
                'img_enc': img_enc,
                'rot_x': str(rot_x),
                'rot_y': str(rot_y),
                'rot_z': str(rot_z),
                'fish_eye_x': str(fish_eye_x),
                'fish_eye_y': str(fish_eye_y)
            }
            self.client.send_now(json.dumps(msg))
            print(f"→ Sent: cam_config")
            
        elif cmd == 'cam_config_b':
            # Same as cam_config but for second camera
            if len(args) < 8:
                print("❌ Usage: track cam_config_b <fov> <offset_x> <offset_y> <offset_z> <img_w> <img_h> <img_d> <img_enc> [rot_x] [rot_y] [rot_z] [fish_eye_x] [fish_eye_y]")
                return
            try:
                fov = float(args[0])
                offset_x = float(args[1])
                offset_y = float(args[2])
                offset_z = float(args[3])
                img_w = int(args[4])
                img_h = int(args[5])
                img_d = int(args[6])
                img_enc = args[7]
                rot_x = float(args[8]) if len(args) > 8 else 0.0
                rot_y = float(args[9]) if len(args) > 9 else 0.0
                rot_z = float(args[10]) if len(args) > 10 else 0.0
                fish_eye_x = float(args[11]) if len(args) > 11 else 0.0
                fish_eye_y = float(args[12]) if len(args) > 12 else 0.0
            except (ValueError, IndexError):
                print("❌ Invalid arguments")
                return
            msg = {
                'msg_type': 'cam_config_b',
                'fov': str(fov),
                'offset_x': str(offset_x),
                'offset_y': str(offset_y),
                'offset_z': str(offset_z),
                'img_w': str(img_w),
                'img_h': str(img_h),
                'img_d': str(img_d),
                'img_enc': img_enc,
                'rot_x': str(rot_x),
                'rot_y': str(rot_y),
                'rot_z': str(rot_z),
                'fish_eye_x': str(fish_eye_x),
                'fish_eye_y': str(fish_eye_y)
            }
            self.client.send_now(json.dumps(msg))
            print(f"→ Sent: cam_config_b")
            
        elif cmd == 'lidar_config':
            if len(args) < 10:
                print("❌ Usage: track lidar_config <offset_x> <offset_y> <offset_z> <rot_x> <degPerSweepInc> <degAngDown> <degAngDelta> <maxRange> <noise> <numSweepsLevels>")
                print("   Example: track lidar_config 0 0.5 0.5 0 2 0 -1.0 50.0 0.4 1")
                return
            try:
                offset_x = float(args[0])
                offset_y = float(args[1])
                offset_z = float(args[2])
                rot_x = float(args[3])
                degPerSweepInc = float(args[4])
                degAngDown = float(args[5])
                degAngDelta = float(args[6])
                maxRange = float(args[7])
                noise = float(args[8])
                numSweepsLevels = int(args[9])
            except (ValueError, IndexError):
                print("❌ Invalid arguments")
                return
            msg = {
                'msg_type': 'lidar_config',
                'offset_x': str(offset_x),
                'offset_y': str(offset_y),
                'offset_z': str(offset_z),
                'rot_x': str(rot_x),
                'degPerSweepInc': str(degPerSweepInc),
                'degAngDown': str(degAngDown),
                'degAngDelta': str(degAngDelta),
                'maxRange': str(maxRange),
                'noise': str(noise),
                'numSweepsLevels': str(numSweepsLevels)
            }
            self.client.send_now(json.dumps(msg))
            print(f"→ Sent: lidar_config")
            
        elif cmd == 'set_position':
            if len(args) < 3:
                print("❌ Usage: track set_position <pos_x> <pos_y> <pos_z> [Qx] [Qy] [Qz] [Qw]")
                print("   Example: track set_position 0 0 0")
                return
            try:
                pos_x = float(args[0])
                pos_y = float(args[1])
                pos_z = float(args[2])
                msg = {
                    'msg_type': 'set_position',
                    'pos_x': str(pos_x),
                    'pos_y': str(pos_y),
                    'pos_z': str(pos_z)
                }
                if len(args) >= 7:
                    msg['Qx'] = str(float(args[3]))
                    msg['Qy'] = str(float(args[4]))
                    msg['Qz'] = str(float(args[5]))
                    msg['Qw'] = str(float(args[6]))
            except ValueError:
                print("❌ Invalid arguments")
                return
            self.client.send_now(json.dumps(msg))
            print(f"→ Sent: set_position")
            
        elif cmd == 'node_position':
            if len(args) < 1:
                print("❌ Usage: track node_position <index>")
                return
            try:
                index = int(args[0])
            except ValueError:
                print("❌ Invalid index")
                return
            msg = {
                'msg_type': 'node_position',
                'index': str(index)
            }
            self.client.send_now(json.dumps(msg))
            print(f"→ Sent: node_position {index}")
            
        elif cmd == 'quit_app':
            msg = {'msg_type': 'quit_app'}
            self.client.send_now(json.dumps(msg))
            print("→ Sent: quit_app")
            
        else:
            print(f"❌ Unknown track command: {cmd}")
            print("Available commands:")
            print("  control <steering> <throttle> [brake]")
            print("  reset_car")
            print("  exit_scene")
            print("  get_protocol_version")
            print("  step_mode <mode> <time_step>")
            print("  regen_road <road_style> <rand_seed> <turn_increment>")
            print("  car_config <body_style> <r> <g> <b> <car_name> [font_size]")
            print("  cam_config <fov> <offset_x> <offset_y> <offset_z> <img_w> <img_h> <img_d> <img_enc> [rot_x] [rot_y] [rot_z] [fish_eye_x] [fish_eye_y]")
            print("  cam_config_b <fov> <offset_x> <offset_y> <offset_z> <img_w> <img_h> <img_d> <img_enc> [rot_x] [rot_y] [rot_z] [fish_eye_x] [fish_eye_y]")
            print("  lidar_config <offset_x> <offset_y> <offset_z> <rot_x> <degPerSweepInc> <degAngDown> <degAngDelta> <maxRange> <noise> <numSweepsLevels>")
            print("  set_position <pos_x> <pos_y> <pos_z> [Qx] [Qy] [Qz] [Qw]")
            print("  node_position <index>")
            print("  quit_app")
    
    def show_help(self):
        """Show help message."""
        print("\n" + "="*70)
        print("Sandbox Server Debug Client - Help")
        print("="*70)
        print("\nCommands:")
        print("\n  Menu API Commands:")
        print("    menu get_scene_names          - Get list of available scenes")
        print("    menu load_scene <name>       - Load a scene")
        print("    menu get_protocol_version    - Get protocol version")
        print("    menu quit_app                - Quit the application")
        print("    menu connected               - Send connected message")
        print("\n  Track API Commands:")
        print("    track control <s> <t> [b]    - Control car (steering, throttle, brake)")
        print("    track reset_car              - Reset car to start position")
        print("    track exit_scene             - Exit current scene")
        print("    track get_protocol_version   - Get protocol version")
        print("    track step_mode <m> <t>      - Set step mode (synchronous/asynchronous)")
        print("    track regen_road <s> <r> <i> - Regenerate road")
        print("    track car_config <style> <r> <g> <b> <name> [font] - Configure car")
        print("    track cam_config <fov> <ox> <oy> <oz> <w> <h> <d> <enc> [rx] [ry] [rz] [fx] [fy] - Configure camera")
        print("    track cam_config_b ...       - Configure second camera (same params as cam_config)")
        print("    track lidar_config <ox> <oy> <oz> <rx> <degInc> <degDown> <degDelta> <maxR> <noise> <sweeps> - Configure lidar")
        print("    track set_position <x> <y> <z> [Qx] [Qy] [Qz] [Qw] - Set car position")
        print("    track node_position <index>  - Get node position by index")
        print("    track quit_app               - Quit the application")
        print("\n  General Commands:")
        print("    help                         - Show this help")
        print("    status                       - Show connection status")
        print("    telemetry                    - Show last telemetry data")
        print("    save_image [filename]        - Save last received image")
        print("    quit                         - Exit debugger")
        print("\nExamples:")
        print("    menu get_scene_names")
        print("    menu load_scene warehouse")
        print("    track control 0.5 0.3 0.0")
        print("    track reset_car")
        print("    track cam_config 120 0 3 0 640 480 3 JPG 90")
        print("="*70 + "\n")
    
    def show_status(self):
        """Show current status."""
        print("\n" + "="*70)
        print("Connection Status")
        print("="*70)
        print(f"Host: {self.host}:{self.port}")
        print(f"Connected: {'Yes' if self.client and not self.client.aborted else 'No'}")
        print(f"Car Loaded: {'Yes' if self.client and self.client.car_loaded else 'No'}")
        print(f"Protocol Version: {self.client.protocol_version if self.client and self.client.protocol_version else 'Unknown'}")
        print(f"Available Scenes: {len(self.client.scene_names) if self.client else 0}")
        if self.client and self.client.last_telemetry:
            print(f"Last Telemetry: {time.strftime('%H:%M:%S', time.localtime())}")
        print("="*70 + "\n")
    
    def save_image(self, filename=None):
        """Save last received image."""
        if not self.client or not self.client.last_image:
            print("❌ No image available")
            return
        if filename is None:
            filename = f"debug_image_{int(time.time())}.png"
        try:
            self.client.last_image.save(filename)
            print(f"✓ Image saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving image: {e}")
    
    def show_telemetry(self):
        """Show last telemetry data."""
        if not self.client or not self.client.last_telemetry:
            print("❌ No telemetry data available")
            return
        print("\n" + "="*70)
        print("Last Telemetry Data")
        print("="*70)
        telemetry = self.client.last_telemetry.copy()
        # Remove large binary fields for display
        for key in ['image', 'imageb', 'lidar']:
            if key in telemetry:
                telemetry[key] = f"<{key} data>"
        print(json.dumps(telemetry, indent=2))
        print("="*70 + "\n")
    
    def run(self):
        """Run interactive debugger."""
        if not self.connect():
            return
        
        print("\n" + "="*70)
        print("Sandbox Server Debug Client")
        print("="*70)
        print("Type 'help' for available commands")
        print("Type 'quit' to exit")
        print("="*70 + "\n")
        
        try:
            while True:
                try:
                    cmd = input("sandbox> ").strip()
                    if not cmd:
                        continue
                    
                    parts = cmd.split()
                    action = parts[0].lower()
                    
                    if action == 'quit' or action == 'exit':
                        print("Disconnecting...")
                        break
                    
                    elif action == 'help':
                        self.show_help()
                    
                    elif action == 'status':
                        self.show_status()
                    
                    elif action == 'telemetry':
                        self.show_telemetry()
                    
                    elif action == 'save_image':
                        filename = parts[1] if len(parts) > 1 else None
                        self.save_image(filename)
                    
                    elif action == 'menu':
                        if len(parts) < 2:
                            print("❌ Usage: menu <command> [args...]")
                            print("   Type 'help' for menu commands")
                            continue
                        self.send_menu_command(parts[1], *parts[2:])
                        time.sleep(0.1)  # Small delay for response
                    
                    elif action == 'track':
                        if len(parts) < 2:
                            print("❌ Usage: track <command> [args...]")
                            print("   Type 'help' for track commands")
                            continue
                        self.send_track_command(parts[1], *parts[2:])
                        time.sleep(0.1)  # Small delay for response
                    
                    else:
                        print(f"❌ Unknown command: {action}")
                        print("   Type 'help' for available commands")
                
                except KeyboardInterrupt:
                    print("\n\nInterrupted. Type 'quit' to exit.")
                except EOFError:
                    print("\n\nExiting...")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        finally:
            if self.client:
                self.client.stop()
                print("✓ Disconnected")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive debugger for Sandbox Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=9091, help='Server port (default: 9091)')
    
    args = parser.parse_args()
    
    debugger = SandboxDebugger(args.host, args.port)
    debugger.run()


if __name__ == '__main__':
    main()

