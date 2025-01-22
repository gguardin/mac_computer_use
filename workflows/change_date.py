import subprocess
import time
import pyautogui
from datetime import datetime
import sys
from tools.computer import ComputerTool, ScalingSource

# Initialize computer tool for coordinate transformation
computer = ComputerTool()


def transform_coords(x: int, y: int) -> tuple[int, int]:
    """Transform coordinates from API space to OS space."""
    return computer.transform_coordinates(ScalingSource.API, x, y)


def click(x, y):
    """Execute a click at the specified coordinates."""
    real_x, real_y = transform_coords(x, y)
    x_str = f"={real_x}" if real_x < 0 else str(real_x)
    y_str = f"={real_y}" if real_y < 0 else str(real_y)
    subprocess.run(["cliclick", f"c:{x_str},{y_str}"], check=True)
    time.sleep(0.5)


def right_click(x, y):
    """Execute a right click at the specified coordinates."""
    real_x, real_y = transform_coords(x, y)
    x_str = f"={real_x}" if real_x < 0 else str(real_x)
    y_str = f"={real_y}" if real_y < 0 else str(real_y)
    subprocess.run(["cliclick", f"rc:{x_str},{y_str}"], check=True)
    time.sleep(0.5)


def move_to(x, y):
    """Move the cursor to the specified coordinates."""
    real_x, real_y = transform_coords(x, y)
    x_str = f"={real_x}" if real_x < 0 else str(real_x)
    y_str = f"={real_y}" if real_y < 0 else str(real_y)
    subprocess.run(["cliclick", f"m:{x_str},{y_str}"], check=True)
    time.sleep(0.5)


def type_text(text):
    """Type the specified text."""
    pyautogui.write(text)
    time.sleep(0.5)


def modify_alert(alert_y_position):
    """Modify a single alert's expiration date."""
    # Click on the alert (x is fixed, y varies)
    click(735, alert_y_position)
    time.sleep(1)  # Wait for any UI updates

    # Right-click to open context menu
    right_click(735, alert_y_position)
    time.sleep(1)  # Wait for menu to appear

    # Click "Edit" option in context menu
    click(785, 242)
    time.sleep(1)  # Wait for edit dialog

    # Click expiration field
    click(346, 530)
    time.sleep(1)

    # Click calendar icon field
    click(500, 530)
    time.sleep(1)

    # Select all existing text and delete it
    pyautogui.hotkey("command", "a")
    time.sleep(0.5)
    pyautogui.press("delete")
    time.sleep(0.5)

    # Type new date
    type_text("2025-03-10")
    time.sleep(0.5)

    # Click Set/Save button
    click(678, 704)
    time.sleep(1)

    # Verify by opening edit dialog again
    right_click(735, alert_y_position)
    time.sleep(1)
    click(785, 242)
    time.sleep(1)

    # Take a screenshot for verification (optional)
    subprocess.run(
        [
            "screencapture",
            f'/tmp/alert_verification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png',
        ],
        check=True,
    )

    # Close verification dialog
    click(609, 704)  # Click Cancel
    time.sleep(1)


def main():
    # Wait a few seconds before starting to give time to switch to the correct window
    print("Starting in 5 seconds... Switch to the target window!")
    time.sleep(5)

    # Click File menu to activate the window
    click(96, 13)
    click(96, 13)  # Click again to hide menu

    # Starting y-position for first alert
    start_y = 186

    # Number of alerts to process (can be passed as command line argument)
    num_alerts = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    # Process each alert
    for i in range(num_alerts):
        current_y = start_y + (i * 77)  # Adjust this offset based on your UI
        print(f"Processing alert {i+1} at y-position {current_y}")
        modify_alert(current_y)
        time.sleep(1)  # Wait between alerts

        # Optionally scroll down after every few alerts
        if (i + 1) % 3 == 0:
            pyautogui.press("down")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
