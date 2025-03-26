import os
import subprocess

def execute_scripts_in_folders(base_path, selected_folders=None):
    for root, _, files in sorted(os.walk(base_path), key=lambda x: x[0]):
        if selected_folders and not any(folder in root for folder in selected_folders):
            continue
        for file in sorted(files):
            if file.endswith(".py") and file != os.path.basename(__file__):  # Avoid running itself
                script_path = os.path.join(root, file)
                print(f"\nExecuting: {script_path}")
                try:
                    subprocess.run(["python", script_path], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"\nError executing {script_path}: {e}")

if __name__ == "__main__":
    base_directory = os.getcwd()
    choice = input("Enter specific folders (comma-separated) or press Enter to scan all: ").strip()
    selected_folders = choice.split(',') if choice else None
    execute_scripts_in_folders(base_directory, selected_folders)

