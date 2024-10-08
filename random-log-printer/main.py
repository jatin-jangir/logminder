import os
import random
import time
import subprocess

def clone_repo(github_url):
    repo_name = github_url.split('/')[-1].replace('.git', '')
    if not os.path.exists(repo_name):
        print(f"Cloning repository {github_url}...")
        subprocess.run(["git", "clone", github_url], check=True)
    else:
        print(f"Repository {repo_name} already exists.")

def print_random_lines(file_path, n_lines,SLEEP_TIME):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            while True:
                start_line = random.randint(0, len(lines) - n_lines)
                print(f"\n--- Printing {n_lines} lines from line {start_line} ---")
                for i in range(start_line, start_line + n_lines):
                    print(lines[i].strip())
                time.sleep(SLEEP_TIME)  # Sleep for 2 seconds before printing again
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    file_location = os.getenv('FILE_LOCATION', '/HDFS/HDFS_2k.log')
    n_lines = int(os.getenv('N_LINES', 10))
    SLEEP_TIME = int(os.getenv('SLEEP_TIME', 10))
    print(f"Reading {n_lines} lines from file: {file_location}")
    # Clone the repository
    github_url = os.getenv('GITHUB_URL', 'https://github.com/logpai/loghub')
    clone_repo(github_url)
    print_random_lines(file_location, n_lines,SLEEP_TIME)
