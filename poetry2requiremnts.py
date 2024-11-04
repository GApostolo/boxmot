import toml

# Load poetry.lock file
with open("poetry.lock", "r") as file:
    poetry_lock = toml.load(file)

# Open a new requirements.txt file to write
with open("requirements.txt", "w") as req_file:
    for package in poetry_lock["package"]:
        # Get package name and version
        name = package["name"]
        version = package["version"]
        req_file.write(f"{name}=={version}\n")