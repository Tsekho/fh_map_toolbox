import os
import asyncio
import aiohttp

async def download_file(session, url, path, name, progress):
    async with session.get(url) as response:
        response.raise_for_status()
        content = await response.read()
        with open(path, "wb") as f:
            f.write(content)
    progress["completed"] += 1
    print(f"[{progress["completed"]}/{progress["total"]}] Downloaded: {name}")

async def count_items(session, repo_owner, repo_name, folder_path):
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}"

    async with session.get(api_url) as response:
        response.raise_for_status()
        items = await response.json()

    count = 0
    tasks = []
    for item in items:
        if item["type"] == "file":
            count += 1
        elif item["type"] == "dir":
            tasks.append(count_items(session, repo_owner, repo_name, item["path"]))

    subcounts = await asyncio.gather(*tasks) if tasks else []
    return count + sum(subcounts)

async def download_github_folder(session, repo_owner, repo_name, folder_path, local_dir, branch, progress):
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}"

    async with session.get(api_url) as response:
        response.raise_for_status()
        items = await response.json()

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    tasks = []
    for item in items:
        if item["type"] == "file":
            file_path = os.path.join(local_dir, item["name"])
            media_url = f"https://media.githubusercontent.com/media/{repo_owner}/{repo_name}/{branch}/{folder_path}/{item["name"]}"
            tasks.append(download_file(session, media_url, file_path, item["name"], progress))
        elif item["type"] == "dir":
            new_local_dir = os.path.join(local_dir, item["name"])
            tasks.append(download_github_folder(session, repo_owner, repo_name, item["path"], new_local_dir, branch, progress))

    await asyncio.gather(*tasks)

async def main():
    owner = "Wolfgang-IX"
    repo = "Foxhole-Map-Project"
    path = "Images/Baked-Mesh-Map"
    destination = "Baked-Mesh-Map"
    branch = "main"

    async with aiohttp.ClientSession() as session:
        print("Counting files...")
        total = await count_items(session, owner, repo, path)
        print(f"Found {total} files to download\n")

        progress = {"completed": 0, "total": total}
        await download_github_folder(session, owner, repo, path, destination, branch, progress)

        print(f'\nDownload complete! {progress["completed"]} files downloaded to {destination}')

if __name__ == "__main__":
    asyncio.run(main())
