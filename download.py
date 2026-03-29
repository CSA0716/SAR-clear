import os
import time
import geopandas as gpd
import requests
from pathlib import Path
from tqdm import tqdm

os.environ["EOTDL_API_KEY"] = "69c7ffa763d77b90757a884f"

from eotdl.auth import auth
from eotdl.repos import FilesAPIRepo

DOWNLOAD_PATH = "/data/hdd2/clearsar/ClearSAR"
CATALOG = f"{DOWNLOAD_PATH}/catalog.v1.parquet"
MAX_RETRIES = 10
RETRY_DELAY = 5

user = auth()
gdf = gpd.read_parquet(CATALOG)

all_assets = []
for _, row in gdf.iterrows():
    for k, v in row["assets"].items():
        all_assets.append(v["href"])

print(f"共 {len(all_assets)} 个文件")

def download_file(url, base_path, user, retries=MAX_RETRIES):
    # 从 URL 解析相对路径
    rel_path = url.split("69a09161bb91b44193d52aa9/")[-1].split("?")[0]
    local_path = Path(base_path) / rel_path
    if local_path.exists():
        return  # 已存在跳过
    local_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            repo = FilesAPIRepo()
            repo.stage_file_url(url, base_path, user)
            return
        except Exception as e:
            if attempt < retries - 1:
                print(f"\n重试 {attempt+1}/{retries}: {rel_path} ({e})")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                print(f"\n跳过失败文件: {rel_path}")

for url in tqdm(all_assets, desc="下载中"):
    download_file(url, DOWNLOAD_PATH, user)

print("下载完成！")
