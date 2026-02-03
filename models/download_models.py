#!/usr/bin/env python3
"""
SafEye Model Downloader
Downloads AI models required for the deepfake detection system.
Run this script to download models after cloning the repository.
"""

import os
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path

try:
    from azure.storage.blob import BlobClient
except Exception:
    BlobClient = None

def download_file(url, dest_path, desc=""):
    """Download a file with progress indication"""
    print(f"📥 Downloading {desc}...")

    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"✅ Downloaded {desc}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {desc}: {e}")
        return False

def download_from_azure_blob(container, blob_name, dest_path, desc=""):
    """Download a blob from Azure Storage using connection string or public URL."""
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    base_url = os.getenv("AZURE_BLOB_BASE_URL")
    sas_token = os.getenv("AZURE_SAS_TOKEN")

    if conn_str and BlobClient:
        try:
            print(f"📥 Downloading {desc} from Azure Blob Storage...")
            blob_client = BlobClient.from_connection_string(
                conn_str=conn_str,
                container_name=container,
                blob_name=blob_name,
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            print(f"✅ Downloaded {desc} from Azure")
            return True
        except Exception as e:
            print(f"❌ Failed to download {desc} from Azure: {e}")
            return False

    if base_url:
        url = f"{base_url.rstrip('/')}/{container}/{blob_name}"
        if sas_token:
            url = f"{url}?{sas_token.lstrip('?')}"
        return download_file(url, dest_path, desc=desc)

    print("ℹ️  Azure config not found. Skipping Azure download.")
    return False

def extract_archive(archive_path, extract_to, desc=""):
    """Extract zip or tar.gz archive"""
    print(f"📦 Extracting {desc}...")

    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.tar.gz'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)

        print(f"✅ Extracted {desc}")
        return True
    except Exception as e:
        print(f"❌ Failed to extract {desc}: {e}")
        return False

def setup_audio_model():
    """Download and setup audio deepfake detection model"""
    print("\n🎵 Setting up Audio Detection Model...")

    audio_dir = Path("models/audio_model")
    audio_dir.mkdir(parents=True, exist_ok=True)

    container = os.getenv("AZURE_STORAGE_CONTAINER", "")
    blob_list = os.getenv("AZURE_AUDIO_BLOBS", "")

    if container and blob_list:
        blobs = [b.strip() for b in blob_list.split(",") if b.strip()]
        for blob_name in blobs:
            dest_path = audio_dir / Path(blob_name).name
            download_from_azure_blob(
                container=container,
                blob_name=blob_name,
                dest_path=dest_path,
                desc=f"audio model file ({blob_name})",
            )
        return

    print("ℹ️  Audio model setup placeholder - set AZURE_STORAGE_CONTAINER and AZURE_AUDIO_BLOBS")

def setup_text_model():
    """Download and setup text misinformation detection model"""
    print("\n📝 Setting up Text Detection Model...")

    text_dir = Path("models/text_model")
    text_dir.mkdir(parents=True, exist_ok=True)

    container = os.getenv("AZURE_STORAGE_CONTAINER", "")
    blob_list = os.getenv("AZURE_TEXT_BLOBS", "")

    if container and blob_list:
        blobs = [b.strip() for b in blob_list.split(",") if b.strip()]
        for blob_name in blobs:
            dest_path = text_dir / Path(blob_name).name
            download_from_azure_blob(
                container=container,
                blob_name=blob_name,
                dest_path=dest_path,
                desc=f"text model file ({blob_name})",
            )
        return

    print("ℹ️  Text model setup placeholder - set AZURE_STORAGE_CONTAINER and AZURE_TEXT_BLOBS")

def setup_image_model():
    """Download and setup image deepfake detection model"""
    print("\n🖼️  Setting up Image Detection Model...")

    image_dir = Path("models/image_model")
    image_dir.mkdir(parents=True, exist_ok=True)

    container = os.getenv("AZURE_STORAGE_CONTAINER", "")
    image_blobs = os.getenv("AZURE_IMAGE_BLOBS", "")
    image_blob = os.getenv("AZURE_IMAGE_BLOB", "")

    if image_blobs or image_blob:
        blobs = [b.strip() for b in image_blobs.split(",") if b.strip()]
        if image_blob:
            blobs.append(image_blob)

        for blob_name in blobs:
            filename = Path(blob_name).name
            dest_path = image_dir / filename
            if dest_path.exists():
                print(f"✅ Image model file already exists: {filename}")
                continue

            downloaded = download_from_azure_blob(
                container=container,
                blob_name=blob_name,
                dest_path=dest_path,
                desc=f"image model file ({blob_name})",
            )

            if downloaded and (filename.endswith('.zip') or filename.endswith('.tar.gz')):
                extract_archive(str(dest_path), str(image_dir), desc="image model archive")
        return

    # Fallback: use direct URL if provided
    model_url = os.getenv("IMAGE_MODEL_URL", "")
    if model_url:
        filename = Path(model_url).name or "image_model.bin"
        dest_path = image_dir / filename
        if download_file(model_url, dest_path, desc="image model"):
            if filename.endswith('.zip') or filename.endswith('.tar.gz'):
                extract_archive(str(dest_path), str(image_dir), desc="image model archive")
        return

    print("ℹ️  Image model setup placeholder - set AZURE_STORAGE_CONTAINER and AZURE_IMAGE_BLOBS")

def main():
    """Main setup function"""
    print("🤖 SafEye Model Downloader")
    print("=" * 50)

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Setup each model
    setup_audio_model()
    setup_text_model()
    setup_image_model()

    print("\n✅ Model setup complete!")
    print("\n📋 Next steps:")
    print("1. Install Python dependencies: pip install -r requirements.txt")
    print("2. Install Node.js dependencies: npm install")
    print("3. Start the backend: python app.py")
    print("4. Start the frontend: npm run dev")

if __name__ == "__main__":
    main()
