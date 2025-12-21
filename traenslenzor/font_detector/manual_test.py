import asyncio
import uuid
import os
import random
import glob
from traenslenzor.file_server.client import FileClient, SessionClient
from traenslenzor.file_server.session_state import SessionState, TextItem, BBoxPoint
from PIL import Image

# Import logic directly from server
from traenslenzor.font_detector.server import detect_font_logic

async def run_manual_test():
    print("1. Uploading test image to File Server...")
    
    # Get all png images
    image_dir = "traenslenzor/font_detector/test_images"
    all_images = glob.glob(os.path.join(image_dir, "*.png"))
    
    if not all_images:
        print(f"No images found in {image_dir}")
        return

    # Pick one random image
    image_path = random.choice(all_images)
    
    print(f"\n--- Testing with random image: {image_path} ---")
    
    if not os.path.exists(image_path):
        print(f"Skipping {image_path} (not found)")
        return
        
    print(f"   Using image: {image_path}")
    
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Upload image using put_bytes
    file_id = await FileClient.put_bytes(os.path.basename(image_path), image_data)
    if not file_id:
        print("Error: Failed to upload image")
        return
    print(f"   Image uploaded with ID: {file_id}")

    print("\n2. Creating a test session...")
    # Create a text item that spans most of the image width
    text_item = TextItem(
        extractedText="Testing Font Detection 123",
        confidence=0.99,
        bbox=[
            BBoxPoint(x=10, y=10),    # UL
            BBoxPoint(x=990, y=10),   # UR
            BBoxPoint(x=990, y=190),  # LR
            BBoxPoint(x=10, y=190)    # LL
        ]
    )
    
    session = SessionState(
        rawDocumentId=file_id,
        text=[text_item]
    )
    
    # HACK: The File Server's POST /sessions endpoint isn't in the client, so we'll use httpx directly
    import httpx
    from traenslenzor.file_server.client import SESSION_ENDPOINT
    
    async with httpx.AsyncClient() as client:
        # The server uses POST /sessions to create a new session and returns the ID
        # It does NOT accept an ID in the URL for creation
        resp = await client.post(f"{SESSION_ENDPOINT}", json=session.model_dump())
        if resp.status_code != 200:
            print(f"Error creating session: {resp.text}")
            return
        
        # Get the ID returned by the server
        session_id = resp.json()["id"]
            
    print(f"   Session created with ID: {session_id}")

    print("\n3. Running Font Detector MCP Tool...")
    result = await detect_font_logic(session_id)
    print(f"   Result: {result}")

    print("\n4. Verifying Session Update...")
    updated_session = await SessionClient.get(session_id)
    
    if updated_session and updated_session.text:
        item = updated_session.text[0]
        print(f"   Detected Font: {item.detectedFont}")
        print(f"   Estimated Size: {item.font_size}pt")
        
        if item.detectedFont and item.font_size:
            print("\nSUCCESS: Font detection and sizing worked!")
        else:
            print("\nFAILURE: Fields were not updated.")
    else:
        print("\nFAILURE: Could not retrieve updated session.")

if __name__ == "__main__":
    asyncio.run(run_manual_test())
