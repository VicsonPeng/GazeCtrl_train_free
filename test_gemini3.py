"""Minimal test: Gemini 3 Pro image edit — matching official demo exactly."""
import os, mimetypes
from google import genai
from google.genai import types

import os
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE"))

# Read test image
img_path = "eval_images/" + os.listdir("eval_images")[0]
print(f"Using: {img_path}")
with open(img_path, "rb") as f:
    img_bytes = f.read()
mime = mimetypes.guess_type(img_path)[0] or "image/jpeg"

contents = [
    types.Content(role="user", parts=[
        types.Part.from_text(text="Make this person smile. Keep everything else the same."),
        types.Part(inline_data=types.Blob(mime_type=mime, data=img_bytes)),
    ]),
]

config = types.GenerateContentConfig(
    image_config=types.ImageConfig(
        image_size="1K",
    ),
    response_modalities=["IMAGE", "TEXT"],
)

print("Calling Gemini 3 Pro...")
file_index = 0
for chunk in client.models.generate_content_stream(
    model="gemini-3-pro-image-preview",
    contents=contents,
    config=config,
):
    if chunk.parts is None:
        continue
    if chunk.parts[0].inline_data and chunk.parts[0].inline_data.data:
        inline_data = chunk.parts[0].inline_data
        ext = mimetypes.guess_extension(inline_data.mime_type) or ".png"
        out = f"test_output_{file_index}{ext}"
        with open(out, "wb") as f:
            f.write(inline_data.data)
        print(f"OK! Saved: {out}")
        file_index += 1
    else:
        print(f"Text: {chunk.text}")

print("Done.")
