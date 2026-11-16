# HECRIG

## Local model

HECRIG runs the model **inside its own Python process** using
[`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python). There is no
`llama-server`, no background service, and no HTTP endpoint: the app loads the
`.gguf` itself and generates tokens in-process.

It looks for a `.gguf` model in `models/`, the project root, any sub-folder of
the project (up to four levels deep), or the Hugging Face cache. That means you
can simply paste a model folder into the app directory.

Install the binding once:

```bash
pip install llama-cpp-python
```

Override the defaults with environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLAMA_MODEL_PATH` | auto-detected | Full path to the `.gguf` model |
| `HECRIG_CTX` | `8192` | Context size |
| `HECRIG_NGL` | *(all layers when a GPU build is detected)* | GPU layers to offload (`0` = CPU only) |
| `HECRIG_THREADS` | half the logical cores | CPU threads |
| `HECRIG_BATCH` | `512` | Prompt batch size |
| `HECRIG_FLASH_ATTN` | *(llama.cpp default)* | Set `on`/`off` to override |
| `HECRIG_MAX_TOKENS` | `0` (unlimited) | Cap on one reply |
| `HECRIG_MMPROJ` | auto-detected | Path to the vision projector (`mmproj*.gguf`) |
| `HECRIG_VISION_GPU` | `0` | Set to `1` to run the vision projector on the GPU too |

If a matching `mmproj*.gguf` sits next to the model (or in the Hugging Face
cache), HECRIG loads it with `MTMDChatHandler`, which lets the model see images
directly. The projector stays on the CPU on purpose: with a 9B model fully
offloaded to an 8 GB card there is no VRAM left for it, and the GPU path
crashes instead of falling back.

### Speed

The model runs on whatever backend your `llama-cpp-python` build was compiled
with. Check it with:

```python
import llama_cpp
print(llama_cpp.llama_supports_gpu_offload())
```

If that prints `False`, the wheel is CPU-only: about 6 tokens/sec of generation
and ~30 tokens/sec of prompt processing for this 9B model — roughly a minute
before the first word of a reply, and this build re-processes the prompt on
every turn.

This machine now runs a **Vulkan build: ~35 tokens/sec, first token in well
under a second**. To build it yourself (Vulkan works on NVIDIA, AMD and Intel
GPUs):

1. Install the Vulkan SDK: `winget install KhronosGroup.VulkanSDK`.
2. Use a *x64 Native Tools Command Prompt for VS* (or call `vcvars64.bat`
   first) — CMake cannot find the MSVC compiler without it.
3. Build:

```bat
set VULKAN_SDK=C:\VulkanSDK\1.4.363.0
set PATH=%VULKAN_SDK%\Bin;%PATH%
set CMAKE_ARGS=-DGGML_VULKAN=on
pip install --force-reinstall --no-cache-dir llama-cpp-python
```

A CUDA build (`set CMAKE_ARGS=-DGGML_CUDA=on`) is often faster still on NVIDIA
cards, but it needs the full CUDA toolkit.

HECRIG offloads every layer automatically when the build supports it, and falls
back to CPU if VRAM is short. Other speed levers:

- `HECRIG_NGL=99` — offload every layer (only useful with a GPU build).
- `HECRIG_THREADS=16` — more CPU threads if you are CPU-bound.
- A smaller model is the biggest single win: a 4B at the same quantization is
  roughly twice as fast as a 9B.
- A smaller `HECRIG_CTX` reduces memory use and prompt-processing time.

## Stopping a reply

While HECRIG is answering, the send button turns into a red stop button. Press it
to abort the reply; whatever text already arrived stays in the chat and you can
send a new message straight away. If nothing had arrived yet, the chat shows
"Stopped." instead.

## While the model loads

The message box stays locked until the model is loaded and ready, with the
placeholder "Waiting for the model to load…". The badge in the header shows
`starting…` and turns green when the model is ready; the input unlocks at the
same moment. This stops you from typing a message that would fail because the
model is not loaded yet.

## Internet access

HECRIG can search the live web and read pages using llama.cpp tool calling.
Web search is on by default; the globe icon in the header toggles it.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HECRIG_WEB_SEARCH` | `1` | Set to `0` to disable web tools |
| `TAVILY_API_KEY` | empty | Optional Tavily key for higher-quality search |
| `HECRIG_SEARCH_RESULTS` | `5` | Results returned per search |
| `HECRIG_FETCH_CHARS` | `6000` | Max characters read from a page |
| `HECRIG_MAX_TOOL_STEPS` | `4` | Max search/read rounds per answer |

Without a Tavily key it uses DuckDuckGo, which needs no account.

## Images, PDFs, and documents

Click the paperclip in the message bar to attach up to four files. HECRIG reads
them locally before sending your message:

| Type | How it is read |
| --- | --- |
| Images (PNG, JPG, WEBP, BMP, GIF, TIFF) | Sent to the model's vision projector; OCR is used if vision is unavailable |
| PDF | Text is extracted per page; scanned pages are sent as images or OCR'd |
| DOCX, ODT, ODS, ODP, RTF | Text and tables are extracted |
| XLSX | Rows from each sheet are extracted |
| PPTX | Text from each slide is extracted |
| TXT, MD, CSV, JSON, XML, YAML, code | Read as plain text |

The model can also read a file by path with the `read_document` tool, for
example when you paste a path into the chat.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HECRIG_MAX_ATTACHMENTS` | `4` | Files per message |
| `HECRIG_MAX_ATTACHMENT_MB` | `40` | Max size per file |
| `HECRIG_ATTACH_CHARS` | `6000` | Max text characters per file |
| `HECRIG_ATTACH_TOTAL_CHARS` | `12000` | Max text characters per message |
| `HECRIG_IMAGE_MAX_SIDE` | `1024` | Images are resized to this maximum side |
| `HECRIG_PDF_MAX_PAGES` | `30` | PDF pages read per file |
| `HECRIG_OCR_MAX_PAGES` | `6` | Scanned PDF pages OCR'd per file |
| `HECRIG_MAX_IMAGES` | `4` | Images sent to the model per message |

OCR uses the OCR engine built into Windows 10/11. If `pytesseract` or
`rapidocr_onnxruntime` is installed it is used as a fallback.

## Memory and skills

HECRIG can remember facts about you and save reusable skills (repeatable
procedures). Both survive restarts and are injected into every new chat.

- **Memory** — say "remember that I prefer short answers" and HECRIG saves it.
  It also saves durable facts automatically when you share them.
- **Skills** — say "learn a skill called Daily briefing: search the news and
  summarize the top 3 stories" and HECRIG stores the steps. Later, "use my
  Daily briefing skill" loads them.

Open the brain icon in the sidebar (`Memory & skills`) to view, add, or delete
memories and skills. The model can also call `remember`, `forget`, `save_skill`,
`use_skill`, `list_skills`, and `delete_skill` itself.

Data is stored as JSON under `%LOCALAPPDATA%\HECRIG` on Windows
(`~/.hecrig` elsewhere):

| File | Contents |
| --- | --- |
| `memory.json` | Remembered facts |
| `skills.json` | Saved skills and instructions |

| Variable | Default | Purpose |
| --- | --- | --- |
| `HECRIG_DATA_DIR` | OS app data | Override the storage folder |
| `HECRIG_MAX_MEMORIES` | `50` | Max saved memories (oldest dropped) |
| `HECRIG_MAX_SKILLS` | `50` | Max saved skills (oldest dropped) |

Secrets such as passwords and API keys are never saved.

## Run the app

### uv

Run as a desktop app:

```bash
uv run flet run
```

Run as a web app:

```bash
uv run flet run --web
```

For more details on running the app, refer to the [Getting Started Guide](https://flet.dev/docs/).

## Build the app

### Android

```bash
flet build apk -v
```

For more details on building and signing `.apk` or `.aab`, refer to the [Android Packaging Guide](https://flet.dev/docs/publish/android/).

### iOS

```bash
flet build ipa -v
```

For more details on building and signing `.ipa`, refer to the [iOS Packaging Guide](https://flet.dev/docs/publish/ios/).

### macOS

```bash
flet build macos -v
```

For more details on building macOS package, refer to the [macOS Packaging Guide](https://flet.dev/docs/publish/macos/).

### Linux

```bash
flet build linux -v
```

For more details on building Linux package, refer to the [Linux Packaging Guide](https://flet.dev/docs/publish/linux/).

### Windows

```bash
flet build windows -v
```

For more details on building Windows package, refer to the [Windows Packaging Guide](https://flet.dev/docs/publish/windows/).

### Web

```bash
flet build web -v
```

For more details on building Web app, refer to the [Web Packaging Guide](https://flet.dev/docs/publish/web/).
