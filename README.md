# chatterbot_tts
This is a flask api code to be deployed on kaggle and train the model on custom voice so that text given in a certain way be given as sound given
Okay, here's the documentation converted into a README.md format, including common errors to avoid and troubleshooting tips specifically for your Kaggle environment.

```markdown
# Chatterbox TTS FastAPI Application on Kaggle

This project provides a web API to generate speech from text using the Chatterbox TTS model. The API is built with FastAPI for easy interaction and served using Uvicorn. It is designed to run efficiently on Kaggle notebooks, leveraging GPU acceleration when available.

## Table of Contents

1.  [Project Overview](#1-project-overview)
2.  [File Structure](#2-file-structure)
3.  [Setup and Dependencies](#3-setup-and-dependencies)
4.  [Running the FastAPI Server](#4-running-the-fastapi-server)
5.  [Accessing Your API (Public Endpoint)](#5-accessing-your-api-public-endpoint)
6.  [Using the API Endpoints](#6-using-the-api-endpoints)
7.  [Common Errors and Troubleshooting](#7-common-errors-and-troubleshooting)

---

### 1. Project Overview

This FastAPI application serves as an interface to the Chatterbox TTS (Text-to-Speech) model. It allows users to send text input via HTTP requests and receive generated speech audio. The application is optimized for deployment within Kaggle's notebook environment.

### 2. File Structure

Your application's files should be organized within a sub-directory inside your Kaggle notebook's working directory (`/kaggle/working/`). The typical structure is:

```
/kaggle/working/
└── kplor-app/
    ├── app.py           # Your FastAPI application code
    ├── requirements.txt # (Optional, but recommended for explicit dependency management)
    └── kplor_voice.wav  # Your custom voice audio file (if used)
```

The `app.py` file should contain your FastAPI application instance, conventionally named `app = FastAPI()`.

### 3. Setup and Dependencies

All required Python libraries must be installed within the Kaggle environment. Place the following commands in a code cell at the **very beginning** of your Kaggle notebook. This ensures they run and install dependencies every time the notebook is executed (e.g., during "Save & Run All (Commit)").

```python
# Install core web server and API framework
!pip install fastapi uvicorn

# Install Chatterbox TTS and its core dependencies
!pip install chatterbox-tts
!pip install torch torchaudio transformers accelerate

# Install utility for audio file processing (e.g., .wav files)
!pip install pydub
```

### 4. Running the FastAPI Server

To start your FastAPI application, you will use Uvicorn. Given your `app.py` is located inside the `kplor-app` subdirectory, you must specify the correct module path in the Uvicorn command.

**Execute this command in a Kaggle notebook code cell:**

```bash
!uvicorn kplor-app.app:app --host 0.0.0.0 --port 8080 --reload
```

* **`!`**: Executes the command in the shell environment of the Kaggle notebook.
* **`uvicorn`**: The ASGI (Asynchronous Server Gateway Interface) server that runs your FastAPI application.
* **`kplor-app.app`**: This specifies the Python module to load. It tells Uvicorn to look for the `app.py` file inside the `kplor-app` directory.
* **`:app`**: Indicates that the FastAPI application instance named `app` should be loaded from within the `kplor-app.app` module.
* **`--host 0.0.0.0`**: Binds the server to all available network interfaces within the Kaggle container, making it accessible for Kaggle's proxy.
* **`--port 8080`**: Instructs Uvicorn to listen for incoming requests on port 8080.
* **`--reload`**: (Optional, for development) Enables auto-reloading of the server if code changes are detected in monitored directories.

**Expected Successful Output:**

Upon successful startup, your notebook cell will display logs similar to this, indicating the server is running and the model is loaded:

```
INFO:     Will watch for changes in these directories: ['/kaggle/working']
INFO:     Uvicorn running on [http://0.0.0.0:8080](http://0.0.0.0:8080) (Press CTRL+C to quit)
INFO:     Started reloader process [XXX] using WatchFiles
... (GPU/CUDA setup messages, model loading logs) ...
[INIT] Using device: cuda
[INIT] Loading Chatterbox TTS model...
/usr/local/lib/python3.11/dist-packages/diffusers/models/lora.py:393: FutureWarning: `LoRACompatibleLinear` is deprecated and will be removed in version 1.0.0. Use of `LoRACompatibleLinear` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`.
  deprecate("LoRACompatibleLinear", "1.0.0", deprecation_message)
loaded PerthNet (Implicit) at step 250,000
[INIT] Chatterbox TTS model loaded successfully on cuda.
[INIT] Setting up custom voice...
[INIT] Warning: Custom voice 'kplor_voice.wav' not found. Creating dummy.
[INIT] Initial Memory Usage (after model load): CPU: XXXX.XX MB, GPU: YYYY.YY MB
------------------------------------------------------------
INFO:     Application startup complete.
```
The cell will continue to run indefinitely as long as the server is active. To stop it, click the square "Stop" button next to the cell or use `Run > Interrupt`.

### 5. Accessing Your API (Public Endpoint)

To access your running FastAPI application from outside the Kaggle environment (e.g., from your web browser or another tool), you need to deploy it as a **Public Endpoint**. This provides a stable, public URL for your service.

**Steps to Deploy a Public Endpoint:**

1.  **Stop the running Uvicorn server:** In your notebook, click the square "Stop" button next to the running cell or go to `Run > Interrupt` from the top menu.
2.  **Save a new version (Commit):**
    * In the Kaggle notebook interface (top right), click **"Save Version"**.
    * Select **"Save & Run All (Commit)"**.
    * Provide a descriptive name for your version.
    * **Wait for the entire commit process to complete.** This involves re-running all cells from scratch, installing dependencies, and successfully starting your server. This can take several minutes to over an hour depending on model loading and environment setup.
3.  **Navigate to the Committed Version's Output:** Once the commit is successful, you'll need to go to the "Output" section of that specific committed version. You can usually find past versions via your notebook's "Version" history.
4.  **Find the "Deploy" or "Public Endpoint" Option:**
    * In the "Output" view of your committed notebook, look for a button or section labeled **"Deploy"** or **"Public Endpoint"**. Its exact placement or availability can vary based on Kaggle's UI and feature rollout.
5.  **Configure the Endpoint:**
    * A configuration dialog will prompt you for details:
        * **Application File:** Set this to `kplor-app/app.py`
        * **Application Object:** Set this to `app`
        * **Port:** Ensure this is set to `8080`
        * Choose a **Machine Type** (e.g., GPU for better performance) and set visibility (Public).
6.  **Deploy and Get URL:** Click "Deploy" and wait for the deployment process to complete. Once deployed, Kaggle will provide you with a **stable, public URL** for your FastAPI application.

### 6. Using the API Endpoints

Once your Public Endpoint is deployed and you have its URL, you can interact with your FastAPI application.

* **API Documentation (Swagger UI):**
    * Open your Public Endpoint URL in a web browser.
    * Append `/docs` to the URL to access the interactive API documentation.
    * **Example:** `[Your_Public_Endpoint_URL]/docs`
    * The Swagger UI allows you to explore all available endpoints (like `/tts/generate`), view their expected inputs, and test them directly.

* **Custom Voice Warning Handling:**
    * If you see the warning `[INIT] Warning: Custom voice 'kplor_voice.wav' not found. Creating dummy.` and you intend to use your specific custom voice:
        * Ensure your `kplor_voice.wav` file is present in your Kaggle notebook's working directory (`/kaggle/working/`).
        * If the file was uploaded as a "Dataset Input" (e.g., to `/kaggle/input/kplor-voice/`), you need to explicitly copy it to your working directory. Add a cell in your notebook with this command:
            ```bash
            !cp /kaggle/input/kplor-voice/kplor_voice.wav /kaggle/working/kplor_voice.wav
            ```
            (Adjust `kplor-voice` if your dataset has a different name).
        * After copying, save a new version of your notebook ("Save & Run All (Commit)") and redeploy your Public Endpoint.

### 7. Common Errors and Troubleshooting

Here are some common issues you might encounter and how to resolve them:

* **`ModuleNotFoundError: No module named 'fastapi'` (or any other library)**
    * **Cause:** The required Python library is not installed in the environment where the notebook is running (especially during a "commit" where a fresh environment is used).
    * **Solution:** Ensure all necessary `!pip install` commands are at the very beginning of your notebook and run successfully during the "Save & Run All (Commit)" process.

* **`Error loading ASGI app. Could not import module "app".`**
    * **Cause:** Uvicorn cannot find your `app.py` file or the `app` object within it. This typically happens if `app.py` is in a subdirectory, and the `uvicorn` command doesn't specify the correct path.
    * **Solution:** Ensure your `uvicorn` command correctly points to your module, like `!uvicorn kplor-app.app:app ...`, if `app.py` is inside `kplor-app/`.
    * Also, verify that the file is indeed named `app.py` and that your FastAPI instance within it is `app = FastAPI()`.

* **`SyntaxError: Invalid syntax` (especially after running `uvicorn` in console)**
    * **Cause:** Extra characters (like `cancel`, `arrow_right`) or incorrect commands were inadvertently typed or pasted into the console after the `uvicorn` command, causing the shell to interpret them as invalid Python or shell syntax.
    * **Solution:** Always use the `!uvicorn ...` command directly in a **Kaggle notebook code cell**, not the interactive console. This bypasses manual input issues. Avoid typing anything else after executing the command.

* **Public Endpoint Link Not Working ("Page not found", "Site can't be reached", or loading forever)**
    * **Cause:** Kaggle's temporary proxy URL might be incorrect, inactive, or the notebook session might have gone idle. The Public Endpoint might not have been correctly deployed or might still be in a pending state.
    * **Solution:**
        1.  Ensure your notebook completed its "Save & Run All (Commit)" successfully, and the `uvicorn` server indicated `Application startup complete.` in the commit logs.
        2.  **Verify the Public Endpoint deployment status.** It must show "Running" (or similar success) in your Kaggle "Output" or "Deployment" section.
        3.  Wait patiently after deployment; it can take time for the endpoint to become active.
        4.  If the "Deploy" button is not visible for your notebook, consult Kaggle's official documentation for "Public Endpoints" or "Deploy web app" for the most up-to-date and specific instructions for your account/notebook type.

* **Server runs for a long time (hundreds of seconds) after `Application startup complete.`**
    * **Cause:** This is normal! Once the server starts, it stays running and keeps the cell active to listen for requests. The timer indicates how long the server has been online, not a remaining time.
    * **Solution:** No action needed. The server is ready. To stop it, manually interrupt the cell.

```
