# 🦙 Local Multi-LLMs Research Assistant & PDF RAG Demo

Welcome, fellow explorer of AI frontiers! 🚀  
This is your all-in-one **Research/Maths Assistant** and **PDF RAG** demo, powered by local LLMs (no OpenAI API key needed, your secrets are safe!).

---

## 🛠️ Step 0: Install Python & Git (If You Haven't Already)

Before you start, make sure you have **Python** (version 3.9 or newer) and **Git** installed on your computer.

### 🐍 Install Python

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download the latest Python 3.x installer for your OS.
3. **IMPORTANT:** During installation, check the box that says **"Add Python to PATH"**.
4. Finish the installation.

To check if Python is installed, open a terminal (Command Prompt/PowerShell on Windows, Terminal on Mac/Linux) and run:
```sh
python --version
```
You should see something like `Python 3.10.12`.

### 🐙 Install Git

1. Go to [https://git-scm.com/downloads](https://git-scm.com/downloads)
2. Download and install Git for your OS (just click "Next" a bunch of times).
3. To check if Git is installed, run:
```sh
git --version
```
You should see something like `git version 2.42.0`.

---

## 🦙 Step 1: Get Ollama (Your Local LLM Zookeeper)

Before you can unleash the power of local LLMs, you need **Ollama** – the magical tool that wrangles AI models on your own machine.

1. **Download Ollama:**  
   Go to [https://ollama.com/download](https://ollama.com/download) and grab the installer for your OS.  
   (Yes, it’s free. Yes, it’s awesome.)

2. **Install Ollama:**  
   Run the installer. Follow the prompts.  
   If you can install Zoom, you can install Ollama.

3. **Start the Ollama server:**  
   Open a terminal and type:
   ```sh
   ollama serve
   ```
   (Leave this terminal open! It’s your AI engine room.)

---

## 🧠 Step 2: Download the Required LLMs

You’ll need three models. Don’t worry, it’s just a few commands:

```sh
ollama pull llava:7b
ollama pull qwen2.5vl:latest
ollama pull nomic-embed-text
```

Go grab a coffee ☕ while they download. (Or just stare at the progress bar, we won’t judge.)

---

## 🐍 Step 3: Set Up Python Stuff

1. **Clone this repo from GitHub:**

   ```sh
   git clone https://github.com/mjaiarchitect/Local_Multi_LLMs.git
   cd Local_Multi_LLMs
   ```

2. **Install Python requirements:**

   ```sh
   pip install -r requirements.txt
   ```

   If you see a lot of text flying by, that’s good. If you see errors, check you’re using Python 3.9+ and pip is up to date.

---

## 🏁 Step 4: Run the Demo!

1. **Fire up the app:**

   ```sh
   python Local_Multi_LLMs.py
   ```

2. **Open your browser and go to:**  
   [http://127.0.0.1:7860](http://127.0.0.1:7860)

3. **Enjoy!**  
   - Try the **Research/Maths Assistant** tab for web-powered research and math queries.
   - Try the **PDF RAG** tab to upload PDFs and ask questions about their content.
   - Marvel at the workflow and execution graphs.  
   - No cloud. No API keys. Just pure, local AI magic.

---

## 💻 Minimum Hardware Requirements

To run all three Ollama models (**llava:7b**, **qwen2.5vl:latest**, and **nomic-embed-text**) smoothly on your local machine, you’ll want at least:

- **CPU:** Modern quad-core (Intel i5/Ryzen 5 or better recommended)
- **RAM:** **16 GB** minimum (more is better, especially for multitasking)
- **Disk Space:** At least **20 GB** free (the models themselves take up several GB each)
- **GPU:** Not required, but if you have one, Ollama can use it for faster inference (NVIDIA with CUDA support recommended)
- **OS:** Windows 10/11, macOS, or Linux

> ⚠️ **Note:** You can run with 8 GB RAM, but you may need to close other apps and only load one model at a time for best results. For a smooth experience with all three models loaded, 16 GB+ RAM is strongly recommended.

---

## 🛠️ Troubleshooting

- **Ollama not found?**  
  Make sure it’s installed and running (`ollama serve`).

- **Models not found?**  
  Did you run the `ollama pull ...` commands?

- **Graphviz errors?**  
  Install [Graphviz](https://graphviz.org/download/) for your OS and add it to your PATH.

- **Python errors?**  
  Try `pip install --upgrade pip` and then `pip install -r requirements.txt` again.

---

## 💡 Pro Tips

- This app runs entirely on your machine. Your data never leaves your computer.
- You can run this offline (after models are downloaded).
- If you break something, just delete the folder and start again. We won’t tell anyone.

---

## 🎉 That’s it!  
You’re now the proud owner of a local, private, multi-LLM research assistant.  
Go forth and ask questions, solve math, and interrogate your PDFs like a boss!

---

*Made with ❤️, Python, and a dash of AI magic. Your loving ❤️ MJ!*
