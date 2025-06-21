# Local_Multi_LLMs

# 🦙 Local Multi-LLMs Research Assistant & PDF RAG Demo

Welcome, fellow explorer of AI frontiers! 🚀  
This is your all-in-one **Research/Maths Assistant** and **PDF RAG** demo, powered by local LLMs (no OpenAI API key needed, your secrets are safe!).

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
   git clone https://github.com/<your-github-username>/<your-repo-name>.git
   cd <your-repo-name>
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

*Made with ❤️, Python, and a dash of AI
