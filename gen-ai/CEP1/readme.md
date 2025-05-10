
# **Edgerunners Archive — Llama-3-8B Experimental Setup**

**Complete CEP Source Code:**  
[chatBotWithGradio_lin.py](https://github.com/Rimbik/assessments/blob/main/gen-ai/CEP1/code/chatBotWithGradio_lin.py)  
**Reference:**  
[Llama Offline Documentation](https://github.com/Rimbik/ai-nextGen/blob/main/genai/Llama/Llama_Offline.md)

---

## 🔍 **Model Overview**

- **Type:** Llama-3-8B  
- **Name:** `EdgerunnersArchive-Llama-3-8B-Instruct-ortho-baukit-toxic-v2-Q2_K.gguf`  
- **Base Model:** Meta Llama 3 8B Instruct  
- **Quantization:** Q2_K  
  - 2-bit (Q2_K): ~3.03 GB to run  
  - Other versions: Q3_K_S, Q3_K_L, Q5_K_S (2 to 8-bit variants available)  
- **Fine-Tuning:** Applied with `ortho-baukit-toxic-v2` adjustments  
- **Download Size:** ~3.18 GB  
- **License:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use)  

---

## ⚙️ **Technical Details**

- **Quantization Type:**  
  - Q2_K = Lower-bit quantization  
  - Results in smaller model size but reduced performance vs. Q4_K or Q5_K
- **Usage:**  
  - Ideal for low-resource environments  
  - Tested on a Linux Celeron machine with 2GB RAM  
- **Performance Note:**  
  - Q2_K trades off speed and accuracy for compact size

---

## 🛠️ **Setup and Testing**

- **High-end Test:**  
  - Q4_K model on a Ryzen i7 Mini PC (32 GB RAM)  
- **Edge Test:**  
  - Q2_K model on Linux Mint (Celeron + 2GB RAM + 2GB swap)  
  - Result: Model ran successfully but search latency was high  
- **To-Do:**  
  - Test on Raspberry Pi *(not done yet)*

---

## ⚡ **Build Strategy**

- `llama.cpp` compiled on Linux  
- On edge devices:
  - Used prebuilt binary of `llama.cpp` (compiling was too time-consuming)  
  - `llama-cpp-python` compiled manually (no prebuilt binary available)  
- Swap used: 2GB  
- **Observation:**  
  - Model ran under constrained RAM setup, but latency was significant

---

## 🌐 **Gradio Hosting**

> Hugging Face integration not configured for offline Gradio hosting  
> → Live URL currently unavailable

---

## 🧩 **Other References**

- **ollama**  
- **LMStudio**  
- **llama.cpp (Linux build)**  
- **Main reference:** [Llama Offline Guide](https://github.com/Rimbik/ai-nextGen/blob/main/genai/Llama/Llama_Offline.md)
