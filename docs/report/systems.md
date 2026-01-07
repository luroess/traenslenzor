# Smart Translator
- Smart Translator taks an image of a single page document from hard drive or webcam.
- It then translates detected text blocks to another language and generates a new document with the translated text blocks. Important:
  - The font and font size should match, and the rendered text should blend seamlessly with the background.
  - The orientation should match.
  - Text block size and position should match.
- In addition, the system recognizes the document type.
- In short:
  - Input: document image
  - Output: document image, document type

# Interaction
The interaction works via text prompt.
- The user describes the document translation task (see next slide).
- A supervisor (language model) answers with the precise steps he would take (tools to use in order) to fulfill the document translation task.
- The supervisor also asks for the necessary/missing tool parameters (e.g. the image path or if he should take a webcam image)
- The user can cancel the process by describing another document translation task.

# UI
- You can use a Python backend/browser frontend architecture
- You can also use a Desktop-GUI library (e.g. Qt for Python).

# Supervisor
- The supervisor interacts with the user and plans/executes tool chains.
- **Tech requirements:**
  - Use Python (like everything else except UI frontend if implemented for the browser)
  - It must be implemented using a local LLM (e.g. gemma3:1b with Ollama).
  - Tools must be deployed as MCP servers (use fastmcp).
  - Consider using langgraph / langchain.

# Image Provider
- Loads an image from hard drive or captures a webcam image.
- Input: source (path or webcam), Output: document image
- Can contain pre-processing steps (contrast adjustment, etc.)

# Layout Detector
- Extracts and transforms the document within the document image and returns text boxes.
- Input: document image, Output: extracted & transformed document image, text boxes + text.
- **Tech requirements:**
  - Must be implemented as a MCP tool
  - Extract the document's outline and transform the image portion.
  - Consider using PaddleOCR for text box extraction.

It is ok, if text boxes only contain a single line!

# Document Class Detector
- Detects the document class
- Input: document image, Output: document class label
- Train your own models on the [RVL-CDIP](https://adamharley.com/rvl-cdip/) dataset.

## Training
- **Tech requirements:**
  - Models must be implemented in PyTorch.
  - Model interpretation with captum.
  - Use a experiment tracking tool (e.g. mlflow, w&b).
  - Must be implemented as an MCP tool.

### AlexNet

our own implementation
from scratch

### ResNet-50
use existing implementation

pre-trained on ImageNet
- classification head only on [RVL-CDIP](https://adamharley.com/rvl-cdip/)
- fine-tuned on [RVL-CDIP](https://adamharley.com/rvl-cdip/)

### Vision Transformer
use existing implementation

pre-trained on ImageNet
- classification head only on [RVL-CDIP](https://adamharley.com/rvl-cdip/)
- fine-tuned on [RVL-CDIP](https://adamharley.com/rvl-cdip/)

### Evaluation
- Test set results (choose suitable metrics)
- Loss curves
- Worst and best case examples with model interpretation
- **Training should also contain a sensible hyper parameter search (learning rate, batch size, …). You can use tools (like optuna) for it or do it by hand.**
- **Use appropriate regularization techniques**

-> Use the best model for your system.

# Font Detector
- Detects font name and font size for a text block.
  - Input: text box image, text box size, text
  - Output: font name, font size

- **Tech requirements**
  - Must be implemented as an MCP tool
  - For font name detection: use [this font identifier model](https://huggingface.co/gaborcselle/font-identifier). -> [gh repo](https://github.com/gaborcselle/font-identifier)
  - For font size estimation:
  - Formulate it as a per-font regression problem <- That means, train one model per font (do this for 5 fonts of your choice)
    - Input: text box size, text length, letter distribution
    (appearance of letters in the text)
    - Output: font size
  - Complete & use our own tiny_diff package and implement an MLP model. (Note from Bene: are we meant to implement our own pytorch? e.g. like [micrograd](https://github.com/karpathy/micrograd))
  - Implement a data-set generator for training/validation/test sets
  - Use ReLU and MSE-Loss
  - Don’t forget input normalization!
  - Train and test your model

# Document Translator
- Detects the language of the input text and translates it to a language of choice.
- Input: text in language A, target language B, Output: text in language B
- **Tech requirements:**
  - Must be implemented as a MCP tool
  - Use your local LLM for that.

# Document Image Renderer
- Takes the input document image, replaces the text of all text boxes with the translated text.
- Input: document image, document image transform, text boxes + translated text + font info
- Output: translated document image
- **Tech requirements:**
  - Remove the text boxes from the input image (use lama for example).
  - Draw the translated text (using font info, transp. background) in the text boxes.
  - Apply the inverse document image transform
  - Add result to the input document image

# X
- A component for which the team defines the functionality.
- Input: Y, Output Z
- (Tech) requirements:
  ‒ Must provide value to the system as a whole.
  ‒ Must have sufficient technical depth.
