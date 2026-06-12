# Image_captioner

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/bahulmishra/Image_captioner/blob/main/IMAGE_CAPTIONER.ipynb)


## 1. Project Scope and Purpose
Welcome to the Image Caption Generator! I built this notebook as an interactive project to help demystify the magic behind Artificial Intelligence.

When we look at a photograph, we instantly understand the context i.e we see a dog running, a child laughing, or a sunset over a bridge. But how do we teach a machine to do the same? This project is designed to help you visualize exactly how Machine Learning (ML), Deep Learning (DL), and Natural Language Processing (NLP) work together to look at an image and generate a meaningful, human-readable description from scratch.


## The Input: What Are We Feeding the Model?
To teach a machine, you need a lot of examples. For this project, we are using the famous Flickr8k dataset.

* Dataset Name: Flickr8k
* Total Images: 8,091 unique images containing various everyday scenes (people, animals, sports, nature).
* Image Resolution: The images come in various shapes and sizes, but during our preprocessing pipeline, we mathematically resize and standardise every single image to 299x299 pixels. This specific resolution is required by our vision model.
* The Text Data: This dataset is special because it doesn't just have one label per image. Every single image is paired with 5 different human-written captions. This allows our model to learn the nuances of grammar, synonyms, and different ways to describe the exact same scene.

## 2. The Brains of the Operation: Model Architecture & Choices
This model is a classic "Encoder-Decoder" setup. It uses two separate "brains" that merge together to make a final decision.

### The Vision Branch (The Eyes)
* Model Used: InceptionV3 (Pre-trained on ImageNet).

- Why we used it: We use InceptionV3 as an Encoder. Instead of asking it to classify the image, we chop off its final classification layer and pull the raw mathematical features (a 2048-dimensional vector). It's incredibly efficient at identifying textures, shapes, and objects.

- Alternatives: VGG16 or ResNet50. VGG16 is a bit too heavy and slow, while ResNet50 is great, but InceptionV3 provides a fantastic balance of speed and feature richness.

### The Language Branch (The Memory)
* Model Used: LSTM (Long Short-Term Memory) network.

- Why we used it: LSTMs are a type of Recurrent Neural Network (RNN) designed to remember the sequence of data. It looks at the words generated so far and uses its memory to predict what grammatically makes sense next.

- Alternatives: GRUs (faster but sometimes less accurate) or Transformers (like GPT). Transformers are the modern industry standard, but they require massive amounts of data. For a small 8,000-image dataset where we want to learn from scratch, an LSTM is the perfect tool for the job.

### Custom Modifications & "The Goldilocks Zone"
- To stop the model from simply memorizing the training data (overfitting), I made a few highly specific architectural tweaks:

- Dimensionality Reduction: I compressed the vision and language features down to 128 dimensions. This drastically speeds up training without sacrificing the model's ability to learn.

- Dropout (0.3): I added a 30% dropout rate. This acts as a "blindfold" during training, randomly turning off 30% of the neurons. It forces the network to actually learn the underlying patterns instead of taking shortcuts.

### The Under the Hood Mechanics
* Loss Function: Categorical Cross-Entropy. Why? Because predicting the next word is essentially a massive multiple-choice question. Our optimized dictionary has 3,319 words, so the model is choosing 1 correct "class" out of 3,319 options. (Alternative: Sparse Categorical Cross-Entropy).

* Optimizer: Adam Optimizer with a custom Learning Rate Scheduler (decaying by 5% every epoch).
   Why? : Adam dynamically adjusts how fast it learns. Our scheduler slows it down smoothly as it gets closer to the perfect answer, preventing it from overshooting the goal.

* Validation Metric: BLEU Score (Bilingual Evaluation Understudy).
   Why? BLEU specifically measures how closely the machine-generated text matches the human-written reference texts, rewarding correct n-grams (phrases). (Alternative: METEOR or CIDEr)

  ## Important Libraries & Step-by-Step Workflow
The Toolkit:

- TensorFlow / Keras: The heavy lifters. Used to build, train, and run the neural network.

- NLTK: Specifically for calculating the BLEU scores to evaluate our grammar.

- Matplotlib / Seaborn: For plotting training loss curves and building our beautiful final UI dashboard.

- NumPy / Pandas: For matrix math and dataset manipulation.

  ## The Workflow:

1. Data Preprocessing: We clean the human captions (removing punctuation, making everything lowercase) and wrap them in start and end tags so the model knows where sentences begin and finish.

2. Feature Extraction: We pass all 8,000+ images through InceptionV3 and save the mathematical outputs to disk so we don't have to recalculate them every time.

3. Tokenization: We build a dictionary out of the text. To keep the math clean, we filter out rare words (words appearing less than 4 times), resulting in a hyper-optimized vocabulary of 3,319 words.

4. Training: We feed the image vectors and the tokenized text sequences into the model using a custom Data Generator, running for up to 25 epochs with Early Stopping enabled to catch the exact moment of peak performance.

5. Inference (Generation): We use Greedy Search (picking the highest probability word every time) and Beam Search (exploring multiple paths to find the most logical sentence) to generate new text.

6. Visualization: We render the results in a clean dashboard.


graph TD
    %% Define Colors and Styles for a clean UI look
    classDef input fill:#f8f9fa,stroke:#ced4da,stroke-width:2px,color:#212529,rx:8px,ry:8px;
    classDef vision fill:#e3f2fd,stroke:#2196f3,stroke-width:2px,color:#0d47a1,rx:8px,ry:8px;
    classDef language fill:#e8f5e9,stroke:#4caf50,stroke-width:2px,color:#1b5e20,rx:8px,ry:8px;
    classDef merge fill:#fff3e0,stroke:#ff9800,stroke-width:2px,color:#e65100,rx:8px,ry:8px;
    classDef output fill:#fce4ec,stroke:#e91e63,stroke-width:2px,color:#880e4f,rx:8px,ry:8px;

    subgraph Vision_Pipeline ["👁️ Vision Branch (InceptionV3 Features)"]
        direction TB
        V1["Features Input<br>(Shape: 2048)"]:::input
        V2["Batch Normalization"]:::vision
        V3["Dense Layer<br>(128 Units, ReLU)"]:::vision
        V4["Batch Normalization"]:::vision
        
        V1 --> V2 --> V3 --> V4
    end

    subgraph Language_Pipeline ["📝 Language Branch (Caption Memory)"]
        direction TB
        L1["Sequence Input<br>(Shape: 34 tokens)"]:::input
        L2["Embedding Layer<br>(Vocab: 3319, Dim: 128, Masked)"]:::language
        L3["Dropout<br>(Rate: 0.3)"]:::language
        L4["LSTM Layer<br>(128 Units)"]:::language
        L5["Dropout<br>(Rate: 0.3)"]:::language
        
        L1 --> L2 --> L3 --> L4 --> L5
    end

    subgraph Decoder ["🧠 Decoder (Merger & Prediction)"]
        direction TB
        D1{"Add Layer<br>(Combines Features)"}:::merge
        D2["Dense Layer<br>(128 Units, ReLU)"]:::merge
        D3["Output Dense Layer<br>(3319 Units, Softmax)"]:::output
        D4(("Predicted<br>Next Word")):::output
        
        D1 --> D2 --> D3 --> D4
    end

    %% Routing connections between subgraphs
    V4 -->|128-dim Vector| D1
    L5 -->|128-dim Vector| D1
    
    %% Loop back arrow for the recursive generation process
    D4 -.->|Appended to text sequence<br>for next prediction step| L1


   ## 5. The Outputs: What You Will See
* Image
* The Generated Caption: What our trained AI believes is happening in the picture.
* The Original Captions: The 5 actual human reference texts, so you can immediately compare how closely the machine's thought process matches human perception.

Enjoy exploring the model! Watching a neural network successfully string together its first coherent sentence about a picture is one of the most rewarding experiences in deep learning.
