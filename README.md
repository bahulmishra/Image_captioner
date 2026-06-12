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

  <img width="2473" height="1254" alt="trainVsvalloss" src="https://github.com/user-attachments/assets/f8b7cc85-4e5c-4c7a-99ce-8206669352e6" />


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

<img width="1505" height="6098" alt="image_captioner_architercture" src="https://github.com/user-attachments/assets/0105c637-afc4-448a-82a3-44423d8e8bd5" />



   ## 5. The Outputs: What You Will See
* Image
* The Generated Caption: What our trained AI believes is happening in the picture.
* The Original Captions: The 5 actual human reference texts, so you can immediately compare how closely the machine's thought process matches human perception.

Enjoy exploring the model! Watching a neural network successfully string together its first coherent sentence about a picture is one of the most rewarding experiences in deep learning.

Sample Outputs:
<img width="1363" height="453" alt="Screenshot 2026-06-12 133759" src="https://github.com/user-attachments/assets/4d62bf8c-8be8-4e77-a501-8b2bc38b2aff" />
<img width="1195" height="422" alt="Screenshot 2026-06-12 133357" src="https://github.com/user-attachments/assets/4e07e0bf-e7fb-4dd6-8c8d-02383e34da1e" />
<img width="912" height="372" alt="Screenshot 2026-06-12 131312" src="https://github.com/user-attachments/assets/0f6324a7-3fde-4eb8-906f-adeb7494a76d" />
<img width="1566" height="397" alt="Screenshot 2026-06-12 133714" src="https://github.com/user-attachments/assets/a8b14574-04b0-4440-a94d-c309cacb9933" />







