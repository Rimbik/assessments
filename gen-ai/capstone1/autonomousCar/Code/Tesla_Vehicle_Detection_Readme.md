
# Tesla Autonomous Vehicle Detection - YOLOv8 Project

## Part 1

[the doc can be read in conjunction with ref: https://github.com/Rimbik/ai/blob/main/capStoneToSubmit_AC/capstoneReadmeVD.pdf]

For Tesla autonomous vehicle detection, building a self-made CNN-based model from scratch is extremely challenging.  
The **first challenge** is object identification—determining **whether an object is a vehicle or not** is itself a major hurdle. Even if - if you read the part2 dataset for all accident case, will see the tesla did hit pedestrian, cyclist and even an innocent 🌴 tree. So its at this point not only vehicle.
Then comes the challenge of identifying the **type** and **number of vehicles**.

Given the objective and the annotation `.csv` provided with the assignment, the approach chosen was:

> **Use a pre-built (semi/full) model and customize it using the dataset**, which aligns with the assignment's purpose.

Among many available models and technologies, we chose the **YOLOv8** model to train on our custom images and dataset.

---

## Implementation Steps

### Step 1: What We Have

- 📷 All the training images  
- 📝 Their annotation CSV file

---

### Step 2: Why We Didn't Use the Provided CSV

The existing `.csv` annotation was found to be **faulty**:
- Many entries in the CSV had **filenames without corresponding images**.
- For instance, the image `00000009.jpg` should have an annotation entry labeled `"00000009"`, but this consistency was not found.

> ✅ **Decision:** Create a **fresh annotation CSV** and generate **YOLOv8-compatible YAML** for training data.

---

### Step 3: Image Uploads

Due to GitHub's file size limits, images were uploaded in `.rar` format (~19MB each).

- 🗂 All images were uploaded to GitHub in split `.rar` format
- 📥 Downloaded and extracted via code to recreate the complete image dataset locally

---

### Step 4: Annotation Preparation

- 📝 Created a **fresh annotation CSV**
- 🔁 Converted CSV to **YOLOv8 YAML** format for training
- 🛠️ Used manual tools to rectify annotation coordinates when required

---

### Step 5: Model Training

- 🔁 Trained the YOLOv8 model with:
  - **10 epochs**
  - **Early stopping**
  - 📦 Saved model as `.pt` file for future inference

---

### Step 6: Inference

Ran inference on the saved `.pt` model to achieve:
- 🚗 **Vehicle detection with bounding boxes**
- 🔢 **Vehicle count on roads**

---

## GitHub Repository

📂 The entire codebase is available publicly on GitHub for:
- Easier access
- Further learning and experimentation
https://github.com/Rimbik/assessments/blob/main/gen-ai/capstone1/autonomousCar/Code/Part1/autonomouisCar_july6.ipynb
---

## Project Directory Structure
[all are autogenerated on code exucution in google colab]
```plaintext
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/

multifiles/
├── annot/
│   └── annotation.csv
├── extractedImg/
│   ├── Images/
│   │   └── *.jpg
│   ├── labels/
│   │   └── *.txt
│   ├── Images.part1.rar
│   ├── Images.part2.rar
│   └── Images.part*.rar
```
-----
The above is based on google colab code, a linux based repo will be shared shortly that runs on local machine with local resources using BSD xml annotation.

Thank you for reading.
eol
