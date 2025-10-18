# NTI Advanced AI Track Projects

> This repo documents my work from the NTI Advanced AI Track - HireReady Program.

The projects within demonstrate a practical command of machine learning, from classical predictive modeling and computer vision to modern LLM-powered agents and MLOps principles. Each entry highlights a unique problem, the technical approach, and the tangible results achieved.

---

# Projects

## 1. 🩺 Cirrhosis Patient Survival Prediction

> <img src="https://github.com/user-attachments/assets/5aa13b41-c3d8-4cd7-8390-7cf8e004183d" alt="Predictive model for cirrhosis patient survival" width="300" align="right">

* **Core Technologies:** `Python`, `XGBoost`, `CatBoost`, `Optuna`, `Pandas`, `Scikit-learn`
  

* **Overview:** Developed a high-stakes predictive model to estimate survival probabilities for cirrhosis patients from clinical data. The core of the solution is a fine-tuned XGBoost model, augmented by a novel CatBoost-based imputation pipeline that intelligently handles missing values—a critical challenge in medical datasets.

* **Key Achievements & Contributions:**
    * **Secured the top score** in the competition, outperforming all other participants through rigorous hyperparameter optimization and model calibration.
    * **Engineered a custom CatBoost imputation strategy** that reduced data loss from null values by 50%, significantly improving model robustness and reliability.
    * **Delivered a fully reproducible pipeline** for feature encoding, tuning (Optuna), and probability calibration, ensuring the model's clinical interpretability.

---

## 2. 🌲 Forest Cover Type Prediction

> <img src="https://github.com/user-attachments/assets/d03c38b5-a8e4-4ff5-8bfa-490a2eccc89e" alt="AI predicting forest cover types from environmental data" width="300" align="right">

* **Core Technologies:** `PyTorch`, `Scikit-learn`, `Pandas`, `NumPy`, `Optuna`
  
* **Overview:** Executed a multi-class classification task to predict forest cover types using cartographic and environmental data. The project involved deep exploration of neural network architectures, strategic resampling to correct for severe class imbalance, and systematic hyperparameter optimization.

* **Key Achievements & Contributions:**
    * **Achieved the #1 ranking** on the competition leaderboard by systematically optimizing the neural architecture and implementing effective data balancing techniques.
    * **Maintained detailed experimentation logs**, ensuring full reproducibility and providing clear interpretability of the model's decision-making process.

---

## 3. 🎯 YOLO — Custom Object Detection Implementation from Scratch

> <img src="https://github.com/user-attachments/assets/082c69f4-a56f-4d72-868b-48e22cd6708e" alt="YOLO custom object detection showing bounding boxes and metrics" width="300" align="right">

* **Core Technologies:** `PyTorch`, `OpenCV`, `NumPy`, `Matplotlib`

* **Overview:** Built a complete YOLO object detector from the ground up to gain a foundational understanding of its mechanics. This involved implementing core components like anchor box generation, bounding box regression, and non-trivial loss functions (IoU, CIoU), along with custom evaluation methods like Intersection over Union (IoU) and Non-Maximum Suppression (NMS).

* **Key Achievements & Contributions:**
    * **Achieved competitive detection accuracy** with a custom model by architecting highly optimized training loops and data augmentation pipelines.
    * **Produced clear visual comparisons** between the from-scratch model and pretrained YOLO versions, effectively illustrating the trade-offs between model complexity, speed, and accuracy.

---

## 4. 🤖 Intelligent CV Screening Agent

> <img src="https://github.com/user-attachments/assets/8f47784f-4cd3-4366-a884-c1c9c5e6a2b2" alt="Intelligent CV screening agent analyzing resumes with Llama 3.1" width="300" align="right">

* **Core Technologies:** `Python`, `LangChain`, `Llama 3.1`, `OpenAI Embeddings`, `Streamlit`, `Pandas`

* **Overview:** Designed and built an automated, intelligent agent for evaluating job candidates. The system uses Llama 3.1 to perform semantic matching between CVs and job descriptions, assessing skill overlap, identifying competency gaps, and generating contextual feedback for dynamic acceptance or rejection emails.

* **Key Achievements & Contributions:**
    * **Significantly improved precision in skill-matching** and contextual analysis compared to traditional keyword-based methods.
    * **Automated the end-to-end screening pipeline**, delivering explainable scoring and feedback generation to streamline the hiring process.

---

## 5. 🎬 Video RAG System for YouTube Content

> <img src="https://github.com/user-attachments/assets/84c3c18e-51da-4395-bd11-c004e46f5523" alt="Video RAG system for YouTube content Q&A" width="300" align="right">

* **Core Technologies:** `LangChain`, `Whisper`, `Streamlit`, `LangGraph`, `YouTube API`, `FAISS`

* **Overview:** Architected a Retrieval-Augmented Generation (RAG) pipeline that transforms YouTube video content into a searchable, conversational knowledge base. The system automatically downloads videos, generates high-quality transcriptions with Whisper, chunks and embeds the text for efficient retrieval, and enables interactive Q&A.

* **Key Achievements & Contributions:**
    * **Built a fully automated pipeline:** from YouTube URL input to a fully interactive RAG system.
    * **Achieved sub-second query responses** for moderately sized video libraries by optimizing the FAISS vector index.
    * **Demonstrated highly accurate semantic retrieval**, capable of synthesizing answers from multiple distinct segments across a video collection.

---

## 6. 🇪🇬 Egyptian License Plate Recognition

> <img src="https://github.com/user-attachments/assets/5ca66146-462a-4ed4-993a-0bf6589716dd" alt="Egyptian License Plate Recognition system identifying a vehicle plate" width="300" align="right">

* **Core Technologies:** `OpenCV`, `TesseractOCR`, `YOLO`, `NumPy`

* **Overview:** Engineered a specialized detection and recognition system for Egyptian vehicle license plates, which present unique challenges in script, layout, and formatting. The solution integrates a YOLO-based model for robust plate localization with a fine-tuned Tesseract OCR engine for text extraction.

* **Key Achievements & Contributions:**
    * **Achieved over 95% recognition accuracy** in real-world conditions with diverse lighting, motion blur, and weather.
    * **Implemented advanced image preprocessing techniques** to normalize skewed, angled, and partially obscured plates, dramatically improving OCR performance.

---

## 7. 🗣️ Automated Video Captioning & Voiceover

> <img src="https://github.com/user-attachments/assets/ad033a49-04b4-4d04-bf38-d4f65e8e7e04" alt="Automated video captioning and voiceover generation" width="300" align="right">

* **Core Technologies:** `CLIP`, `BLIP`, `OpenCV`, `gTTS`, `Transformers`

* **Overview:** Developed a multimodal AI pipeline that automatically generates descriptive subtitles and natural-sounding audio narration directly from video input. The system uses CLIP to identify keyframes, BLIP to generate captions, and gTTS to synthesize a voiceover.

* **Key Achievements & Contributions:**
    * **Produced coherent and contextually relevant subtitles** that are automatically synchronized with video events.
    * **Delivered realistic and clear text-to-speech narration**, enhancing video accessibility for visually impaired users.

---

## 8. 😷 Real-Time Face Mask Detection

> <img src="https://github.com/user-attachments/assets/0e2bedba-3ba3-4996-adf4-a33cb5b19863" alt="Real-time face mask detection with a Swin Transformer model" width="300" align="right">

* **Core Technologies:** `Python`, `Streamlit`, `TensorFlow`, `Swin Transformer`, `OpenCV`

* **Overview:** Deployed a real-time face mask detection system powered by a fine-tuned Swin Transformer model. The application, built with a user-friendly Streamlit interface, supports both live webcam feeds and static image uploads.

* **Key Achievements & Contributions:**
    * **Attained 98% precision** on live video detection tasks, demonstrating high reliability.
    * **Utilized a compact and optimized model**, enabling smooth real-time inference on consumer-grade hardware.

---

## 9. ☀️ Sunspot Time Series Forecasting

> <img src="https://github.com/user-attachments/assets/ef604605-a749-42ae-b6a9-b76a61173796" alt="Sunspot time series forecasting with AI models" width="300" align="right">

* **Core Technologies:** `Python`, `Pandas`, `Prophet`, `ARIMA`, `Matplotlib`

* **Overview:** Conducted a comparative forecasting study on solar sunspot patterns, evaluating statistical (ARIMA) and machine learning-based (Prophet, LSTM) models. The goal was to predict long-term solar cycles from historical time-series data.

* **Key Achievements & Contributions:**
    * **Determined that Prophet produced the most stable multi-horizon forecasts** with minimal long-term drift, making it ideal for this use case.
    * **Generated clear, interpretable visualizations** of solar activity cycles, effectively communicating the model's predictions and decomposition of trends.

---

## 10. 🐦 Arabic Tweets Sentiment Analysis

> <img src="https://github.com/user-attachments/assets/eca3e610-100d-498c-a754-17bbf844bc13" alt="Arabic Tweets Sentiment Analysis with transformer models" width="300" align="right">

* **Core Technologies:** `Python`, `Scikit-learn`, `Transformers (AraBERT)`, `NLTK`, `Pandas`, `FARIS`

* **Overview:** Constructed a robust sentiment analysis pipeline for Arabic-language tweets. The project involved creating a comprehensive preprocessing framework (token normalization, diacritic removal) and comparing classical ML models against a fine-tuned AraBERT transformer.

* **Key Achievements & Contributions:**
    * **Achieved a >12% F1-score improvement** by fine-tuning AraBERT compared to the best-performing classical ML model.
    * **Established a reusable and effective preprocessing framework** specifically designed for the nuances of Arabic NLP tasks.

---

## 11. 📚 Book Information Retrieval System

> <img src="https://github.com/user-attachments/assets/0ab86106-9f1a-4c00-b988-ceb352e85652" alt="Semantic book information retrieval system" width="300" align="right">

* **Core Technologies:** `LangChain`, `Scikit-learn`, `Gradio`, `Python`

* **Overview:** Built an embedding-based retrieval engine that enables semantic question-answering over large volumes of text from book datasets. The system leverages dense vector search with intelligent document chunking for context-aware information retrieval.

* **Key Achievements & Contributions:**
    * **Delivered highly accurate retrieval** for both factual ("who did what?") and conceptual ("what is the theme of...") queries.
    * **Architected a reusable retrieval base** that can be easily adapted for broader RAG applications across different domains.

---

## 12. 📊 Sales Performance Dashboard

<img src="https://github.com/user-attachments/assets/3b2a8369-db8e-428a-941f-b0db1b2e4e61" alt="Interactive sales performance dashboard in Power BI" width="300" align="right">

* **Core Technologies:** `Power BI`, `DAX`, `SQL`

* **Overview:** Designed and deployed an interactive Power BI dashboard for analyzing sales KPIs and performance trends. The dashboard features drill-down visuals, custom tooltips, and complex DAX measures to surface actionable insights from raw sales data.

* **Key Achievements & Contributions:**
    * **Produced a publication-quality, interactive dashboard** suitable for executive-level presentations.
    * **Translated complex business questions into powerful DAX metrics**, enabling stakeholders to perform self-service analytics.


# Repository Roadmap & Project Code

This documentation serves as a high-level summary of the projects completed. This repository is actively under construction, and the full source code, datasets, and detailed documentation for each project are being prepared for public release.

> Each project will be added to its own dedicated directory in the coming weeks. Thank you for your interest.
