
# 🛡️ PhishGuard AI

A machine-learning web app that checks whether a link looks like phishing, using only the text of the URL. Paste a link and get a verdict, a risk meter, the real domain behind the address, and the reasons for the result.

**Live demo:** https://two520030240-ml-s4.onrender.com
*Hosted on Render's free tier. If the page has been idle, it can take about a minute to wake up.*

---

## Features

* **URL scanner** with a four-level verdict:

  * Likely phishing
  * Probably phishing
  * Uncertain
  * Looks legitimate
* **Risk meter** showing the estimated level of risk
* **URL breakdown** that separates the scheme, subdomain, and real domain
* **Plain-language findings**, such as:

  * No HTTPS
  * Raw IP address
  * Risky domain ending
  * Sensitive keywords
  * Many subdomains
  * Suspicious URL characteristics
* **Brand impersonation check** for links that mention well-known brands on a domain that does not belong to them
* **Typosquat finder** that generates look-alike domain variants, including:

  * Missing letters
  * Character swaps
  * `0` for `o`
  * Alternative domain endings
* **Session statistics and scan history**
* Dark cybersecurity interface using glassmorphism and neumorphism

---

## How It Works

The model never visits or opens the link. It analyzes only the **text of the URL** and converts it into **10,239 features**.

| Feature Group        |      Count | Description                                                                                                    |
| -------------------- | ---------: | -------------------------------------------------------------------------------------------------------------- |
| Numeric URL features |         10 | Length, dots, HTTPS, IP address, path depth, parameters, suspicious words, special characters, digits, entropy |
| Top-level domain     |        229 | One-hot encoded TLD such as `.com`, `.tk`, etc.                                                                |
| Character TF-IDF     |     10,000 | Character-pattern frequencies extracted from the URL text                                                      |
| **Total**            | **10,239** | Combined feature representation                                                                                |

These features are used by two machine-learning models:

### Random Forest

* 300 trees
* `class_weight="balanced"`
* `random_state=42`

### Logistic Regression

* `liblinear` solver
* `class_weight="balanced"`

Both models were trained and evaluated for comparison.

---

## Dataset

The original dataset contains **160,064 labelled URLs** with 13 columns, including `url`, `label`, and engineered features.

The dataset was highly imbalanced:

* **159,244 phishing URLs**
* **820 legitimate URLs**
* Approximately **99.5% phishing URLs**

To address this imbalance, a balanced dataset containing **1,640 URLs** was created:

* 820 phishing URLs
* 820 legitimate URLs

The balanced dataset was split into:

* **1,312 training URLs**
* **328 testing URLs**

---

## Results and Limitations

Please read this section before relying on the model.

* Both models achieved **100% accuracy on the held-out 328-URL test split**.
* A separate **15-URL real-world check** was also performed using 10 legitimate sites and 5 phishing URLs.
* On this external check:

  * Logistic Regression classified all 15 URLs correctly.
  * Random Forest achieved **33% accuracy** and incorrectly flagged several legitimate websites, including `google.com` and `github.com`.
* The difference between the balanced test results and the external evaluation indicates that the models may have learned patterns specific to the training dataset rather than general phishing behaviour.
* The legitimate class in the original dataset contains only 820 URLs, so the model has relatively few examples of normal websites.
* The model analyzes only the URL text. It does **not** check:

  * Web page content
  * Domain age
  * SSL certificate details
  * Website reputation
  * Hosting information
* Results should therefore be treated as a **risk estimate, not a guarantee**.
* For important websites, verify the URL through official channels rather than relying solely on the model.

### Planned Improvements

Future improvements include:

* A larger and more diverse legitimate URL dataset
* A larger external evaluation dataset
* Better validation against unseen domains
* Re-evaluation of which trained model should be deployed
* Additional URL and domain-level security features

---

## Project Structure

The main application and trained model files are stored in the **`project code/`** folder.

```text
2520030240_ML_S4/
│
├── project code/
│   ├── app.py
│   ├── balanced_logistic_model.pkl
│   ├── balanced_rf_model.pkl
│   ├── balanced_tfidf_vectorizer.pkl
│   ├── balanced_tld_encoder.pkl
│   ├── ml_project.ipynb
│   ├── model_metadata.json
│   ├── numeric_features.pkl
│   └── requirements.txt
│
├── project/
│   ├── README.md
│   ├── ML PROJECT DATASET.zip
│   ├── ML_Project PPT(1).pdf
│   ├── Abstract_template - ML_FINAL_2.002.pdf
│   ├── Phishing_Literature_Review_18_Papers.xlsx
│   └── eda_ml_project_ipynb.ipynb
│
├── practical/
├── skill/
├── README.md
└── week1_eda.ipynb
```

### Main Application Files

| File                            | Purpose                                                               |
| ------------------------------- | --------------------------------------------------------------------- |
| `app.py`                        | Gradio web application, prediction, URL analysis and typosquat finder |
| `balanced_rf_model.pkl`         | Trained Random Forest model                                           |
| `balanced_logistic_model.pkl`   | Trained Logistic Regression model                                     |
| `balanced_tfidf_vectorizer.pkl` | Character-level TF-IDF vectorizer                                     |
| `balanced_tld_encoder.pkl`      | TLD encoder                                                           |
| `numeric_features.pkl`          | Numeric feature names and feature order                               |
| `model_metadata.json`           | Model and feature metadata                                            |
| `ml_project.ipynb`              | Machine-learning training and evaluation notebook                     |
| `requirements.txt`              | Python dependencies                                                   |

---

## Run It Yourself

### Local Setup

Clone the repository:

```bash
git clone https://github.com/noorulhuda07/2520030240_ML_S4.git
```

Navigate to the application folder:

```bash
cd 2520030240_ML_S4/project code
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

For Mac/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Then open:

```text
http://localhost:7860
```

---

## Google Colab

Upload the files from the `project code/` folder to a Colab environment.

Place the application and model files inside:

```text
/content/phishguard_deploy
```

Then run:

```python
import app as pg
pg.app.launch(share=True)
```

---

## Deployment

The application is deployed using **Render**.

### Render Configuration

**Root Directory:**

```text
project code
```

**Build Command:**

```bash
pip install -r requirements.txt
```

**Start Command:**

```bash
python app.py
```

**Environment Variable:**

```text
PYTHON_VERSION=3.12.8
```

The deployed application is available at:

**https://two520030240-ml-s4.onrender.com**

---

## Tech Stack

* Python
* scikit-learn
* pandas
* NumPy
* SciPy
* joblib
* Gradio
* Character-level TF-IDF
* Random Forest
* Logistic Regression
* Render

---

## Author

**noorulhuda07**

Machine Learning Project

