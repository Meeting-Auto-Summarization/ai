## Step for running Flask Web Server
### 0. Clone Repository
```
git clone https://github.com/Meeting-Auto-Summarization/ai.git
```
### 1. Install Packages that project requires
```
pip install -r requirements.txt
```
#### 1-1. Requirements
```
torch>=1.7.1
transformers>=4.3.3
flask>=2.0.3
flask_cors
gdown
docx
```
### 2. Get model from Google drive
```
$ python get_model.py
```
### 3. Run Web Server!
```
$ python app.py
```
