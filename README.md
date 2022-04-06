## Step for running Flask Web Server
### 0. Clone Repository
```
git clone https://github.com/Meeting-Auto-Summarization/ai.git
cd ai
```
### 1. Install Packages that project requires
```
pip install -r requirements.txt
```
#### 1-1. Requirements
```
torch>=1.7.1
transformers>=4.3.3
fastapi
uvicorn
python-docx
kiwipiepy
sentence_transformers
```
### 2. Run Web Server!
```
uvicorn main:app --reload --host=0.0.0.0
```
