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
torch==1.11.0
transformers==4.17.0
sentence_transformers==2.2.0
fastapi==0.75.0
uvicorn==0.17.6
python-docx
kiwipiepy==0.11.1
ray==1.11.0
```
### 2. Run Web Server!
```
uvicorn main:app --reload --host=0.0.0.0
```
