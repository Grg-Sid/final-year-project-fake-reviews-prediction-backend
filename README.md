# Review Analysis API

This project is a FastAPI-based application for analyzing reviews to detect authenticity and flag potential issues using machine learning.

## Prerequisites

- Python 3.12 or higher
- `pip` (Python package manager)

## Setup Instructions

### 1. Create a Virtual Environment (Only Once)

To isolate the project dependencies, create a virtual environment:

```bash
# On Unix/Linux/MacOS
python3 -m venv .venv

# On Windows
python -m venv .venv
```
### 2. Activate the Virtual Environment
```bash
# On Unix/Linux/MacOS
source .venv/bin/activate
# On Windows    
.venv\Scripts\activate
```
### 3. Install Dependencies
Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```
### 4. Run the Application  
To start the FastAPI application, run the following command:

```bash
python main.py
```
### 5. Access the API
Once the application is running, you can access the API documentation at:

```
http://127.0.0.1:8000/docs
```