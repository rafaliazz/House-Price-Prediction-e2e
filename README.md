# House Price Prediction End 2 End project 
This is a simple Python MLOps end-to-end project that solves the classical Ames Housing regression task and integrates it into a fully runnable application with:

* A working FastAPI backend

* A Streamlit UI frontend

* MLflow experiment tracking

* Docker & Docker Compose support

The goal of this project is to demonstrate a complete ML lifecycle:
data → training → experiment tracking → API → UI → containerization.

# Instructions

There are 2 main components you can run and demonstrate:

## Run the API and Streamlit App
**Step 1 — Run the FastAPI Backend**

Navigate to the project root directory and run:

```uvicorn src.api.main:app --reload```

The API will start at:

http://127.0.0.1:8000

You can test it using:

http://127.0.0.1:8000/docs

**Step 2 — Run the Streamlit Frontend**

In a new terminal, run:

```streamlit run src/app.py```

The Streamlit app will open automatically in your browser

## Run with Docker & Docker Compose

This project includes:

Dockerfile

docker-compose.yaml

**Step 1 — Build Containers**

```docker-compose build```

**Step 2 — Run Containers**

```docker-compose up```

Or detached mode:

```docker-compose up -d```

This will:

Run three services:- 

* Run the API 
* Run the Streamlit UI 
* Run the house monitoring service (checks for new data presence by checking for instances of dataset/new_data.csv and then checks for data drift)

To stop:

```docker-compose down```