# Automation Extension Backend

This is a FastAPI server backend for the Automation Extension that processes natural language requests and returns automation action sequences.

## Setup

1. Create a virtual environment (recommended):
```
python -m venv venv
```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
```
pip install -r requirements.txt
```

## Running the Server

Start the FastAPI server:
```
python app.py
```

This will start the server at `http://127.0.0.1:5000`. 

Alternatively, you can use uvicorn directly:
```
uvicorn app:app --reload --port 5000
```

## API Endpoints

- `POST /process` - Process natural language commands and return automation actions
- `GET /health` - Health check endpoint

## Documentation

Once the server is running, access the interactive API documentation at:
- Swagger UI: `http://127.0.0.1:5000/docs`
- ReDoc: `http://127.0.0.1:5000/redoc`

## Development

To implement your own natural language processing logic:
1. Find the `process_request` function in `app.py`
2. Replace the dummy data implementation with your custom logic
3. The response should follow the structure defined in `dummy_data.json`

## Testing

You can test the API using curl:
```
curl -X POST "http://127.0.0.1:5000/process" -H "Content-Type: application/json" -d '{"query":"Play Veritasium newest video and like it"}'
```

Or using the browser extension directly by entering a natural language command.