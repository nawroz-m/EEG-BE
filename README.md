## Setup

1. Create a virtual environment

```bash
python -m venv venv
```

2. Activate the virtual environment
   On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the server:

```bash
python app.py
or
uvicorn app:app --host 0.0.0.0 --port 5001 --reload
```

The server will run on `http://localhost:5001`

### GET `/health`

Health check endpoint.

**Response:**

```json
{
  "status": "ok"
}
```

## Notes

- CORS enabled for frontend integration
