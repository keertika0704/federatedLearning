# federatedLearning

## Connect to Supabase

1. In Supabase, open **Project Settings > API** and copy the project URL and a
	publishable key (or legacy anon key).
2. Copy `.env.example` to `.env` and replace the example values with your
	project credentials. Keep `.env` private; never commit a `service_role` or
	secret key.
3. Install the Python dependencies:

	```bash
	.venv/bin/python -m pip install -r requirements.txt
	```

4. Get a client from any Python script in the project:

	```python
	from supabase_client import get_supabase_client

	supabase = get_supabase_client()
	result = supabase.table("disease_master").select("*").limit(5).execute()
	print(result.data)
	```

The key and Row Level Security policies determine which operations are allowed.
Use a publishable/anon key with appropriate RLS policies for application access.
The client setup does not create tables or upload CSV files; import the schema
and data into Supabase separately.




brew install libomp

.venv/bin/uvicorn backend.main:app --reload --port 8000