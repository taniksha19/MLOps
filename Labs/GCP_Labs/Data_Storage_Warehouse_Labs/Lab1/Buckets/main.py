import os
from google.cloud import storage, bigquery
from dotenv import load_dotenv

load_dotenv()
# --- Configuration ---
# Load configuration from .env file
PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")


LOCAL_FILE = "dataset.csv"               


BQ_DATASET = "lab1_warehouse"            # The dataset we created
BQ_TABLE = "my_dataset_table"            # Table to be created
BLOB_NAME = f"data/{LOCAL_FILE}"         # The "folder" path in GCS

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the GCS bucket."""
    print(f"Connecting to GCS...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    print(f"Uploading local file {source_file_name} to gs://{bucket_name}/{destination_blob_name}...")
    blob.upload_from_filename(source_file_name)
    print("File uploaded successfully.")

    # Return the GCS URI
    return f"gs://{bucket_name}/{destination_blob_name}"


def load_gcs_to_bigquery(gcs_uri, dataset_id, table_id):
    """Loads data from a GCS file into a BigQuery table."""
    print(f"Connecting to BigQuery...")
    bq_client = bigquery.Client(project=PROJECT_ID)

    # Set the destination table
    table_ref = bq_client.dataset(dataset_id).table(table_id)

    # Configure the load job
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,  # Assuming a header row
        autodetect=True,      # Automatically detect schema
    )

    print(f"Starting BigQuery load job from {gcs_uri} to {dataset_id}.{table_id}...")
    load_job = bq_client.load_table_from_uri(
        gcs_uri, table_ref, job_config=job_config
    )

    # Wait for the job to complete
    load_job.result()  
    print(f"Job finished. Loaded data into {dataset_id}.{table_id}.")

def query_bigquery(dataset_id, table_id):
    """Runs a simple query to verify data was loaded."""
    bq_client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        SELECT COUNT(*) as row_count
        FROM `{PROJECT_ID}.{dataset_id}.{table_id}`
    """

    print(f"Running verification query: {query}")
    query_job = bq_client.query(query)

    for row in query_job.result():
        print(f"Query result: {row.row_count} rows found.")

# --- Main execution ---
if __name__ == "__main__":
    print("--- Starting Data Pipeline ---")

    # Step 1: Upload local file to GCS staging area
    gcs_data_uri = upload_to_gcs(BUCKET_NAME, LOCAL_FILE, BLOB_NAME)

    # Step 2: Load data from GCS into BigQuery warehouse
    load_gcs_to_bigquery(gcs_data_uri, BQ_DATASET, BQ_TABLE)

    # Step 3: Verify by running a query
    query_bigquery(BQ_DATASET, BQ_TABLE)

    print("--- Pipeline Finished Successfully ---")