"""Quick smoke test to verify AWS Bedrock connectivity."""
import os
from dotenv import load_dotenv
import boto3
import json

load_dotenv()

print("Loading AWS credentials from .env...")
print(f"Region: {os.getenv('AWS_DEFAULT_REGION')}")
print(f"Generation Model: {os.getenv('BEDROCK_GENERATION_MODEL')}")
print(f"Embedding Model: {os.getenv('BEDROCK_EMBEDDING_MODEL')}")
print()

# Initialize Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
)

# --- Test 1: Claude generation ---
print("Test 1: Calling Claude 3.5 Haiku...")
try:
    response = bedrock.invoke_model(
        modelId=os.getenv("BEDROCK_GENERATION_MODEL"),
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Say 'AskAWS RAG project setup is complete!' in exactly those words."}
            ]
        })
    )
    result = json.loads(response["body"].read())
    print(f"   SUCCESS - Claude response: {result['content'][0]['text']}")
except Exception as e:
    print(f"   FAILED - Claude error: {e}")

# --- Test 2: Titan embeddings ---
print()
print("Test 2: Generating embedding with Titan...")
try:
    response = bedrock.invoke_model(
        modelId=os.getenv("BEDROCK_EMBEDDING_MODEL"),
        body=json.dumps({"inputText": "Amazon S3 is an object storage service."})
    )
    result = json.loads(response["body"].read())
    embedding = result["embedding"]
    print(f"   SUCCESS - Embedding dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
except Exception as e:
    print(f"   FAILED - Embedding error: {e}")

print()
print("=" * 60)
print("Phase 1 setup complete if both tests passed!")
print("=" * 60)