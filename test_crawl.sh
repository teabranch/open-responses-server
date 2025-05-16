#!/bin/bash

# --- 1. Submit the crawl job ---

echo "Submitting crawl job..."

# Send the POST request, store the JSON response in a variable
# -s: Silent mode (no progress meter)
# -X POST: Specify POST method
# -H "Content-Type: application/json": Set the Content-Type header
# -d '{...}': Provide the JSON data payload
RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"urls": "https://en.wikipedia.org/wiki/Web_crawler", "priority": 10, "session_id": "13"}' \
  http://localhost:11235/crawl)

# Check if curl command was successful and response is not empty
if [ $? -ne 0 ] || [ -z "$RESPONSE" ]; then
  echo "Error: Failed to submit crawl job or received empty response."
  exit 1
fi

# Extract the task_id from the JSON response using jq
# -r: Raw output (removes quotes from the string)
TASK_ID=$(echo "$RESPONSE" | jq -r '.task_id')

# Check if TASK_ID was extracted successfully (jq returns null if not found)
if [ "$TASK_ID" == "null" ] || [ -z "$TASK_ID" ]; then
    echo "Error: Could not extract task_id from response:"
    echo "$RESPONSE"
    exit 1
fi

echo "Job submitted successfully. TASK_ID: $TASK_ID"
echo "---"


# --- 2. Poll for task completion ---

STATUS=""
RESULT_JSON=""

# Loop until the status is "completed"
while [ "$STATUS" != "completed" ]; do
    echo "Checking status for task $TASK_ID..."

    # Send the GET request
    # Note: Using double quotes around the URL allows variable expansion ($TASK_ID)
    RESULT_JSON=$(curl -s "http://localhost:11235/task/$TASK_ID")

    # Check if curl command was successful and response is not empty
    if [ $? -ne 0 ] || [ -z "$RESULT_JSON" ]; then
      echo "Warning: Failed to get task status or received empty response. Retrying in 5 seconds..."
      sleep 5
      continue # Skip to the next iteration
    fi

    # Extract the status from the JSON response
    # Assuming the field name is 'status'. Adjust if necessary.
    STATUS=$(echo "$RESULT_JSON" | jq -r '.status')

    # Check if status was extracted
    if [ "$STATUS" == "null" ] || [ -z "$STATUS" ]; then
        echo "Warning: Could not determine status from response. Retrying in 5 seconds..."
        echo "Response was:"
        echo "$RESULT_JSON"
        sleep 5
        continue
    fi

    echo "Current status: $STATUS"

    # If not completed, wait before polling again
    if [ "$STATUS" != "completed" ]; then
        sleep 5 # Wait for 5 seconds
    fi
done

echo "---"
echo "Task $TASK_ID completed!"
echo "Final Result:"
# Pretty-print the final JSON result using jq
echo "$RESULT_JSON" | jq .

exit 0