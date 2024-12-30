import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";

// Create a Bedrock Runtime client in the specified region
const client = new BedrockRuntimeClient({ region: "us-east-1" });

// Specify the model ID for Amazon Titan Text Premier
const modelId = "amazon.titan-text-premier-v1:0";

/**
 * Lambda handler function
 * @param {Object} event - The AWS Lambda event object (includes request data)
 * @returns {Object} - HTTP response with status code and JSON body
 */
export const handler = async (event) => {
  try {
    // 1. Parse the event body to extract user_query
    //    - If using Lambda Proxy Integration, 'event.body' is often a JSON string
    //    - If 'event.body' is undefined, fallback to checking 'event.user_query' directly
    const body = typeof event.body === "string" ? JSON.parse(event.body) : event.body || {};
    const userQuery = body.user_query || event.user_query;

    // 2. Validate user input
    if (!userQuery) {
      return {
        statusCode: 400,
        body: JSON.stringify({ error: "Missing 'user_query' in request." }),
      };
    }

    // 3. Prepare the invocation payload for Bedrock
    const input = {
      modelId,                              // The foundation model ID
      contentType: "application/json",      // Sending JSON input
      accept: "application/json",           // Expecting JSON output
      body: JSON.stringify({
        inputText: userQuery,               // The text prompt from the user
        textGenerationConfig: {
          maxTokenCount: 3072,             // Max tokens in the response
          stopSequences: [],
          temperature: 0.7,                // Controls randomness
          topP: 0.9,                       // Controls nucleus sampling
        },
      }),
    };

    // Log the payload for debugging/tracking
    console.log("Payload Sent:", JSON.stringify(input, null, 2));

    // 4. Send the command to Bedrock Runtime
    const command = new InvokeModelCommand(input);
    const response = await client.send(command);

    // 5. Convert the response body from a Buffer to a string, then parse as JSON
    const responseBody = Buffer.from(response.body).toString("utf-8");
    const parsedResponse = JSON.parse(responseBody);

    // 6. Check if the model filtered the response (content policy triggered)
    if (parsedResponse.results?.[0]?.completionReason === "CONTENT_FILTERED") {
      console.warn("Response was filtered by the model.");
    }

    // 7. Return a successful response with the query and the entire model output
    return {
      statusCode: 200,
      body: JSON.stringify({
        query: userQuery,
        generated_response: parsedResponse,
      }),
    };

  } catch (error) {
    // Log any error that occurs during processing
    console.error("Error querying model:", error);

    // Return a 500 error with details
    return {
      statusCode: 500,
      body: JSON.stringify({
        error: "Failed to process request",
        details: error.message,
      }),
    };
  }
};
