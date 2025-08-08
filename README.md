Dual-Agent AI Pipeline for HSN Code Classification
![alt text](https://img.shields.io/badge/License-MIT-yellow.svg)

![alt text](https://img.shields.io/badge/python-3.9+-blue.svg)

![alt text](https://img.shields.io/badge/Built%20with-LangChain-green)

This project implements a robust, two-agent AI pipeline using Google's Gemini models via LangChain to automate the Harmonized System Nomenclature (HSN) classification of product descriptions from an Excel file.

The core methodology is a "Classify and Review" workflow designed to maximize accuracy and efficiency. A fast, cost-effective model performs an initial classification, and a more powerful model audits the high-confidence results, ensuring a high-quality final output while flagging ambiguous items for human review.

✨ Features
Dual-Agent System: Utilizes a two-step process for higher accuracy than a single-pass system.
Agent 1 (Classifier): A fast model (gemini-1.5-flash-latest) for high-throughput initial classification.
Agent 2 (Reviewer): A powerful model (gemini-1.5-pro-latest) for meticulous auditing of high-confidence results.
Batch Processing: Efficiently processes large Excel files by breaking them into manageable chunks.
Structured & Verifiable Output: Leverages Pydantic and JSON mode to enforce a reliable output schema.
Confidence-Based Routing: Automatically separates high-confidence, AI-reviewed results from low-confidence items that require manual inspection.
Error Handling & Resilience: Includes logic to handle API errors and output mismatches, ensuring the process continues and problematic items are logged.
Clear Audit Trail: The final output provides visibility into both the classifier's initial suggestion and the reviewer's final decision.
⚙️ How It Works: The Pipeline Logic
The process follows a clear, sequential flow managed by the main script.

code
Mermaid
sequenceDiagram
    participant User
    participant Script
    participant Classifier (Gemini Flash)
    participant Reviewer (Gemini Pro)

    User->>Script: Runs script with input Excel file
    Script->>Script: 1. Reads & Chunks Data
    loop For Each Chunk
        Script->>Classifier (Gemini Flash): 2. Sends batch of descriptions
        Classifier (Gemini Flash)-->>Script: 3. Returns initial classifications & confidence scores
        Script->>Script: 4. Filters items by confidence threshold
        Script->>Reviewer (Gemini Pro): 5. Sends high-confidence items for review
        Reviewer (Gemini Pro)-->>Script: 6. Returns final, verified classifications
        Script->>Script: 7. Aggregates results (Reviewed vs. Manual)
    end
    Script->>User: 8. Generates output CSV files
Load Data: The script loads an Excel file containing product descriptions.
Chunking: The data is split into small batches (e.g., 20 items) to ensure reliable processing and avoid API limits.
Classify (Agent 1): Each chunk is sent to the Classifier Agent. This agent quickly assigns a likely HSN code and a confidence score to every item in the batch.
Triage: The script evaluates the classifier's output.
Items with confidence above a set threshold (e.g., 80) are passed to the next stage.
Items with confidence below the threshold are immediately sent to the "manual review" list.
Review (Agent 2): The high-confidence items are batched and sent to the Reviewer Agent. This more powerful agent acts as a meticulous auditor, either confirming the suggested code or correcting it with detailed reasoning.
Output Generation: The script consolidates the results into two separate CSV files:
reviewed_classifications.csv: Contains items that were successfully classified and reviewed by the AI pipeline.
manual_review_needed.csv: Contains items that had low initial confidence or failed during processing, requiring human attention.
🚀 Getting Started
Prerequisites
Python 3.9+
An active Google AI API Key.
1. Clone the Repository
code
Bash
git clone https://github.com/your-username/hsn-classifier-pipeline.git
cd hsn-classifier-pipeline
2. Set Up a Virtual Environment
It is highly recommended to use a virtual environment.

code
Bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
Create a requirements.txt file with the following content:

code
Code
langchain
langchain-google-genai
pandas
openpyxl
python-dotenv
tqdm
ipython
jupyter
Then, install the packages:

code
Bash
pip install -r requirements.txt
4. Configure Your API Key
Create a file named .env in the root of the project directory and add your Google API key:

code
Code
GOOGLE_API_KEY="YOUR_GOOGLE_AI_API_KEY_HERE"
📋 Usage
Prepare Your Input File:
Create an Excel file (e.g., benchmark1000.xlsx) in the project's root directory.
The file must contain at least two columns: S.No. (for a unique serial number) and Description (for the product text).
Example benchmark1000.xlsx:
| S.No. | Description |
|-------|-------------------------------------------|
| 1 | 1 liter stainless steel water bottle |
| 2 | Men's 100% cotton blue denim jeans |
| 3 | Wooden dining table 4-seater |
| ... | ... |
Run the Script:
The easiest way to run the script is within a Jupyter Notebook environment, as it uses IPython.display for rich output. Open the .py file in Jupyter Lab or VS Code's notebook editor.
Alternatively, you can convert the display() calls to print() and run it as a standard Python script.
Execute the cells or run the script from top to bottom.
Check the Output:
Once the process is complete, two files will be generated in your project directory:
reviewed_classifications.csv: For successfully processed items.
| S.No. | Description | HSN_Final | Confidence_Final | Classifier_HSN | Classifier_Confidence |
|---|---|---|---|---|---|
| 1 | ... | 96170010 | 98 | 96170010 | 90 |
| 2 | ... | 62034200 | 95 | 62034200 | 85 |
manual_review_needed.csv: For items requiring human attention.
| S.No. | Description | HSN (classifier) | Confidence (classifier) |
|---|---|---|---|
| 3 | ... | 94036000 | 75 |
| ... | ... | CLASSIFICATION_ERROR | 0 |
🔧 Configuration and Customization
You can easily tweak the pipeline's behavior by modifying these key variables in the process_hsn_classification_from_excel function:

chunk_size: Change the number of items processed per batch (default: 20). Smaller chunks are more reliable but slower; larger chunks are faster but risk API timeouts.
confidence_threshold: Adjust the cutoff for sending items to the reviewer agent (default: 80).
classifier_llm / reviewer_llm: You can swap out the Gemini models for other compatible LangChain models (e.g., from OpenAI or Anthropic) by changing their initializers.
💡 Future Improvements
Implement RAG (Retrieval-Augmented Generation): Ground the models by providing them with the official HSN nomenclature and explanatory notes in a vector database. This will dramatically improve accuracy and reasoning.
Build a Simple UI: Use a framework like Streamlit or Gradio to create a user-friendly interface for uploading files and viewing results.
Advanced Feedback Loop: Create a mechanism to feed corrected classifications from the manual_review_needed.csv back into a dataset for fine-tuning a model.
Dynamic Model Selection: Implement logic to use the more powerful (and expensive) reviewer model only when the classifier's confidence is in an ambiguous range (e.g., 60-85).
🤝 Contributing
Contributions are welcome! If you have suggestions for improvements or find a bug, please feel free to open an issue or submit a pull request.

📄 License
This project is licensed under the Apache 2.0. See the LICENSE file for details.
