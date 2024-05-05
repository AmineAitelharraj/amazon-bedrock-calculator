# Generative AI Model Cost Calculator

A Streamlit application designed to estimate costs using the Amazon Bedrock cost calculator, supporting a wide range of
model providers, with the goal of benchmarking model costs to help users select the most appropriate one.

## Features

- Supports multiple model providers: Anthropic, Mistral AI, Cohere, and Meta Llama
- Calculates costs based on model parameters, number of queries per day, and language-specific settings
- Supports different application types: RAG Application and Chatbot Application
- Allows users to input infrastructure costs for Amazon OpenSearch Serverless and Amazon DynamoDB (for RAG Application)
- Displays calculated costs per day, per month, and per year
- Allows users to remove individual rows from the results table
- Provides a clear button to remove all results
- Supports multiple languages and allows users to set the ratio of questions in each language

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/generative-ai-cost-calculator.git
```

2. Change to the project directory:

```bash
cd generative-ai-cost-calculator
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:

```bash
streamlit run app.py
```

2. Access the application in your web browser at the provided URL (usually `http://localhost:8501`)

3. Use the sidebar to select the model provider, model, and input the required parameters

4. Select the languages you want to support and set the ratio of questions in each language. The total ratio should add
   up to 100%

5. Click the "Calculate Costs" button to generate the cost estimation

6. Review the results in the main panel, and remove individual rows or clear all results as needed

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Contact

Email : a.aitelharraj@reply.com