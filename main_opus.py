import streamlit as st
import pandas as pd
import math


def calculate_costs(model, context_window, prompt_dollars, completion_dollars, queries_per_day, language_data,
                    prompt_context_words, app_type, opensearch_cost, dynamodb_cost):
    total_prompt_tokens = prompt_context_words * 1.33
    total_completion_tokens = 0

    for lang in language_data:
        prompt_tokens = lang['prompt_words'] * lang['tokenizer']
        total_prompt_tokens += prompt_tokens * lang['ratio'] / 100

        response_tokens = lang['response_words'] * lang['tokenizer']
        total_completion_tokens += response_tokens * lang['ratio'] / 100

    estimated_tokens_per_prompt = total_prompt_tokens
    estimated_tokens_per_completion = total_completion_tokens
    total_tokens_per_day = (estimated_tokens_per_prompt + estimated_tokens_per_completion) * queries_per_day
    cost_per_day = round((prompt_dollars / 1_000_000 * estimated_tokens_per_prompt * queries_per_day) + (
            completion_dollars / 1_000_000 * estimated_tokens_per_completion * queries_per_day), 2)
    cost_per_month = round(cost_per_day * 30, 2)
    cost_per_year = round(cost_per_month * 12, 2)

    input_dollars = round(prompt_dollars / 1_000_000 * estimated_tokens_per_prompt * queries_per_day, 2)
    output_dollars = round(completion_dollars / 1_000_000 * estimated_tokens_per_completion * queries_per_day, 2)

    if app_type == 'RAG Application':
        cost_per_day = round(cost_per_day + opensearch_cost + dynamodb_cost, 2)
        cost_per_month = round(cost_per_month + (opensearch_cost + dynamodb_cost) * 30, 2)
        cost_per_year = round(cost_per_year + (opensearch_cost + dynamodb_cost) * 365, 2)

    return [model, format_number(context_window), f"${prompt_dollars}", f"${completion_dollars}", queries_per_day,
            round(estimated_tokens_per_prompt, 2), round(estimated_tokens_per_completion, 2), f"${input_dollars}",
            f"${output_dollars}", f"${cost_per_day}", f"${cost_per_month}", f"${cost_per_year}"]


def format_number(num):
    if num >= 1000000:
        return f"{num // 1000000}M"
    elif num >= 1000:
        return f"{num // 1000}K"
    else:
        return str(num)


def main():
    st.set_page_config(
        page_title="Amazon Bedrock Calculator",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.reply.com/data-reply/fr/',
            'Report a bug': "https://www.github.com/",
            'About': "A Streamlit application designed to estimate costs using the Amazon Bedrock cost calculator,"
                     "supporting a wide range of model providers, with the goal of benchmarking model costs "
                     "to help users select the most appropriate one."
        }
    )
    st.title('Generative AI Model Cost Calculator')

    app_type = st.sidebar.selectbox('Select the type of Generative AI application',
                                    ['RAG Application', 'Chatbot Application'])

    st.sidebar.header('Model Parameters')

    model_details = {
        'Anthropic': {
            'Claude 3 Opus': {'Context Window': 200000, 'Prompt ($/1M)': 15.0, 'Completion ($/1M)': 75.0},
            'Claude 3 Sonnet': {'Context Window': 200000, 'Prompt ($/1M)': 3.0, 'Completion ($/1M)': 15.0},
            'Claude 3 Haiku': {'Context Window': 200000, 'Prompt ($/1M)': 0.25, 'Completion ($/1M)': 1.25},
            'Claude 2.1': {'Context Window': 200000, 'Prompt ($/1M)': 8.0, 'Completion ($/1M)': 24.0},
            'Claude 2.0': {'Context Window': 100000, 'Prompt ($/1M)': 8.0, 'Completion ($/1M)': 24.0},
            'Claude Instant': {'Context Window': 100000, 'Prompt ($/1M)': 0.8, 'Completion ($/1M)': 2.4},
        },
        'Mistral AI': {
            'Mistral 7B': {'Context Window': 8000, 'Prompt ($/1M)': 0.15, 'Completion ($/1M)': 0.2},
            'Mixtral 8*7B': {'Context Window': 32000, 'Prompt ($/1M)': 0.45, 'Completion ($/1M)': 0.7},
            'Mistral Large': {'Context Window': 32000, 'Prompt ($/1M)': 8.0, 'Completion ($/1M)': 24.0},
        },
        'Cohere': {
            'Command': {'Context Window': 4000, 'Prompt ($/1M)': 1.5, 'Completion ($/1M)': 2.0},
            'Command Light': {'Context Window': 4000, 'Prompt ($/1M)': 0.3, 'Completion ($/1M)': 0.6},
            'Command R+': {'Context Window': 128000, 'Prompt ($/1M)': 3.0, 'Completion ($/1M)': 15.0},
            'Command R': {'Context Window': 128000, 'Prompt ($/1M)': 0.5, 'Completion ($/1M)': 1.5},
        },
        'Meta Llama': {
            'Llama 2 13B': {'Context Window': 4000, 'Prompt ($/1M)': 0.75, 'Completion ($/1M)': 1.0},
            'Llama 2 70B': {'Context Window': 4000, 'Prompt ($/1M)': 1.95, 'Completion ($/1M)': 2.56},
            'Llama 3 8B': {'Context Window': 8000, 'Prompt ($/1M)': 0.4, 'Completion ($/1M)': 0.6},
            'Llama 3 70B': {'Context Window': 8000, 'Prompt ($/1M)': 2.65, 'Completion ($/1M)': 3.5},
        },
    }

    provider = st.sidebar.selectbox('Select Model Provider', list(model_details.keys()))
    model = st.sidebar.selectbox('Select Model', list(model_details[provider].keys()))

    show_details = st.sidebar.checkbox('Show Model Details', value=False)

    if show_details:
        details = model_details[provider][model]
        st.sidebar.write(f"**Model Details:**")
        st.sidebar.write(f"Context Window (aka token limit): {format_number(details['Context Window'])}")
        st.sidebar.write(f"Prompt ($/1M): ${details['Prompt ($/1M)']}")
        st.sidebar.write(f"Completion ($/1M): ${details['Completion ($/1M)']}")

    context_window = model_details[provider][model]['Context Window']
    prompt_dollars = model_details[provider][model]['Prompt ($/1M)']
    completion_dollars = model_details[provider][model]['Completion ($/1M)']
    queries_per_day = st.sidebar.number_input('Number of queries per day', value=500)

    language_data = []
    total_ratio = 0

    languages = ['English', 'Spanish', 'French']
    tokenizers = {'English': 1.33, 'Spanish': 2.00, 'French': 2.00}
    output_factors = {'English': 3, 'Spanish': 2, 'French': 2}

    selected_languages = st.sidebar.multiselect('Select languages', languages)

    for language in selected_languages:
        st.sidebar.write(f"The tokenizer ratio is {tokenizers[language]}")
        prompt_words = st.sidebar.number_input(f"Length of question in {language} (words)", value=75,
                                               key=f"prompt_words_{language}")
        prompt_tokens = math.ceil(prompt_words * tokenizers[language])
        st.sidebar.write(
            f"Number of tokens for {language} question: {prompt_tokens}")

        response_words = st.sidebar.number_input(f"Length of response in {language} (words)", value=75,
                                                 key=f"response_words_{language}")
        response_tokens = math.ceil(response_words * tokenizers[language])
        st.sidebar.write(
            f"Number of tokens for {language} response: {response_tokens}")

        tokenizer = tokenizers[language]

        output_factor = st.sidebar.number_input(f"Output factor for {language}", value=output_factors[language],
                                                key=f"output_factor_{language}")

        ratio = st.sidebar.number_input(f"Ratio of questions in {language} (%)", min_value=0.0, max_value=100.0,
                                        value=0.0, key=f"ratio_{language}")

        total_ratio += ratio
        language_data.append({"language": language, "prompt_words": prompt_words, "tokenizer": tokenizer,
                              "response_words": response_words, "ratio": ratio, "output_factor": output_factor})

    if total_ratio != 100:
        st.sidebar.warning(f"The total ratio of questions should be 100%. Current total: {total_ratio}%")

    prompt_context_words = st.sidebar.number_input('Prompt Context (words) in English', value=200)
    prompt_context_tokens = math.ceil(prompt_context_words * 1.33)
    st.sidebar.write(f"Number of tokens for prompt context: {prompt_context_tokens}")

    if app_type == 'RAG Application':
        st.sidebar.header('Infrastructure Needs')
        opensearch_cost = st.sidebar.number_input('Amazon OpenSearch Serverless cost per day ($)', value=0.0)
        dynamodb_cost = st.sidebar.number_input('Amazon DynamoDB cost per day ($)', value=0.0)

    else:
        opensearch_cost = 0.0
        dynamodb_cost = 0.0

    if 'results' not in st.session_state:
        st.session_state.results = []

    if st.sidebar.button('Calculate Costs') and total_ratio == 100:
        result_row = calculate_costs(model, context_window, prompt_dollars, completion_dollars, queries_per_day,
                                     language_data, prompt_context_words, app_type, opensearch_cost,
                                     dynamodb_cost)
        st.session_state.results.append(result_row)

    if 'results' in st.session_state and st.session_state.results:
        results_df = pd.DataFrame(st.session_state.results,
                                  columns=['Model', 'Context Window\n(aka token limit)', 'Prompt ($/1M)',
                                           'Completion ($/1M)', 'Number of queries/day',
                                           '#tokens in prompt (input)',
                                           '#tokens in completions (output)', 'Input $', 'Output $',
                                           'Cost per day', 'Cost per month', 'Cost per year'])
        results_df.index.name = 'ID'
        results_df.reset_index(inplace=True)

        st.dataframe(results_df, use_container_width=False)

        selected_id = st.selectbox('Select row to remove', options=results_df['ID'])

        if st.button('Remove Selected Row'):
            new_results = [row for i, row in enumerate(st.session_state.results) if
                           i != results_df[results_df['ID'] == selected_id].index[0]]
            st.session_state.results = new_results

    if st.sidebar.button('Clear Results'):
        st.session_state.results = []


if __name__ == '__main__':
    main()
