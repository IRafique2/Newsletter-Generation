# **Newsletter Generator Application**

!
This project is a **Streamlit-based** web application that allows users to generate newsletters by providing a topic. The application performs a series of tasks, such as searching the web for related articles, summarizing the content, and compiling the information into a professional newsletter format. This is useful for content creators, marketers, researchers, and anyone who needs automated newsletter creation.

## **Features**

* **User Input**: Accepts user input for a specific topic.
* **Web Scraping**: Searches the web for articles related to the user-provided topic.
* **Article Summarization**: Summarizes the articles extracted from the search results.
* **Newsletter Generation**: Generates a professional newsletter using the summaries.
* **Data Presentation**: Displays search results, extracted data, and summaries in an organized format.

## **Technologies Used**

* **Streamlit**: Web application framework for creating the user interface.
* **Langchain**: Chain of LLM (Large Language Model) components to process and manipulate data.
* **OpenAI GPT Models**: Used for summarization, content generation, and best URL selection.
* **FAISS**: Used for similarity search on extracted documents.
* **HuggingFace Embeddings**: To embed the textual content for similarity search.
* **Unstructured URL Loader**: To load and extract content from URLs.

## **Setup Instructions**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/newsletter-generator.git
   cd newsletter-generator
   ```

2. **Set up environment variables:**

   * Create a `.env` file in the project root directory.
   * Add your **SERPER_API_KEY** for Google Search results.

   Example:

   ```bash
   SERPER_API_KEY=your-serper-api-key
   ```

3. **Run the Streamlit app:**

   Once the dependencies are installed and environment variables are set up, run the app using:

   ```bash
   streamlit run app.py
   ```

   This will start the Streamlit app locally, and you can access it by visiting `http://localhost:8501` in your browser.

## **How It Works**

1. **Search for Articles**:

   * When a user inputs a topic, the app uses Google’s Serper API to search for relevant articles. It returns a list of URLs based on the search results.

2. **Pick Best Articles**:

   * The best articles are selected from the search results using the `pick_best_articles_urls` function, which leverages a language model to filter the most relevant URLs.

3. **Extract Content**:

   * Content from these URLs is extracted using the `UnstructuredURLLoader` and split into manageable chunks using a `CharacterTextSplitter`. The content is then stored in a FAISS vector store for similarity search.

4. **Summarize Content**:

   * The selected content is summarized by the app. The summaries are tailored based on the user’s query and are generated using an LLM.

5. **Generate Newsletter**:

   * Finally, the summaries are compiled into a structured format for the newsletter, which is displayed to the user. The newsletter includes:

     * Search results
     * Best URLs
     * Data from articles
     * Summaries
     * The final newsletter content

## **File Structure**

```
.
├── app.py              # Main Streamlit application
├── helper.py           # Helper functions for data extraction, summarization, etc.
├── .env                # Environment variables (e.g., SERPER_API_KEY)
└── README.md           # This README file
```

## **Dependencies**

The project needs the necessary Python libraries:

```txt
streamlit
langchain
openai
requests
huggingface-hub
faiss-cpu
python-dotenv
pypdf
```

## **Environment Variables**

This project requires the following environment variable to run:

* **SERPER_API_KEY**: API key for Google Serper API to fetch search results.

---

Let me know if you need further modifications or details!
