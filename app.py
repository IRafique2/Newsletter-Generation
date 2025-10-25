import os
import streamlit as st
from helper import *
# Define the main function
def main():
    # Set Streamlit page configuration
    st.set_page_config(page_title="Researcher...", page_icon=":parrot:", layout="wide")

    # Display a header in the app
    st.header("Generate a Newsletter :parrot:")

    # Input field for the user to enter a topic
    query = st.text_input("Enter a topic...")

    # Check if the user has entered a query
    if query:
        # Print the query to the console
        print(query)
        
        # Display the query in the app
        st.write(query)

        # Show a spinner while generating the newsletter
        with st.spinner(f"Generating newsletter for {query}"):

            # Search the web for articles related to the query
            search_results = search_serp(query=query)

            # Extract the best URLs from the search results
            if search_results and 'organic' in search_results and len(search_results['organic']) > 0:
                urls = pick_best_articles_urls(response=search_results, query=query)
            else:
                st.error("No valid search results found.")


            # Extract content from the URLs
            data = extract_content_from_urls(urls)

            # Generate summaries for the extracted content
            summaries = summarizer(data, query)

            # Generate the newsletter thread using the summaries
            newsletter_thread = generate_newsletter(summaries, query)

            # Display search results in an expander
            with st.expander("Search Results"):
                st.info(search_results)

            # Display best URLs in an expander
            with st.expander("Best URLs"):
                st.info(urls)

            # Display extracted data in an expander
            with st.expander("Data"):
                # Fetch and display similarity search data from a FAISS database
                data_raw = " ".join(
                    d.page_content for d in data.similarity_search(query, k=4)
                )
                st.info(data_raw)

            # Display summaries in an expander
            with st.expander("Summaries"):
                st.info(summaries)

            # Display the generated newsletter thread in an expander
            with st.expander("Newsletter:"):
                st.info(newsletter_thread)

        # Display success message when the process is completed
        st.success("Done!")

# Invoke the main function when the script is run directly
if __name__ == "__main__":
    main()